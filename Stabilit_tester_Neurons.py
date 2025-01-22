import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import random

###############################################################################
# 1) PARAMETERS
###############################################################################
SPIKERATE_PKL  = "spikeratedata.pkl"
FORCE_PKL      = "force.pkl"

# Remove channels (systematic or random) *before* smoothing/PCA
DO_SYSTEMATIC_REMOVAL = True
DO_RANDOM_REMOVAL     = True
N_RANDOM_RUNS         = 2
RANDOM_SEED           = 42

BIN_SIZE       = 0.05
SMOOTH_LEN     = 0.05
GAUSS_SIGMA    = (SMOOTH_LEN / BIN_SIZE) / 2
WINDOW_START   = -1.0
WINDOW_END     =  4.0
N_PCA_COMP     = 16  # number of PCA components after removal
# If fewer neurons remain than N_PCA_COMP, PCA uses min(remaining_neurons, N_PCA_COMP).

K_LAG          = 15  # how many time-lag steps for the linear/RNN models
TRAIN_SPLIT    = 0.75 # fraction of trials for "training" slice (even though we're just evaluating)
HIDDEN_DIM     = 32   # same dimension as your saved models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device = {DEVICE}")

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
def remove_neurons(original_spikerate, neurons_to_remove):
    """
    Returns a *new* spikeratedata dict with the specified (channel, neuron) removed.
    `neurons_to_remove` = set of (channel, neuron_name) pairs.
    """
    reduced_data = {}
    for channel, neurons_dict in original_spikerate.items():
        if channel == "Event time":
            # keep event times as-is
            reduced_data[channel] = neurons_dict
            continue
        reduced_data[channel] = {}
        for neuron_name, rate_array in neurons_dict.items():
            if (channel, neuron_name) not in neurons_to_remove:
                reduced_data[channel][neuron_name] = rate_array
    return reduced_data

def smooth_spikerate_data(spikeratedata, sigma):
    """
    Gaussian smooth each neuron's spike rate array.
    """
    smoothed = {}
    for channel, neurons in spikeratedata.items():
        if channel == "Event time":
            # just copy
            smoothed[channel] = neurons
            continue
        smoothed[channel] = {}
        for neuron_name, rates in neurons.items():
            smoothed[channel][neuron_name] = gaussian_filter1d(rates, sigma=sigma).astype(float)
    return smoothed

def flatten_spikerate(smoothed_spikerate):
    """
    Flatten spikerate dict into 2D array: shape (time_points, total_neurons).
    Returns (flat_array, list_of_neuron_ids).
    """
    arrays = []
    neuron_ids = []
    for channel, neurons_dict in smoothed_spikerate.items():
        if channel == "Event time":
            continue
        for neuron_name, rate_array in neurons_dict.items():
            arrays.append(rate_array)
            neuron_ids.append((channel, neuron_name))
    if len(arrays) == 0:
        # no neurons left
        return np.zeros((len(smoothed_spikerate["Event time"]), 0)), []
    # Stack horizontally => shape (time, #neurons)
    mat = np.stack(arrays, axis=-1)
    return mat, neuron_ids

def do_pca(data_2d, n_components=16):
    """
    data_2d: shape (time, #neurons).
    Returns PCA-projected data of shape (time, n_proj).
    """
    from sklearn.decomposition import PCA
    if data_2d.shape[1] == 0:
        return np.zeros((data_2d.shape[0], 0))
    n_comp = min(n_components, data_2d.shape[1])
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(data_2d)
    return proj

def extract_projected_data_per_trial(projected_data, event_times, bin_size, window_start, window_end):
    """
    projected_data: shape (time, dims)
    Return dict: trial_idx -> array of shape (dims, #common_time_bins)
                 (with linear interpolation onto a consistent time base)
    """
    common_times = np.arange(window_start, window_end, bin_size)
    trial_data_dict = {}
    
    total_time = projected_data.shape[0]
    absolute_times = np.arange(total_time) * bin_size

    for idx, t0 in enumerate(event_times):
        rel_times = absolute_times - t0
        in_window = (rel_times >= window_start) & (rel_times <= window_end)
        indices   = np.where(in_window)[0]

        if len(indices) == 0:
            # skip
            continue

        seg         = projected_data[indices, :]  # shape (num_in_window, dims)
        seg_times   = rel_times[indices]          # shape (num_in_window,)

        # Interpolate each dimension
        dims = seg.shape[1]
        interpolated = np.zeros((len(common_times), dims))
        for d in range(dims):
            vals_d = seg[:, d]
            # build 1D interpolator
            # fill_value="extrapolate" or you can do bounds_error=False
            from scipy.interpolate import interp1d
            f = interp1d(seg_times, vals_d, kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            interpolated[:, d] = f(common_times)
        
        # shape => (len(common_times), dims). We often want (dims, T)
        trial_data_dict[idx] = interpolated.T
    return trial_data_dict

def extract_force_per_trial(force, event_times, bin_size, window_start, window_end):
    """
    Return force_trials = {"x": dict, "y": dict} for each trial.
    We'll just slice the array indices. 
    If you need interpolation, do similarly to above.
    """
    force_trials = {"x": {}, "y": {}}
    
    fx_full = np.array(force["Force"]["x"])
    fy_full = np.array(force["Force"]["y"])

    for idx, t0 in enumerate(event_times):
        start_idx = int((t0 + window_start) / bin_size)
        end_idx   = int((t0 + window_end) / bin_size)

        # clip to within bounds
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(fx_full))

        # slice
        fx_seg = fx_full[start_idx:end_idx]
        fy_seg = fy_full[start_idx:end_idx]

        force_trials["x"][idx] = fx_seg
        force_trials["y"][idx] = fy_seg

    return force_trials

###############################################################################
# Functions to Build RNN/Linear Input from Trials
###############################################################################
def build_rnn_arrays(pca_trials, force_trials, n_components=16):
    """
    We want (num_trials, T, PCA_dims) for X,
            (num_trials, T) for Y
    pca_trials[trial_idx] => shape (dims, T_i)
    force_trials["y"][trial_idx] => 1D shape (T_i,)
    We'll zero-pad or just keep them 'ragged'?

    For simplicity, let's assume we keep them unpadded 
    and store them in a list => we'll do per-trial predictions.
    """
    trial_keys_sorted = sorted(pca_trials.keys())
    X_list = []
    Y_list = []
    for k in trial_keys_sorted:
        pc_data = pca_trials[k]   # shape (dims, T)
        force_y = force_trials["y"][k]  # shape (T,) if we used the same bin_size

        # Make sure they have the same time dimension
        # If mismatch, you can min() or interpolation. 
        T_pca = pc_data.shape[1]
        T_force = len(force_y)
        T_min = min(T_pca, T_force)

        # slice to T_min
        X_list.append(pc_data[:, :T_min].T)  # => shape (T_min, dims)
        Y_list.append(force_y[:T_min])
    # convert to arrays-of-arrays or keep them separate. We'll do list-of-arrays
    return X_list, Y_list, trial_keys_sorted

# From your earlier code:
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    if len(y_true_valid) < 2:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, Corr=np.nan, VAF=np.nan)

    mse_val = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse_val)
    mae  = mean_absolute_error(y_true_valid, y_pred_valid)
    r2   = r2_score(y_true_valid, y_pred_valid)
    corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1] if len(y_true_valid) > 1 else np.nan

    # VAF
    num = np.sum((y_true_valid - y_pred_valid) ** 2)
    den = np.sum(y_true_valid ** 2)
    vaf = 1.0 - (num/den) if den > 1e-12 else np.nan

    return dict(RMSE=rmse, MAE=mae, R2=r2, Corr=corr, VAF=vaf)

def average_metrics(list_of_dicts):
    """
    Averages each metric across the list, ignoring NaN.
    """
    if len(list_of_dicts) == 0:
        return {}
    out = {}
    keys = list_of_dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in list_of_dicts if not np.isnan(d[k])]
        if len(vals) == 0:
            out[k] = np.nan
        else:
            out[k] = np.mean(vals)
    return out

###############################################################################
# 3) DECODER DEFINITIONS & LOADING WEIGHTS
###############################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x shape: (batch, seq, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LinearLagModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.linear_hidden = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape: (batch, input_dim)
        # input_dim = K_LAG * #PCA comps
        x = self.linear_hidden(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x

# Load your previously saved state_dict
# (Adjust filenames to your local weights.)
GRU_WEIGHTS  = "gru_weights.pth"
LSTM_WEIGHTS = "lstm_weights.pth"
LIN_WEIGHTS  = "linear_weights.pth"

def load_models(pca_dim, hidden_dim=32, k_lag=15):
    # RNN input = pca_dim
    gru = GRUModel(input_size=pca_dim, hidden_size=hidden_dim).to(DEVICE)
    lstm= LSTMModel(input_size=pca_dim, hidden_size=hidden_dim).to(DEVICE)

    # Linear input = pca_dim * k_lag
    lin = LinearLagModel(input_dim=pca_dim*k_lag, hidden_dim=hidden_dim).to(DEVICE)

    gru.load_state_dict(torch.load(GRU_WEIGHTS,  map_location=DEVICE))
    lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=DEVICE))
    lin.load_state_dict(torch.load(LIN_WEIGHTS,  map_location=DEVICE))

    gru.eval(); lstm.eval(); lin.eval()
    return gru, lstm, lin

###############################################################################
# 4) PREDICTION HELPER
###############################################################################
def get_trialwise_preds_rnn(model, X_triallist, seq_len=15):
    """
    X_triallist: list of arrays, each shape (T, pca_dim)
    Returns: list of shape (num_trials), each 1D array shape (T,) with the first seq_len as NaN.
    """
    preds_by_trial = []
    model.eval()
    with torch.no_grad():
        for X in X_triallist:
            T_i = X.shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            # build a window of length seq_len for each time
            y_hat_list = []
            for t in range(seq_len, T_i):
                x_window = X[t-seq_len:t, :]  # shape (seq_len, pca_dim)
                x_ten = torch.tensor(x_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                out = model(x_ten)  # shape (1,1)
                y_hat_list.append(out.item())
            # align to timeline
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

def get_trialwise_preds_linear(model, X_triallist, seq_len=15):
    """
    X_triallist: list of arrays, each shape (T, pca_dim)
    We'll flatten the window of size seq_len -> (seq_len*pca_dim).
    """
    preds_by_trial = []
    model.eval()
    with torch.no_grad():
        for X in X_triallist:
            T_i = X.shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            y_hat_list = []
            for t in range(seq_len, T_i):
                x_window = X[t-seq_len:t, :].reshape(-1)
                x_ten = torch.tensor(x_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                out = model(x_ten)  # shape (1,1)
                y_hat_list.append(out.item())
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

###############################################################################
# 5) LOAD RAW DATA
###############################################################################
with open(SPIKERATE_PKL, "rb") as f:
    spikeratedata = pickle.load(f)

with open(FORCE_PKL, "rb") as f:
    force = pickle.load(f)

event_times = spikeratedata["Event time"]

###############################################################################
# 6) SYSTEMATIC REMOVAL (Optional)
###############################################################################
if DO_SYSTEMATIC_REMOVAL:
    # 6a) List out all neurons
    all_smooth = smooth_spikerate_data(spikeratedata, GAUSS_SIGMA)  # or remove first, then smooth if you prefer
    full_mat, all_neurons = flatten_spikerate(all_smooth)
    total_neurons = len(all_neurons)
    print(f"Total neurons: {total_neurons}")

    # We'll systematically keep n_keep = total_neurons..1
    results_systematic = []
    for n_keep in range(total_neurons, 0, -1):
        # define the remove set => remove everything except the first n_keep
        keep_set   = set(all_neurons[:n_keep])
        remove_set = set(all_neurons) - keep_set

        # Rebuild raw data with these removed
        spk_reduced = remove_neurons(spikeratedata, remove_set)
        # Smooth
        spk_reduced_sm = smooth_spikerate_data(spk_reduced, GAUSS_SIGMA)
        # Flatten
        mat_reduced, _ = flatten_spikerate(spk_reduced_sm)
        # PCA
        pca_reduced = do_pca(mat_reduced, n_components=N_PCA_COMP)  # shape (time, pca_dim<=N_PCA_COMP)
        pca_dim_actual = pca_reduced.shape[1]

        # Per-trial
        pca_trials = extract_projected_data_per_trial(
            pca_reduced,
            event_times,
            BIN_SIZE,
            WINDOW_START,
            WINDOW_END
        )
        force_trials = extract_force_per_trial(force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END)

        # Build lists for RNN
        X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials, n_components=pca_dim_actual)

        # Z-score each trial separately (optional) - do same for X and Y
        for i in range(len(X_list)):
            if X_list[i].shape[0] > 0:
                X_list[i] = zscore(X_list[i], axis=0)
                Y_list[i] = zscore(Y_list[i])

        # Split train/test
        num_trials = len(X_list)
        num_train  = int(num_trials * TRAIN_SPLIT)
        X_train_list = X_list[:num_train]
        Y_train_list = Y_list[:num_train]
        X_test_list  = X_list[num_train:]
        Y_test_list  = Y_list[num_train:]

        # Load your 3 decoders (all reloaded each iteration)
        gru, lstm, lin = load_models(pca_dim=pca_dim_actual, hidden_dim=HIDDEN_DIM, k_lag=K_LAG)

        # Evaluate on test set
        gru_preds  = get_trialwise_preds_rnn(gru,  X_test_list, seq_len=K_LAG)
        lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=K_LAG)
        lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=K_LAG)

        # Compute metrics
        gru_metrics_list = []
        lstm_metrics_list= []
        lin_metrics_list = []
        for i in range(len(X_test_list)):
            y_true = Y_test_list[i]
            yg = gru_preds[i]
            yl = lstm_preds[i]
            yn = lin_preds[i]
            gru_metrics_list.append(compute_metrics(y_true, yg))
            lstm_metrics_list.append(compute_metrics(y_true, yl))
            lin_metrics_list.append(compute_metrics(y_true, yn))

        avg_gru  = average_metrics(gru_metrics_list)
        avg_lstm = average_metrics(lstm_metrics_list)
        avg_lin  = average_metrics(lin_metrics_list)

        print(f"[Systematic] n_keep={n_keep} => GRU VAF={avg_gru['VAF']:.3f}, LSTM VAF={avg_lstm['VAF']:.3f}, LIN VAF={avg_lin['VAF']:.3f}")

        results_systematic.append((n_keep, avg_gru, avg_lstm, avg_lin))

###############################################################################
# 7) RANDOM REMOVAL (Optional)
###############################################################################
if DO_RANDOM_REMOVAL:
    random.seed(RANDOM_SEED)
    all_smooth = smooth_spikerate_data(spikeratedata, GAUSS_SIGMA)
    full_mat, all_neurons = flatten_spikerate(all_smooth)
    total_neurons = len(all_neurons)

    for run_i in range(N_RANDOM_RUNS):
        print(f"\n==== Random Removal - Run {run_i+1} ====")
        perm = np.random.permutation(total_neurons)
        removed_set = set()

        # Step 0: Evaluate with no removal
        spk_reduced = remove_neurons(spikeratedata, removed_set)
        spk_reduced_sm = smooth_spikerate_data(spk_reduced, GAUSS_SIGMA)
        mat_reduced, _ = flatten_spikerate(spk_reduced_sm)
        pca_reduced = do_pca(mat_reduced, n_components=N_PCA_COMP)
        pca_dim_actual = pca_reduced.shape[1]

        pca_trials = extract_projected_data_per_trial(
            pca_reduced,
            event_times,
            BIN_SIZE,
            WINDOW_START,
            WINDOW_END
        )
        force_trials = extract_force_per_trial(force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END)
        X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials, n_components=pca_dim_actual)

        # z-score
        for i in range(len(X_list)):
            if X_list[i].shape[0] > 0:
                X_list[i] = zscore(X_list[i], axis=0)
                Y_list[i] = zscore(Y_list[i])

        num_trials = len(X_list)
        num_train = int(num_trials * TRAIN_SPLIT)
        X_test_list = X_list[num_train:]
        Y_test_list = Y_list[num_train:]

        gru, lstm, lin = load_models(pca_dim=pca_dim_actual, hidden_dim=HIDDEN_DIM, k_lag=K_LAG)
        gru_preds  = get_trialwise_preds_rnn(gru,  X_test_list, seq_len=K_LAG)
        lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=K_LAG)
        lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=K_LAG)

        gru_mlist  = []
        lstm_mlist = []
        lin_mlist  = []
        for i in range(len(X_test_list)):
            y_true = Y_test_list[i]
            y_g = gru_preds[i]
            y_l = lstm_preds[i]
            y_n = lin_preds[i]
            gru_mlist.append(compute_metrics(y_true, y_g))
            lstm_mlist.append(compute_metrics(y_true, y_l))
            lin_mlist.append(compute_metrics(y_true, y_n))
        print(f"  Step=0, No removal => GRU VAF={average_metrics(gru_mlist)['VAF']:.3f} | "
              f"LSTM VAF={average_metrics(lstm_mlist)['VAF']:.3f} | "
              f"LIN VAF={average_metrics(lin_mlist)['VAF']:.3f}")

        # Now remove neurons one by one in perm order
        for step_idx, remove_idx in enumerate(perm):
            neuron_to_remove = all_neurons[remove_idx]
            removed_set.add(neuron_to_remove)

            # Re-run pipeline
            spk_reduced = remove_neurons(spikeratedata, removed_set)
            spk_reduced_sm = smooth_spikerate_data(spk_reduced, GAUSS_SIGMA)
            mat_reduced, _ = flatten_spikerate(spk_reduced_sm)
            pca_reduced = do_pca(mat_reduced, n_components=N_PCA_COMP)
            pca_dim_actual = pca_reduced.shape[1]

            pca_trials = extract_projected_data_per_trial(
                pca_reduced,
                event_times,
                BIN_SIZE,
                WINDOW_START,
                WINDOW_END
            )
            force_trials = extract_force_per_trial(force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END)
            X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials, n_components=pca_dim_actual)

            for i in range(len(X_list)):
                if X_list[i].shape[0] > 0:
                    X_list[i] = zscore(X_list[i], axis=0)
                    Y_list[i] = zscore(Y_list[i])

            num_trials = len(X_list)
            num_train = int(num_trials * TRAIN_SPLIT)
            X_test_list = X_list[num_train:]
            Y_test_list = Y_list[num_train:]

            # Load & eval
            gru, lstm, lin = load_models(pca_dim=pca_dim_actual, hidden_dim=HIDDEN_DIM, k_lag=K_LAG)
            gru_preds  = get_trialwise_preds_rnn(gru,  X_test_list, seq_len=K_LAG)
            lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=K_LAG)
            lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=K_LAG)

            gru_mlist  = []
            lstm_mlist = []
            lin_mlist  = []
            for i in range(len(X_test_list)):
                y_true = Y_test_list[i]
                y_g = gru_preds[i]
                y_l = lstm_preds[i]
                y_n = lin_preds[i]
                gru_mlist.append(compute_metrics(y_true, y_g))
                lstm_mlist.append(compute_metrics(y_true, y_l))
                lin_mlist.append(compute_metrics(y_true, y_n))

            avg_gru  = average_metrics(gru_mlist)
            avg_lstm = average_metrics(lstm_mlist)
            avg_lin  = average_metrics(lin_mlist)
            print(f"  Step={step_idx+1}, removed={neuron_to_remove} => "
                  f"GRU VAF={avg_gru['VAF']:.3f}, LSTM VAF={avg_lstm['VAF']:.3f}, LIN VAF={avg_lin['VAF']:.3f}")

print("\nAll done!")
