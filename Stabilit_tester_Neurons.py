import numpy as np
import pickle
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import random

###############################################################################
# 1) PARAMETERS
###############################################################################
SPIKERATE_PKL        = "spikeratedata.pkl"
FORCE_PKL            = "force.pkl"

DO_SYSTEMATIC_REMOVAL= True
DO_RANDOM_REMOVAL    = False
N_RANDOM_RUNS        = 2
RANDOM_SEED          = 42

BIN_SIZE             = 0.05
SMOOTH_LEN           = 0.05
GAUSS_SIGMA          = (SMOOTH_LEN / BIN_SIZE) / 2
WINDOW_START         = -1.0
WINDOW_END           =  4.0
N_PCA_COMP           = 14   # The fixed PCA dimension required by decoders
TRAIN_SPLIT          = 0.75 # fraction of trials for "training" slice
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device = {DEVICE}")

# ----------------------------------------------------------------------
# 1.A) PER-DECODER PARAMETERS
# ----------------------------------------------------------------------
# Suppose we want different hidden sizes / lag parameters / weight files
# for each decoder. You can keep them in one place, like a dictionary.

# GRU parameters
N_PCA_COMP     = 14 
GRU_HIDDEN_DIM = 5
GRU_K_LAG      = 16
GRU_WEIGHTS    = "gru_weights_5.pth"

# LSTM parameters
LSTM_HIDDEN_DIM= 55
LSTM_K_LAG     = 16
LSTM_WEIGHTS   = "lstm_weights_55.pth"

# Linear-lag parameters
LIN_HIDDEN_DIM = 64
LIN_K_LAG      = 16
LIN_WEIGHTS    = "linear_weights.pth"


###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
def smooth_spikerate_data(spikeratedata, sigma):
    """Applies a Gaussian filter (sigma = sigma) to each neuron's spikerate."""
    from scipy.ndimage import gaussian_filter1d
    smoothed = {}
    for channel, neurons in spikeratedata.items():
        if channel == "Event time":
            smoothed[channel] = neurons
            continue
        smoothed[channel] = {}
        for neuron_name, rates in neurons.items():
            smoothed[channel][neuron_name] = gaussian_filter1d(rates, sigma=sigma).astype(float)
    return smoothed

def flatten_spikerate(smoothed_spikerate):
    """
    Flattens the spikerate dict into a matrix of shape (time, #neurons).
    Also returns a list of (channel, neuron_name) in the same order as columns.
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
        T_len = len(smoothed_spikerate["Event time"]) if "Event time" in smoothed_spikerate else 0
        return np.zeros((T_len, 0)), []
    mat = np.stack(arrays, axis=-1)  # shape (time, #neurons)
    return mat, neuron_ids

def extract_projected_data_per_trial(projected_data, event_times, bin_size, window_start, window_end):
    """
    Given a 2D array (time, PCA_dims) and a list of event_times,
    extracts trial segments in [window_start, window_end] around each event,
    and returns {trial_idx: (PCA_dims, time_bins_for_that_trial)} as a dict.
    """
    from scipy.interpolate import interp1d
    common_times = np.arange(window_start, window_end, bin_size)
    trial_data_dict = {}

    total_time = projected_data.shape[0]
    absolute_times = np.arange(total_time) * bin_size

    for idx, t0 in enumerate(event_times):
        rel_times = absolute_times - t0
        in_window = (rel_times >= window_start) & (rel_times <= window_end)
        indices   = np.where(in_window)[0]

        if len(indices) == 0:
            continue

        seg       = projected_data[indices, :]  # shape (num_in_window, dims)
        seg_times = rel_times[indices]

        dims = seg.shape[1]
        interpolated = np.zeros((len(common_times), dims))
        for d in range(dims):
            vals_d = seg[:, d]
            f = interp1d(seg_times, vals_d, kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            interpolated[:, d] = f(common_times)
        
        # shape => (len(common_times), dims) => transpose => (dims, T)
        trial_data_dict[idx] = interpolated.T
    return trial_data_dict

def extract_force_per_trial(force, event_times, bin_size, window_start, window_end):
    """
    Returns a dict with force x,y for each trial.
    e.g. force_trials["x"][trial_idx] and force_trials["y"][trial_idx].
    """
    force_trials = {"x": {}, "y": {}}
    fx_full = np.array(force["Force"]["x"])
    fy_full = np.array(force["Force"]["y"])

    for idx, t0 in enumerate(event_times):
        start_idx = int((t0 + window_start) / bin_size)
        end_idx   = int((t0 + window_end) / bin_size)
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(fx_full))

        fx_seg = fx_full[start_idx:end_idx]
        fy_seg = fy_full[start_idx:end_idx]

        force_trials["x"][idx] = fx_seg
        force_trials["y"][idx] = fy_seg
    return force_trials

def build_rnn_arrays(pca_trials, force_trials):
    """
    For each trial, obtains X (PC dims) and Y (force y-dim) arrays.
    Returns X_list, Y_list, trial_keys_sorted.
    """
    trial_keys_sorted = sorted(pca_trials.keys())
    X_list = []
    Y_list = []
    for k in trial_keys_sorted:
        pc_data = pca_trials[k]         # shape (dims, T_k)
        force_y = force_trials["y"][k]  # shape (T_k,)

        T_pca = pc_data.shape[1]
        T_force = force_y.shape[0]
        T_min = min(T_pca, T_force)

        X_list.append(pc_data[:, :T_min].T)  # => (T_min, dims)
        Y_list.append(force_y[:T_min])
    return X_list, Y_list, trial_keys_sorted

###############################################################################
# 3) METRICS + MODELS
###############################################################################
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
    num = np.sum((y_true_valid - y_pred_valid)**2)
    den = np.sum(y_true_valid**2)
    vaf = 1 - (num/den) if den > 1e-12 else np.nan
    return dict(RMSE=rmse, MAE=mae, R2=r2, Corr=corr, VAF=vaf)

def average_metrics(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    import numpy as np
    out = {}
    keys = list(list_of_dicts[0].keys())
    for k in keys:
        vals = [d[k] for d in list_of_dicts if not np.isnan(d[k])]
        out[k] = np.mean(vals) if len(vals) > 0 else np.nan
    return out

# --- Decoder architectures ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

class LinearLagModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        input_dim = number_of_pca_dims * k_lag
        """
        super().__init__()
        self.linear_hidden = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x

###############################################################################
# 4) LOADING AND INFERENCE HELPERS (Using Per-Decoder Hyperparams)
###############################################################################
def load_gru_model(pca_dim, hidden_dim, weights_path):
    model = GRUModel(pca_dim, hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def load_lstm_model(pca_dim, hidden_dim, weights_path):
    model = LSTMModel(pca_dim, hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def load_linear_model(pca_dim, hidden_dim, k_lag, weights_path):
    """
    pca_dim => dimension of the PCA-transformed data
    hidden_dim => dimension of the linear hidden layer
    k_lag => how many time steps to flatten
    """
    model = LinearLagModel(pca_dim * k_lag, hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def get_trialwise_preds_rnn(model, X_triallist, seq_len=15):
    """
    For either GRU or LSTM (any RNN that processes a sliding window),
    we feed in frames of length 'seq_len' and output the final step.
    """
    preds_by_trial = []
    with torch.no_grad():
        for X in X_triallist:
            T_i = X.shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            y_hat_list = []
            for t in range(seq_len, T_i):
                x_window = X[t-seq_len:t, :]    # shape (seq_len, pca_dim)
                x_ten = torch.tensor(x_window, dtype=torch.float32, device=DEVICE).unsqueeze(0) 
                out = model(x_ten)
                y_hat_list.append(out.item())
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

def get_trialwise_preds_linear(model, X_triallist, seq_len=15):
    """
    For the linear-lag model, we flatten the last 'seq_len' frames
    (i.e. shape (seq_len * pca_dim,)) into a single vector and pass to the network.
    """
    preds_by_trial = []
    with torch.no_grad():
        for X in X_triallist:
            T_i = X.shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            y_hat_list = []
            for t in range(seq_len, T_i):
                x_window = X[t-seq_len:t, :].reshape(-1)  # flatten
                x_ten = torch.tensor(x_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                out = model(x_ten)
                y_hat_list.append(out.item())
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

###############################################################################
# 5) LOAD & PREPARE DATA (ONE-TIME PCA)
###############################################################################
with open(SPIKERATE_PKL, "rb") as f:
    spikeratedata = pickle.load(f)
with open(FORCE_PKL, "rb") as f:
    force = pickle.load(f)

event_times = spikeratedata["Event time"]

# Smooth once
spk_sm = smooth_spikerate_data(spikeratedata, GAUSS_SIGMA)
# Flatten once
full_matrix, neuron_ids = flatten_spikerate(spk_sm)  # shape (T, total_neurons)

# Fit PCA *one time* on the entire data
from sklearn.decomposition import PCA
pca = PCA(n_components=N_PCA_COMP)
pca.fit(full_matrix)  # This sets up your principal components

###############################################################################
# 6) HELPER: ZERO-OUT NEURONS
###############################################################################
def zero_out_neurons(spk_full, neuron_ids_list, neurons_to_remove):
    """
    Returns a *copy* of spk_full (shape (T, #neurons)) but sets columns
    for 'neurons_to_remove' to zero.
    `neurons_to_remove` is a set of (channel, neuron_name).
    """
    spk_copy = spk_full.copy()
    for i, neuron_id in enumerate(neuron_ids_list):
        if neuron_id in neurons_to_remove:
            spk_copy[:, i] = 0.0
    return spk_copy

###############################################################################
# 7) SYSTEMATIC REMOVAL (USING ZERO-OUT)
###############################################################################
results_systematic = []

if DO_SYSTEMATIC_REMOVAL:
    all_channels = neuron_ids[:]  # a list of (channel, neuron)
    total_neurons = len(all_channels)
    print("Total neurons found:", total_neurons)

    # We'll remove neurons from the *end* to the *start* in steps
    for n_keep in range(total_neurons, 0, -1):
        if n_keep < N_PCA_COMP:
            print(f"[Systematic] n_removed={n_keep} < required PCA dim={N_PCA_COMP}. Stopping.")
            break
        rem_neuron = total_neurons - n_keep
        keep_set   = set(all_channels[:n_keep])
        remove_set = set(all_channels[n_keep:])

        # Zero out spikerates for removed neurons
        spk_zeroed = zero_out_neurons(full_matrix, neuron_ids, remove_set)

        # Apply the same PCA
        pca_reduced = pca.transform(spk_zeroed)  # shape (T, N_PCA_COMP)
        pca_dim_actual = pca_reduced.shape[1]

        if pca_dim_actual < N_PCA_COMP:
            print(f"[Systematic] pca_dim={pca_dim_actual} < {N_PCA_COMP}. Stopping.")
            break

        # Trials
        pca_trials   = extract_projected_data_per_trial(
            pca_reduced, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
        )
        force_trials = extract_force_per_trial(
            force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
        )
        X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials)

        # Z-score
        for i in range(len(X_list)):
            if X_list[i].shape[0] > 0:
                X_list[i] = zscore(X_list[i], axis=0)
                Y_list[i] = zscore(Y_list[i])

        # train/test split
        num_trials = len(X_list)
        num_train = int(num_trials * TRAIN_SPLIT)
        X_test_list = X_list[num_train:]
        Y_test_list = Y_list[num_train:]

        # 7A) Load GRU
        gru = load_gru_model(pca_dim_actual, GRU_HIDDEN_DIM, GRU_WEIGHTS)
        gru_preds  = get_trialwise_preds_rnn(gru, X_test_list, seq_len=GRU_K_LAG)

        # 7B) Load LSTM
        lstm = load_lstm_model(pca_dim_actual, LSTM_HIDDEN_DIM, LSTM_WEIGHTS)
        lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=LSTM_K_LAG)

        # 7C) Load Linear
        lin = load_linear_model(pca_dim_actual, LIN_HIDDEN_DIM, LIN_K_LAG, LIN_WEIGHTS)
        lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=LIN_K_LAG)

        # metrics
        gru_mlist, lstm_mlist, lin_mlist = [], [], []
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

        print(f"[Systematic] N_remove={rem_neuron}, "
              f"GRU VAF={avg_gru['VAF']:.3f}, LSTM={avg_lstm['VAF']:.3f}, LIN={avg_lin['VAF']:.3f}")

        results_systematic.append((rem_neuron, avg_gru, avg_lstm, avg_lin))

###############################################################################
# 8) RANDOM REMOVAL (USING ZERO-OUT)
###############################################################################
random_results = {}

if DO_RANDOM_REMOVAL:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_channels = neuron_ids[:]
    total_neurons = len(all_channels)

    for run_i in range(N_RANDOM_RUNS):
        print(f"\n==== Random Removal - Run {run_i+1} ====")
        perm = np.random.permutation(total_neurons)
        removed_set = set()
        run_data_list = []

        # Step 0: no removal
        n_kept = total_neurons
        if n_kept < N_PCA_COMP:
            print(f"[Random run={run_i+1}] n_kept={n_kept} < {N_PCA_COMP}, stopping immediately.")
            random_results[run_i] = run_data_list
            continue

        spk_zeroed = zero_out_neurons(full_matrix, neuron_ids, removed_set)
        pca_reduced = pca.transform(spk_zeroed)
        pca_dim_actual = pca_reduced.shape[1]

        if pca_dim_actual < N_PCA_COMP:
            print(f"[Random run={run_i+1}] step=0 => pca_dim={pca_dim_actual}< {N_PCA_COMP}. Stop.")
            random_results[run_i] = run_data_list
            continue

        pca_trials   = extract_projected_data_per_trial(
            pca_reduced, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
        )
        force_trials = extract_force_per_trial(
            force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
        )
        X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials)

        for i in range(len(X_list)):
            if X_list[i].shape[0] > 0:
                X_list[i] = zscore(X_list[i], axis=0)
                Y_list[i] = zscore(Y_list[i])

        num_trials = len(X_list)
        num_train = int(num_trials * TRAIN_SPLIT)
        X_test_list = X_list[num_train:]
        Y_test_list = Y_list[num_train:]

        # Load each model with appropriate parameters
        gru = load_gru_model(pca_dim_actual, GRU_HIDDEN_DIM, GRU_WEIGHTS)
        lstm= load_lstm_model(pca_dim_actual, LSTM_HIDDEN_DIM, LSTM_WEIGHTS)
        lin = load_linear_model(pca_dim_actual, LIN_HIDDEN_DIM, LIN_K_LAG, LIN_WEIGHTS)

        gru_preds  = get_trialwise_preds_rnn(gru,  X_test_list, seq_len=GRU_K_LAG)
        lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=LSTM_K_LAG)
        lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=LIN_K_LAG)

        gru_mlist, lstm_mlist, lin_mlist = [], [], []
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
        run_data_list.append((0, total_neurons, None, avg_gru, avg_lstm, avg_lin))

        print(f"[Random run={run_i+1}] step=0 => "
              f"GRU VAF={avg_gru['VAF']:.3f}, LSTM={avg_lstm['VAF']:.3f}, LIN={avg_lin['VAF']:.3f}")

        # Now remove channels in random order
        for step_idx, remove_idx in enumerate(perm, start=1):
            removed_neuron = all_channels[remove_idx]
            removed_set.add(removed_neuron)
            n_kept = total_neurons - step_idx

            if n_kept < N_PCA_COMP:
                print(f"[Random run={run_i+1}] step={step_idx}: n_kept={n_kept} < {N_PCA_COMP}, stopping.")
                break

            spk_zeroed = zero_out_neurons(full_matrix, neuron_ids, removed_set)
            pca_reduced = pca.transform(spk_zeroed)
            pca_dim_actual = pca_reduced.shape[1]

            if pca_dim_actual < N_PCA_COMP:
                print(f"[Random run={run_i+1}] step={step_idx}, pca_dim={pca_dim_actual} < {N_PCA_COMP}, stopping.")
                break

            pca_trials   = extract_projected_data_per_trial(
                pca_reduced, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
            )
            force_trials = extract_force_per_trial(
                force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
            )
            X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials, force_trials)

            for i in range(len(X_list)):
                if X_list[i].shape[0] > 0:
                    X_list[i] = zscore(X_list[i], axis=0)
                    Y_list[i] = zscore(Y_list[i])

            num_trials = len(X_list)
            num_train = int(num_trials * TRAIN_SPLIT)
            X_test_list = X_list[num_train:]
            Y_test_list = Y_list[num_train:]

            # Load each model again (with the same param set, but new input dimension)
            gru = load_gru_model(pca_dim_actual, GRU_HIDDEN_DIM, GRU_WEIGHTS)
            lstm= load_lstm_model(pca_dim_actual, LSTM_HIDDEN_DIM, LSTM_WEIGHTS)
            lin = load_linear_model(pca_dim_actual, LIN_HIDDEN_DIM, LIN_K_LAG, LIN_WEIGHTS)

            gru_preds  = get_trialwise_preds_rnn(gru,  X_test_list, seq_len=GRU_K_LAG)
            lstm_preds = get_trialwise_preds_rnn(lstm, X_test_list, seq_len=LSTM_K_LAG)
            lin_preds  = get_trialwise_preds_linear(lin, X_test_list, seq_len=LIN_K_LAG)

            gru_mlist, lstm_mlist, lin_mlist = [], [], []
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
            run_data_list.append((step_idx, n_kept, removed_neuron, avg_gru, avg_lstm, avg_lin))

            print(f"[Random run={run_i+1}] step={step_idx} => "
                  f"GRU VAF={avg_gru['VAF']:.3f}, LSTM={avg_lstm['VAF']:.3f}, LIN={avg_lin['VAF']:.3f}")

        random_results[run_i] = run_data_list

###############################################################################
# 9) PLOT RESULTS
###############################################################################
# SYSTEMATIC
if DO_SYSTEMATIC_REMOVAL and len(results_systematic) > 0:
    results_systematic.sort(key=lambda x: x[0])  # ensure ascending order
    n_keep_vals  = [r[0] for r in results_systematic]
    gru_vaf_vals = [r[1]['VAF'] for r in results_systematic]
    lstm_vaf_vals= [r[2]['VAF'] for r in results_systematic]
    lin_vaf_vals = [r[3]['VAF'] for r in results_systematic]

    plt.figure(figsize=(8,5))
    plt.plot(n_keep_vals, gru_vaf_vals,  'o-', label='GRU')
    plt.plot(n_keep_vals, lstm_vaf_vals, 'o-', label='LSTM')
    plt.plot(n_keep_vals, lin_vaf_vals,  'o-', label='Linear')
    plt.xlabel('Number of Neurons Kept')
    plt.ylabel('VAF')
    plt.title('Systematic Removal: VAF vs. # Neurons Removed')
    plt.grid(True)
    plt.legend()
    plt.savefig('sys_remov_neuron.png',dpi=700)
    plt.show()

# RANDOM
if DO_RANDOM_REMOVAL and len(random_results) > 0:
    # 1) Plot each run separately
    for run_i, run_data in random_results.items():
        if len(run_data) == 0:
            continue
        run_data_sorted = sorted(run_data, key=lambda x: x[0])  # sort by step
        steps     = [d[0] for d in run_data_sorted]
        gru_vaf   = [d[3]['VAF'] for d in run_data_sorted]
        lstm_vaf  = [d[4]['VAF'] for d in run_data_sorted]
        lin_vaf   = [d[5]['VAF'] for d in run_data_sorted]

        plt.figure(figsize=(8,5))
        plt.plot(steps, gru_vaf,  'o-', label='GRU')
        plt.plot(steps, lstm_vaf, 'o-', label='LSTM')
        plt.plot(steps, lin_vaf,  'o-', label='Linear')
        plt.xlabel('Number of channels removed')
        plt.ylabel('VAF')
        plt.title(f'Random Removal - Run {run_i+1}')
        plt.grid(True)
        plt.legend()
        # plt.savefig('rand_remov_neuron.png',dpi=700)
        plt.show()

    # 2) (Optional) Plot the average across runs
    import numpy as np
    max_step = 0
    for run_i, run_data in random_results.items():
        if len(run_data) > 0:
            max_step_run = max(d[0] for d in run_data)
            if max_step_run > max_step:
                max_step = max_step_run

    n_steps = max_step + 1
    all_runs_gru  = np.full((N_RANDOM_RUNS, n_steps), np.nan)
    all_runs_lstm = np.full((N_RANDOM_RUNS, n_steps), np.nan)
    all_runs_lin  = np.full((N_RANDOM_RUNS, n_steps), np.nan)

    for run_i, run_data in random_results.items():
        for item in run_data:
            step_i = item[0]
            avg_gru  = item[3]
            avg_lstm = item[4]
            avg_lin  = item[5]
            if step_i < n_steps:
                all_runs_gru[run_i, step_i]  = avg_gru['VAF']
                all_runs_lstm[run_i, step_i] = avg_lstm['VAF']
                all_runs_lin[run_i, step_i]  = avg_lin['VAF']

    gru_mean  = np.nanmean(all_runs_gru, axis=0)
    lstm_mean = np.nanmean(all_runs_lstm, axis=0)
    lin_mean  = np.nanmean(all_runs_lin, axis=0)

    steps_axis = np.arange(n_steps)
    plt.figure(figsize=(8,5))
    plt.plot(steps_axis, gru_mean,  'o-', label='GRU (mean)')
    plt.plot(steps_axis, lstm_mean, 'o-', label='LSTM (mean)')
    plt.plot(steps_axis, lin_mean,  'o-', label='Linear (mean)')
    plt.xlabel('Number of channels removed)')
    plt.ylabel('VAF')
    plt.title('Random Removal (Avg Across Runs)')
    plt.grid(True)
    plt.legend()
    plt.savefig('rand_remov_neuron.png',dpi=700)
    plt.show()
