import numpy as np
import pickle
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import random

###############################################################################
# 1) GLOBAL PARAMETERS
###############################################################################
SPIKERATE_PKL        = "spikeratedata.pkl"
FORCE_PKL            = "force.pkl"

DO_SYSTEMATIC_NOISE  = True
DO_RANDOM_NOISE      = True
N_RANDOM_RUNS        = 5
RANDOM_SEED          = 42

BIN_SIZE             = 0.05
SMOOTH_LEN           = 0.05
GAUSS_SIGMA          = (SMOOTH_LEN / BIN_SIZE) / 2
WINDOW_START         = -1.0
WINDOW_END           =  4.0
TRAIN_SPLIT          = 0.75  # fraction of trials for "training" slice

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device = {DEVICE}")

###############################################################################
# 1.A) PER-DECODER CONFIG
###############################################################################
# Each decoder can have a different # of PCA components (N_PCA), hidden dim, lag, etc.
decoders_config = {
    "GRU": {
        "N_PCA":      14,
        "hidden_dim": 5,
        "k_lag":      16,
        "weights":    "gru_weights_5.pth"
    },
    "LSTM": {
        "N_PCA":      14,
        "hidden_dim": 55,
        "k_lag":      16,
        "weights":    "lstm_weights_55.pth"
    },
    "LIN": {
        "N_PCA":      14,
        "hidden_dim": 64,
        "k_lag":      16,
        "weights":    "linear_weights.pth"
    }
}

# Maximum N_PCA across all decoders to PCA once.
N_PCA_MAX = max(cfg["N_PCA"] for cfg in decoders_config.values())

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
def smooth_spikerate_data(spikeratedata, sigma):
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
    corr = np.nan
    if len(y_true_valid) > 1:
        corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1]

    num = np.sum((y_true_valid - y_pred_valid)**2)
    den = np.sum(y_true_valid**2)
    vaf = 1.0 - (num / den) if den > 1e-12 else np.nan
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

# Decoder architectures
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
# 4) LOADING AND INFERENCE HELPERS
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
    model = LinearLagModel(pca_dim * k_lag, hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def get_trialwise_preds_rnn(model, X_triallist, seq_len=15):
    preds_by_trial = []
    with torch.no_grad():
        for X in X_triallist:
            T_i = X.shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            y_hat_list = []
            for t in range(seq_len, T_i):
                x_window = X[t-seq_len:t, :]
                x_ten = torch.tensor(x_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                out = model(x_ten)
                y_hat_list.append(out.item())
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

def get_trialwise_preds_linear(model, X_triallist, seq_len=15):
    preds_by_trial = []
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
                out = model(x_ten)
                y_hat_list.append(out.item())
            arr = np.full((T_i,), np.nan)
            arr[seq_len:] = np.array(y_hat_list)
            preds_by_trial.append(arr)
    return preds_by_trial

###############################################################################
# 5) LOAD & PREPARE DATA (ONE-TIME PCA FIT)
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
num_channels = len(neuron_ids)
print("full_matrix shape:", full_matrix.shape)

# Fit PCA *one time* with the maximum dimension needed
from sklearn.decomposition import PCA
pca = PCA(n_components=N_PCA_MAX)
pca.fit(full_matrix)  # sets up your principal components

###############################################################################
# 6) NOISE INJECTION LOGIC
###############################################################################
def inject_noise_to_channels(data_matrix, channels_to_noisify, sigma=1.0):
    """
    data_matrix: shape (T, num_channels), original spikerate data
    channels_to_noisify: list or set of channel indices to add noise
    sigma: noise amplitude
    Returns a copy of data_matrix with noise added only to specified channels.
    """
    noisy_copy = data_matrix.copy()
    T_ = noisy_copy.shape[0]
    for ch in channels_to_noisify:
        noise = np.random.normal(loc=0.0, scale=sigma, size=T_)
        noisy_copy[:, ch] += noise
    return noisy_copy

def build_decoding_dataset(spk_noisy, pca_obj, decoders_cfg):
    """
    1) Apply PCA (with N_PCA_MAX).
    2) For each decoder config, slice the top n_pca dims, build trial data, do z-score.
    3) Return a dict: { decoder_name: (X_list_test, Y_list_test, <model>) }
       so we can easily compute predictions.

    The pipeline is:
      - Flatten spk_noisy => shape (T, num_channels)
      - pca.transform => shape (T, N_PCA_MAX)
      - slice => shape (T, n_pca)
      - extract trials => build_rnn_arrays => z-score => train/test split => build test sets
    """
    # 1) Flatten is already done: spk_noisy is shape (T, num_channels)
    #    so we just do PCA
    pca_all = pca_obj.transform(spk_noisy)  # (T, N_PCA_MAX)

    # 2) We'll build "pca_trials" using the same function as before
    #    This function re-splits the continuous data around event_times
    #    into trial segments
    #    We'll do this *once* at the full dimension, then slice later.
    #    Or we can just re-run for each decoder. We'll do it once at the FULL dimension,
    #    then slice for each trial. But the function doesn't trivially store the "full dimension."
    #    -> simpler approach: We'll do the "extract_projected_data_per_trial" for the full dimension, then slice in each trial array.

    # Actually let's do:
    pca_trials_fulldim = extract_projected_data_per_trial(
        pca_all, event_times, BIN_SIZE, WINDOW_START, WINDOW_END
    )
    force_trials = extract_force_per_trial(force, event_times, BIN_SIZE, WINDOW_START, WINDOW_END)

    out_dict = {}
    for decoder_name, cfg in decoders_cfg.items():
        n_pca      = cfg["N_PCA"]
        hidden_dim = cfg["hidden_dim"]
        k_lag      = cfg["k_lag"]
        weights    = cfg["weights"]

        # 2.a) Slice each trial's array to top n_pca
        # pca_trials_fulldim[trial] = (FULLdims, T) => so we slice axis=0 => [:n_pca, :]
        pca_trials_sliced = {}
        for tr_i, arr2d in pca_trials_fulldim.items():
            # arr2d shape = (FULLdims, T_i)
            arr_sliced = arr2d[:n_pca, :]
            pca_trials_sliced[tr_i] = arr_sliced

        # 2.b) Build rnn arrays
        X_list, Y_list, trial_keys = build_rnn_arrays(pca_trials_sliced, force_trials)
        # Z-score each trial
        for i in range(len(X_list)):
            if X_list[i].shape[0] > 0:
                X_list[i] = zscore(X_list[i], axis=0)
                Y_list[i] = zscore(Y_list[i])

        # 2.c) Train/test split
        num_trials = len(X_list)
        num_train = int(num_trials * TRAIN_SPLIT)
        X_test_list = X_list[num_train:]
        Y_test_list = Y_list[num_train:]

        # 2.d) Load model
        if decoder_name == "GRU":
            model = load_gru_model(n_pca, hidden_dim, weights)
        elif decoder_name == "LSTM":
            model = load_lstm_model(n_pca, hidden_dim, weights)
        else:
            model = load_linear_model(n_pca, hidden_dim, k_lag, weights)

        out_dict[decoder_name] = {
            "X_test_list": X_test_list,
            "Y_test_list": Y_test_list,
            "model": model
        }

    return out_dict

def compute_decoder_metrics(decoder_name, dec_cfg, X_test_list, Y_test_list, model):
    """Helper to compute trialwise predictions -> metrics."""
    k_lag = dec_cfg["k_lag"]
    if decoder_name in ["GRU", "LSTM"]:
        preds_list = get_trialwise_preds_rnn(model, X_test_list, seq_len=k_lag)
    else: # LIN
        preds_list = get_trialwise_preds_linear(model, X_test_list, seq_len=k_lag)

    mlist = []
    for i in range(len(X_test_list)):
        y_true = Y_test_list[i]
        y_hat  = preds_list[i]
        mlist.append(compute_metrics(y_true, y_hat))
    return average_metrics(mlist)

###############################################################################
# 7) SYSTEMATIC NOISE
###############################################################################
# Approach: define a set of sigma values, inject noise into *all* channels
# (or a chosen fixed subset), run decoders, measure performance.

systematic_noise_results = {
    decoder_name: []  # will store list of (sigma, metrics)
    for decoder_name in decoders_config.keys()
}

if DO_SYSTEMATIC_NOISE:
    noise_levels = [i for i in range(0,100)]  # for example
    # Optionally, pick a subset of channels. If you want *all* channels, do:
    subset_channels = range(num_channels)  # or pick random subset

    for sigma in noise_levels:
        # 1) Inject noise
        spk_noisy = inject_noise_to_channels(full_matrix, subset_channels, sigma=sigma)
        # 2) Build decoding dataset (apply PCA, etc.)
        dec_data_dict = build_decoding_dataset(spk_noisy, pca, decoders_config)
        # 3) For each decoder, compute metrics
        for decoder_name, cfg in decoders_config.items():
            X_test_list = dec_data_dict[decoder_name]["X_test_list"]
            Y_test_list = dec_data_dict[decoder_name]["Y_test_list"]
            model       = dec_data_dict[decoder_name]["model"]

            avg_m = compute_decoder_metrics(decoder_name, cfg, X_test_list, Y_test_list, model)
            systematic_noise_results[decoder_name].append((sigma, avg_m))

    # Print summary
    for decoder_name, data_list in systematic_noise_results.items():
        print(f"\n[Systematic Noise] Results for {decoder_name}:")
        for (sig, mm) in data_list:
            print(f"  sigma={sig}, VAF={mm['VAF']:.3f}, R2={mm['R2']:.3f}")

###############################################################################
# 8) RANDOM NOISE
###############################################################################
# Approach: We do N_RANDOM_RUNS. In each run:
#   - We randomly permute all channels.
#   - Step by step, we "add" one new channel to the noisy set, each with a fixed sigma.
#   - We measure performance at each step.
# This parallels "random removal," but we are "randomly adding noise" instead.

random_noise_results = {
    # dictionary: random_noise_results[decoder_name][run_i] = [ (step, n_noisy, metrics), ... ]
    decoder_name: {r: [] for r in range(N_RANDOM_RUNS)}
    for decoder_name in decoders_config.keys()
}

if DO_RANDOM_NOISE:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    total_neurons = num_channels
    fixed_sigma = 30  # use a single amplitude for the random approach

    for run_i in range(N_RANDOM_RUNS):
        print(f"\n==== Random Noise - Run {run_i+1} ====")
        perm = np.random.permutation(total_neurons)
        noisy_set = set()  # empty at start

        # Step 0: no channels noisy
        spk_noisy = inject_noise_to_channels(full_matrix, noisy_set, sigma=fixed_sigma)
        dec_data_dict = build_decoding_dataset(spk_noisy, pca, decoders_config)
        for decoder_name, cfg in decoders_config.items():
            # compute metrics
            X_test_list = dec_data_dict[decoder_name]["X_test_list"]
            Y_test_list = dec_data_dict[decoder_name]["Y_test_list"]
            model       = dec_data_dict[decoder_name]["model"]
            avg_m = compute_decoder_metrics(decoder_name, cfg, X_test_list, Y_test_list, model)
            random_noise_results[decoder_name][run_i].append((0, 0, avg_m))

        # Now iterate through channels in random order, adding them one by one
        for step_idx, ch_idx in enumerate(perm, start=1):
            noisy_set.add(ch_idx)
            spk_noisy = inject_noise_to_channels(full_matrix, noisy_set, sigma=fixed_sigma)
            dec_data_dict = build_decoding_dataset(spk_noisy, pca, decoders_config)

            for decoder_name, cfg in decoders_config.items():
                X_test_list = dec_data_dict[decoder_name]["X_test_list"]
                Y_test_list = dec_data_dict[decoder_name]["Y_test_list"]
                model       = dec_data_dict[decoder_name]["model"]
                avg_m = compute_decoder_metrics(decoder_name, cfg, X_test_list, Y_test_list, model)
                random_noise_results[decoder_name][run_i].append((step_idx, len(noisy_set), avg_m))

            if step_idx % 10 == 0:
                print(f"[Run={run_i+1}] step={step_idx}, #noisy={len(noisy_set)}")

###############################################################################
# 9) PLOT EXAMPLES
###############################################################################
# (A) Systematic Noise: single plot with lines for each decoder
if DO_SYSTEMATIC_NOISE:
    plt.figure(figsize=(7,5))
    for decoder_name, data_list in systematic_noise_results.items():
        # Sort by sigma ascending
        data_list.sort(key=lambda x: x[0])
        sig_vals = [d[0] for d in data_list]
        vaf_vals = [d[1]["VAF"] for d in data_list]
        plt.plot(range(sig_vals), vaf_vals, marker='o', label=decoder_name)

    plt.title("Systematic Noise: VAF vs. sigma (All Decoders)")
    plt.xlabel("sigma")
    plt.ylabel("VAF")
    plt.grid(True)
    plt.legend()
    plt.savefig('syst_noise.png',dpi=700)
    plt.show()

# (B) Random Noise: single plot with lines for each decoder (avg across runs)
if DO_RANDOM_NOISE:
    plt.figure(figsize=(7,5))
    for decoder_name in decoders_config.keys():
        all_runs_dict = random_noise_results[decoder_name]
        # find max step
        max_step = 0
        for run_i, run_data in all_runs_dict.items():
            if len(run_data) > 0:
                max_step_run = max(d[0] for d in run_data)
                max_step = max(max_step, max_step_run)

        n_steps = max_step + 1
        vaf_array = np.full((N_RANDOM_RUNS, n_steps), np.nan)

        for run_i in range(N_RANDOM_RUNS):
            run_data = all_runs_dict[run_i]  # list of (step, n_noisy, m_dict)
            for (step_i, n_noisy, mm) in run_data:
                if mm is not None and step_i < n_steps:
                    vaf_array[run_i, step_i] = mm["VAF"]

        mean_vaf = np.nanmean(vaf_array, axis=0)
        plt.plot(range(n_steps), mean_vaf, marker='o', label=decoder_name)

    plt.title("Random Noise: VAF vs. # of Channels Noisy (All Decoders, Mean Runs)")
    plt.xlabel("Removal Step (channels added to noise)")
    plt.ylabel("VAF")
    plt.grid(True)
    plt.legend()
    plt.savefig('random_noise.png',dpi=700)
    plt.show()