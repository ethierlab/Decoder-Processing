import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

################################################################################
# 1) PARAMETERS
################################################################################
FILE_PATH   = 'Jango_dataset.pkl'
N           = 16   # number of PCA components
k           = 15   # lag length
hidden_dim  = 32
train_split = 0.75 # fraction of trials for training
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

################################################################################
# 2) HELPER FUNCTIONS
################################################################################
def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    if len(y_true_valid) < 2:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, Corr=np.nan, VAF=np.nan)
    
    # RMSE, MAE
    mse_val = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse_val)
    mae  = mean_absolute_error(y_true_valid, y_pred_valid)
    
    # R^2
    r2   = r2_score(y_true_valid, y_pred_valid)
    
    # Correlation
    corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1] if len(y_true_valid) > 1 else np.nan

    # --- Compute VAF (uncentered) ---
    num = np.sum((y_true_valid - y_pred_valid) ** 2)
    den = np.sum(y_true_valid ** 2)
    if den > 1e-12:
        vaf = 1.0 - (num / den)
    else:
        vaf = np.nan
    
    return dict(RMSE=rmse, MAE=mae, R2=r2, Corr=corr, VAF=vaf)

def average_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        out[k] = np.mean(vals) if len(vals) > 0 else np.nan
    return out

def create_lagged_data_linear_per_trial(X_trial, Y_trial, seq_len=16):
    """
    X_trial: (T_i, N), Y_trial: (T_i,)
    Returns X_lagged: (T_i - seq_len, seq_len*N), Y_lagged: (T_i - seq_len,)
    """
    T_i, N_ = X_trial.shape
    if T_i <= seq_len:
        return np.empty((0, seq_len*N_), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_lagged_list = []
    Y_lagged_list = []
    for t in range(seq_len, T_i):
        window = X_trial[t-seq_len:t, :].reshape(-1)
        X_lagged_list.append(window)
        Y_lagged_list.append(Y_trial[t])
    return np.array(X_lagged_list, dtype=np.float32), np.array(Y_lagged_list, dtype=np.float32)

def build_linear_dataset(X_trials, Y_trials, seq_len=16):
    """
    Builds a single large dataset of shape (sum_of_(T_i - seq_len), seq_len*N)
    across all trials for the linear approach.
    """
    X_list, Y_list = [], []
    for i in range(len(X_trials)):
        X_lag, Y_lag = create_lagged_data_linear_per_trial(X_trials[i], Y_trials[i], seq_len=seq_len)
        X_list.append(X_lag)
        Y_list.append(Y_lag)
    if len(X_list) > 0:
        X_cat = np.concatenate(X_list, axis=0)
        Y_cat = np.concatenate(Y_list, axis=0)
    else:
        # fallback if no data
        X_cat = np.empty((0, seq_len*X_trials[0].shape[1]))
        Y_cat = np.empty((0,))
    return X_cat, Y_cat

def get_trialwise_preds_rnn(model, X_trials, seq_len=16, device='cpu'):
    """
    Returns a list of length num_trials, each shape (T_i,) with np.nan for first seq_len steps.
    """
    model.eval()
    preds_by_trial = []
    with torch.no_grad():
        for i in range(X_trials.shape[0]):
            T_i = X_trials[i].shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            # Build lagged sequence for each time step
            X_lag = []
            for t in range(seq_len, T_i):
                X_lag.append(X_trials[i, t-seq_len:t, :])
            X_lag = np.array(X_lag, dtype=np.float32)

            X_lag_t = torch.tensor(X_lag, dtype=torch.float32, device=device)
            y_hat = model(X_lag_t).cpu().numpy().flatten()

            aligned = np.full((T_i,), np.nan)
            aligned[seq_len:] = y_hat
            preds_by_trial.append(aligned)
    return preds_by_trial

def get_trialwise_preds_linear(model, X_trials, Y_trials, seq_len=16, device='cpu'):
    """
    Per-trial approach for the linear model.
    """
    model.eval()
    preds_by_trial = []
    with torch.no_grad():
        for i in range(X_trials.shape[0]):
            T_i = X_trials[i].shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue
            X_lag_list = []
            for t in range(seq_len, T_i):
                window = X_trials[i, t-seq_len:t, :].reshape(-1)
                X_lag_list.append(window)
            X_lag = np.array(X_lag_list, dtype=np.float32)

            X_lag_t = torch.tensor(X_lag, dtype=torch.float32, device=device)
            y_hat = model(X_lag_t).cpu().numpy().flatten()

            aligned = np.full((T_i,), np.nan)
            aligned[seq_len:] = y_hat
            preds_by_trial.append(aligned)
    return preds_by_trial

# Helper for zeroing out selected channels
def zero_out_channels(X, channels_to_remove):
    """
    X: shape (num_trials, T, N)
    channels_to_remove: list of channel indices to zero out.
    Returns a copy of X with the given channels zeroed.
    """
    X_copy = X.copy()
    for ch_idx in channels_to_remove:
        X_copy[:, :, ch_idx] = 0.0
    return X_copy

################################################################################
# 3) LOAD DATA & PREPROCESS
################################################################################
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)

trial_keys = sorted(data['PCA'].keys())
num_trials_total = len(trial_keys)
print("Total trials:", num_trials_total)

# For RNN approach (num_trials, T, N)
X_rnn = np.stack([data['PCA'][k][:N].T for k in trial_keys])    # shape: (num_trials, T, N)
Y_rnn = np.stack([data['Force']['y'][k] for k in trial_keys])   # shape: (num_trials, T)

# (Optional) apply some filter if you had it in training. Left out for brevity.

# Z-score each trial individually
for i in range(len(X_rnn)):
    X_rnn[i] = zscore(X_rnn[i], axis=0)
    Y_rnn[i] = zscore(Y_rnn[i])

num_train_trials = int(num_trials_total * train_split)
X_train_rnn = X_rnn[:num_train_trials]
Y_train_rnn = Y_rnn[:num_train_trials]
X_test_rnn  = X_rnn[num_train_trials:]
Y_test_rnn  = Y_rnn[num_train_trials:]

# We'll build a linear "test" set in the same way
X_lin_test_cat, Y_lin_test_cat = build_linear_dataset(X_test_rnn, Y_test_rnn, seq_len=k)
X_lin_test_t  = torch.tensor(X_lin_test_cat, dtype=torch.float32, device=device)
Y_lin_test_t  = torch.tensor(Y_lin_test_cat,  dtype=torch.float32, device=device)

################################################################################
# 4) DEFINE MODELS (same as training) & LOAD WEIGHTS
################################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time-step
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        x = self.linear_hidden(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x

input_dim_rnn   = X_test_rnn.shape[2]  # = N=16
gru_model       = GRUModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)
lstm_model      = LSTMModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)
input_dim_linear= k * N  # =15*16=240
linear_model    = LinearLagModel(input_dim_linear, hidden_dim=hidden_dim).to(device)

# Load weights
gru_model.load_state_dict(torch.load("gru_weights.pth",  map_location=device))
lstm_model.load_state_dict(torch.load("lstm_weights.pth", map_location=device))
linear_model.load_state_dict(torch.load("linear_weights.pth", map_location=device))

gru_model.eval()
lstm_model.eval()
linear_model.eval()
print("\nModels loaded from disk. Ready for channel-removal testing.\n")

################################################################################
# 5) SYSTEMATIC CHANNEL REMOVAL (from end to beginning)
################################################################################
channel_counts = list(range(N, 0, -1))  # e.g. [16, 15, 14, ..., 1]

results_gru_systematic    = []
results_lstm_systematic   = []
results_linear_systematic = []

for n_keep in channel_counts:
    print(f"\n[Systematic] Keeping {n_keep} channels (zeroing out from {n_keep}..{N-1}).")
    channels_to_remove = list(range(n_keep, N))  # from n_keep up to N-1
    X_test_zeroed = zero_out_channels(X_test_rnn, channels_to_remove)

    # (A) RNN predictions
    gru_preds  = get_trialwise_preds_rnn(gru_model,  X_test_zeroed, seq_len=k, device=device)
    lstm_preds = get_trialwise_preds_rnn(lstm_model, X_test_zeroed, seq_len=k, device=device)

    # Evaluate GRU
    gru_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = gru_preds[i]
        m = compute_metrics(y_true_i, y_pred_i)
        gru_metrics_list.append(m)
    avg_gru = average_metrics(gru_metrics_list)
    results_gru_systematic.append((n_keep, avg_gru))

    # Evaluate LSTM
    lstm_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = lstm_preds[i]
        m = compute_metrics(y_true_i, y_pred_i)
        lstm_metrics_list.append(m)
    avg_lstm = average_metrics(lstm_metrics_list)
    results_lstm_systematic.append((n_keep, avg_lstm))

    # (B) Linear approach
    X_lin_list, Y_lin_list = build_linear_dataset(X_test_zeroed, Y_test_rnn, seq_len=k)
    X_lin_zeroed_t = torch.tensor(X_lin_list, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_lin_zeroed = linear_model(X_lin_zeroed_t).cpu().numpy().flatten()

    # Re-split predictions
    linear_preds_zeroed_by_trial = []
    idx_start = 0
    for i in range(X_test_zeroed.shape[0]):
        T_i = X_test_zeroed[i].shape[0]
        if T_i <= k:
            linear_preds_zeroed_by_trial.append(np.full((T_i,), np.nan))
            continue
        L_pred = T_i - k
        idx_end = idx_start + L_pred
        trial_preds = pred_lin_zeroed[idx_start:idx_end]
        idx_start = idx_end

        aligned = np.full((T_i,), np.nan)
        aligned[k:] = trial_preds
        linear_preds_zeroed_by_trial.append(aligned)

    lin_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = linear_preds_zeroed_by_trial[i]
        m = compute_metrics(y_true_i, y_pred_i)
        lin_metrics_list.append(m)
    avg_lin = average_metrics(lin_metrics_list)
    results_linear_systematic.append((n_keep, avg_lin))

    print("  GRU  : ", avg_gru)
    print("  LSTM : ", avg_lstm)
    print("  LIN  : ", avg_lin)

# Plot VAF vs. number of channels (Systematic)
count_vals_syst    = [r[0] for r in results_gru_systematic]
gru_vaf_syst       = [r[1]['VAF'] for r in results_gru_systematic]
lstm_vaf_syst      = [r[1]['VAF'] for r in results_lstm_systematic]
linear_vaf_syst    = [r[1]['VAF'] for r in results_linear_systematic]

plt.figure(figsize=(8,5))
plt.plot(count_vals_syst, gru_vaf_syst,    marker='o', label="GRU VAF")
plt.plot(count_vals_syst, lstm_vaf_syst,   marker='o', label="LSTM VAF")
plt.plot(count_vals_syst, linear_vaf_syst, marker='o', label="Linear VAF")
plt.title("Systematic Removal: VAF vs. # Channels Kept")
plt.xlabel("Channels Kept")
plt.ylabel("VAF")
plt.grid(True)
plt.legend()
plt.show()


################################################################################
# 6) RANDOM CHANNEL REMOVAL (multiple runs to see variation)
################################################################################
# Number of runs with different seeds
n_runs = 5

all_runs_gru = []
all_runs_lstm = []
all_runs_lin = []

for run in range(n_runs):
    np.random.seed(run)
    random_perm = np.random.permutation(N)  # e.g. [7, 1, 14, 0, 2, 13, ...]

    channels_removed = []
    
    results_gru_this_run    = []
    results_lstm_this_run   = []
    results_lin_this_run    = []

    # Evaluate once before any channel is removed:
    # (i.e. channels_removed = []) => no zero-out
    X_test_zeroed = zero_out_channels(X_test_rnn, channels_removed)
    gru_preds  = get_trialwise_preds_rnn(gru_model, X_test_zeroed, seq_len=k, device=device)
    lstm_preds = get_trialwise_preds_rnn(lstm_model, X_test_zeroed, seq_len=k, device=device)

    # GRU metrics
    gru_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = gru_preds[i]
        gru_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
    avg_gru_0 = average_metrics(gru_metrics_list)

    # LSTM metrics
    lstm_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = lstm_preds[i]
        lstm_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
    avg_lstm_0 = average_metrics(lstm_metrics_list)

    # Linear metrics
    X_lin_list, Y_lin_list = build_linear_dataset(X_test_zeroed, Y_test_rnn, seq_len=k)
    X_lin_zeroed_t = torch.tensor(X_lin_list, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_lin_zeroed = linear_model(X_lin_zeroed_t).cpu().numpy().flatten()
    linear_preds_zeroed_by_trial = []
    idx_start = 0
    for i in range(X_test_zeroed.shape[0]):
        T_i = X_test_zeroed[i].shape[0]
        if T_i <= k:
            linear_preds_zeroed_by_trial.append(np.full((T_i,), np.nan))
            continue
        L_pred = T_i - k
        idx_end = idx_start + L_pred
        trial_preds = pred_lin_zeroed[idx_start:idx_end]
        idx_start = idx_end
        aligned = np.full((T_i,), np.nan)
        aligned[k:] = trial_preds
        linear_preds_zeroed_by_trial.append(aligned)
    lin_metrics_list = []
    for i in range(len(X_test_zeroed)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = linear_preds_zeroed_by_trial[i]
        lin_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
    avg_lin_0 = average_metrics(lin_metrics_list)

    # Store initial (step=0)
    results_gru_this_run.append(avg_gru_0)
    results_lstm_this_run.append(avg_lstm_0)
    results_lin_this_run.append(avg_lin_0)

    # Now remove channels one at a time in random order
    for step in range(N):
        ch_to_remove = random_perm[step]
        channels_removed.append(ch_to_remove)

        X_test_zeroed = zero_out_channels(X_test_rnn, channels_removed)
        # RNN preds
        gru_preds  = get_trialwise_preds_rnn(gru_model,  X_test_zeroed, seq_len=k, device=device)
        lstm_preds = get_trialwise_preds_rnn(lstm_model, X_test_zeroed, seq_len=k, device=device)

        # Evaluate GRU
        gru_metrics_list = []
        for i in range(len(X_test_zeroed)):
            y_true_i = Y_test_rnn[i]
            y_pred_i = gru_preds[i]
            gru_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
        avg_gru = average_metrics(gru_metrics_list)
        
        # Evaluate LSTM
        lstm_metrics_list = []
        for i in range(len(X_test_zeroed)):
            y_true_i = Y_test_rnn[i]
            y_pred_i = lstm_preds[i]
            lstm_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
        avg_lstm = average_metrics(lstm_metrics_list)
        
        # Linear
        X_lin_list, Y_lin_list = build_linear_dataset(X_test_zeroed, Y_test_rnn, seq_len=k)
        X_lin_zeroed_t = torch.tensor(X_lin_list, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_lin_zeroed = linear_model(X_lin_zeroed_t).cpu().numpy().flatten()
        linear_preds_zeroed_by_trial = []
        idx_start = 0
        for i in range(X_test_zeroed.shape[0]):
            T_i = X_test_zeroed[i].shape[0]
            if T_i <= k:
                linear_preds_zeroed_by_trial.append(np.full((T_i,), np.nan))
                continue
            L_pred = T_i - k
            idx_end = idx_start + L_pred
            trial_preds = pred_lin_zeroed[idx_start:idx_end]
            idx_start = idx_end
            aligned = np.full((T_i,), np.nan)
            aligned[k:] = trial_preds
            linear_preds_zeroed_by_trial.append(aligned)
        lin_metrics_list = []
        for i in range(len(X_test_zeroed)):
            y_true_i = Y_test_rnn[i]
            y_pred_i = linear_preds_zeroed_by_trial[i]
            lin_metrics_list.append(compute_metrics(y_true_i, y_pred_i))
        avg_lin = average_metrics(lin_metrics_list)

        results_gru_this_run.append(avg_gru)
        results_lstm_this_run.append(avg_lstm)
        results_lin_this_run.append(avg_lin)

        print(f"[Random Removal | Run {run+1}] Step={step+1}, removed channel={ch_to_remove}")
        print("  GRU  : ", avg_gru)
        print("  LSTM : ", avg_lstm)
        print("  LIN  : ", avg_lin)

    # After one run is done, store in a bigger list
    all_runs_gru.append(results_gru_this_run)
    all_runs_lstm.append(results_lstm_this_run)
    all_runs_lin.append(results_lin_this_run)

# plot just one run or an average across runs. 
# Note that each run has length (N+1) (since we have step=0 plus step=1..N).
run_to_plot = 0  # plot the first run

gru_vaf_rand = [m['VAF'] for m in all_runs_gru[run_to_plot]]
lstm_vaf_rand = [m['VAF'] for m in all_runs_lstm[run_to_plot]]
lin_vaf_rand  = [m['VAF'] for m in all_runs_lin[run_to_plot]]

# x-axis is 0..N => how many channels removed so far
steps = list(range(len(gru_vaf_rand)))  # from 0..N
plt.figure(figsize=(8,5))
plt.plot(steps, gru_vaf_rand,   marker='o', label="GRU VAF (Run 1)")
plt.plot(steps, lstm_vaf_rand,  marker='o', label="LSTM VAF (Run 1)")
plt.plot(steps, lin_vaf_rand,   marker='o', label="Linear VAF (Run 1)")
plt.title("Random Removal (Single Run) - VAF vs. # Channels Removed")
plt.xlabel("# Channels Removed")
plt.ylabel("VAF")
plt.grid(True)
plt.legend()
plt.show()

# (Optional) If you want to show the average across all runs at each step:
n_steps = N + 1  # each run has N+1 steps
gru_vaf_all_runs = np.zeros((n_runs, n_steps))
lstm_vaf_all_runs = np.zeros((n_runs, n_steps))
lin_vaf_all_runs  = np.zeros((n_runs, n_steps))

for run_idx in range(n_runs):
    for step_idx in range(n_steps):
        gru_vaf_all_runs[run_idx, step_idx]  = all_runs_gru[run_idx][step_idx]['VAF']
        lstm_vaf_all_runs[run_idx, step_idx] = all_runs_lstm[run_idx][step_idx]['VAF']
        lin_vaf_all_runs[run_idx, step_idx]  = all_runs_lin[run_idx][step_idx]['VAF']

gru_mean = np.nanmean(gru_vaf_all_runs, axis=0)
lstm_mean = np.nanmean(lstm_vaf_all_runs, axis=0)
lin_mean  = np.nanmean(lin_vaf_all_runs, axis=0)

steps = np.arange(n_steps)  # 0..N
plt.figure(figsize=(8,5))
plt.plot(steps, gru_mean,   marker='o', label="GRU (Avg over runs)")
plt.plot(steps, lstm_mean,  marker='o', label="LSTM (Avg over runs)")
plt.plot(steps, lin_mean,   marker='o', label="Linear (Avg over runs)")
plt.title("Random Removal (Averaged) - VAF vs. # Channels Removed")
plt.xlabel("# Channels Removed")
plt.ylabel("VAF")
plt.grid(True)
plt.legend()
plt.show()