import numpy as np
import torch
import torch.nn as nn
import pickle
from scipy.stats import zscore
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

################################################################################
# 1) PARAMETERS (must match training)
################################################################################
FILE_PATH   = 'Jango_dataset.pkl'
N           = 16         # number of PCA components
k           = 15         # lag length
hidden_dim  = 32
train_split = 0.75       # fraction of trials for training
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

################################################################################
# 2) HELPER FUNCTIONS (same as training/test code)
################################################################################
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, filt_order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filt_order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_lagged_data_rnn(X, Y, seq_len=16):
    """
    For RNN-based approach (GRU or LSTM).
    X: (num_trials, T, N)
    Y: (num_trials, T)
    Returns X_lagged, Y_lagged:
      - X_lagged: (total_windows, seq_len, N)
      - Y_lagged: (total_windows,)
    """
    X_lagged, Y_lagged = [], []
    num_trials = X.shape[0]
    for i in range(num_trials):
        T_i = X[i].shape[0]
        for t in range(seq_len, T_i):
            X_lagged.append(X[i, t-seq_len:t, :])
            Y_lagged.append(Y[i, t])
    return np.array(X_lagged), np.array(Y_lagged)

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

################################################################################
# 3) LOAD DATA & PREPROCESS (Same as training, to isolate the same test set)
################################################################################
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)

trial_keys = sorted(data['PCA'].keys())
num_trials_total = len(trial_keys)
print("Total trials:", num_trials_total)

# For RNN approach (num_trials, T, N)
X_rnn = np.stack([data['PCA'][k][:N].T for k in trial_keys])    # shape: (num_trials, T, N)
Y_rnn = np.stack([data['Force']['y'][k] for k in trial_keys])   # shape: (num_trials, T)

# Optional filter, if used in training:
# for i in range(len(Y_rnn)):
#     Y_rnn[i] = apply_lowpass_filter(Y_rnn[i], cutoff_freq=10, sampling_rate=1000, filt_order=5)

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
# 4) DEFINE MODELS (identical to training) & LOAD WEIGHTS
################################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
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
        # Must match training code naming
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
print("\nModels loaded from disk. Ready for noise testing.\n")

################################################################################
# 5) NOISE-ROBUSTNESS TEST (Vary sigma, measure metrics)
################################################################################
def add_gaussian_noise(X, sigma=0.0):
    """Adds N(0, sigma^2) noise to X. Works for 3D (num_trials, T, N)."""
    noise = np.random.normal(0, sigma, size=X.shape)
    return X + noise

# We'll test these sigma values
sigmas = np.arange(0, 5, 0.05).tolist()

results_gru    = []
results_lstm   = []
results_linear = []

for sigma in sigmas:
    print(f"--- Testing with sigma={sigma} ---")

    # ==== (A) RNN Noise Injection ====
    X_test_rnn_noisy = add_gaussian_noise(X_test_rnn, sigma)
    # Predict with GRU
    gru_preds_noisy  = get_trialwise_preds_rnn(gru_model,  X_test_rnn_noisy, seq_len=k, device=device)
    # Predict with LSTM
    lstm_preds_noisy = get_trialwise_preds_rnn(lstm_model, X_test_rnn_noisy, seq_len=k, device=device)

    # Evaluate metrics for GRU
    gru_metrics_list = []
    for i in range(len(X_test_rnn_noisy)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = gru_preds_noisy[i]
        m = compute_metrics(y_true_i, y_pred_i)
        gru_metrics_list.append(m)
    avg_gru = average_metrics(gru_metrics_list)
    results_gru.append((sigma, avg_gru))

    # Evaluate metrics for LSTM
    lstm_metrics_list = []
    for i in range(len(X_test_rnn_noisy)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = lstm_preds_noisy[i]
        m = compute_metrics(y_true_i, y_pred_i)
        lstm_metrics_list.append(m)
    avg_lstm = average_metrics(lstm_metrics_list)
    results_lstm.append((sigma, avg_lstm))

    # ==== (B) Linear Noise Injection ====
    # For linear approach, we must add noise to the original test input shape: (num_trials, T, N)
    # then rebuild the lagged dataset
    X_test_linear_noisy = add_gaussian_noise(X_test_rnn, sigma)
    # Convert to big cat
    X_lin_list, Y_lin_list = build_linear_dataset(X_test_linear_noisy, Y_test_rnn, seq_len=k)
    X_lin_noisy_t = torch.tensor(X_lin_list, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred_lin_noisy = linear_model(X_lin_noisy_t).cpu().numpy().flatten()

    # Re-split predictions trialwise for metrics
    # We'll replicate get_trialwise_preds_linear logic
    linear_preds_noisy_by_trial = []
    idx_start = 0
    for i in range(X_test_linear_noisy.shape[0]):
        T_i = X_test_linear_noisy[i].shape[0]
        if T_i <= k:
            # no valid predictions
            linear_preds_noisy_by_trial.append(np.full((T_i,), np.nan))
            continue

        # for this trial, we had T_i - k predictions
        L_pred = T_i - k
        idx_end = idx_start + L_pred
        trial_preds = pred_lin_noisy[idx_start:idx_end]
        idx_start = idx_end

        aligned = np.full((T_i,), np.nan)
        aligned[k:] = trial_preds
        linear_preds_noisy_by_trial.append(aligned)

    # Evaluate linear metrics
    lin_metrics_list = []
    for i in range(len(X_test_linear_noisy)):
        y_true_i = Y_test_rnn[i]
        y_pred_i = linear_preds_noisy_by_trial[i]
        m = compute_metrics(y_true_i, y_pred_i)
        lin_metrics_list.append(m)
    avg_lin = average_metrics(lin_metrics_list)
    results_linear.append((sigma, avg_lin))

    print("  GRU  : ", avg_gru)
    print("  LSTM : ", avg_lstm)
    print("  LIN  : ", avg_lin)


# OPTIONAL: Plot VarExp vs. sigma
sigma_vals     = [r[0] for r in results_gru]
gru_vaf    = [r[1]['VAF'] for r in results_gru]
lstm_vaf   = [r[1]['VAF'] for r in results_lstm]
linear_vaf = [r[1]['VAF'] for r in results_linear]

plt.figure(figsize=(7,5))
plt.plot(sigma_vals, gru_vaf,    marker='o', label="GRU VAF")
plt.plot(sigma_vals, lstm_vaf,   marker='o', label="LSTM VAF")
plt.plot(sigma_vals, linear_vaf, marker='o', label="Linear VAF")
plt.title("Variance Accounted For vs. Noise Level (sigma)")
plt.xlabel("sigma")
plt.ylabel("VarExp")
plt.grid(True)
plt.legend()
plt.show()
