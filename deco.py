import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

################################################################################
#                              1) PARAMETERS
################################################################################
FILE_PATH = 'Jango_dataset.pkl'
N = 16         # Number of PCA components
k = 15         # Lag length for RNN or linear
hidden_dim = 32
num_epochs = 150
batch_size = 64
learning_rate = 0.001
train_split = 0.75  # fraction of trials to use for training

################################################################################
#                              2) HELPER FUNCTIONS
################################################################################
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, filt_order=5):
    """Applies a low-pass Butterworth filter to 1D data."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filt_order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_lagged_data_rnn(X, Y, seq_len=16):
    """
    For RNN-based approach (GRU or LSTM).
    X: (num_trials, T, N)
    Y: (num_trials, T)
    seq_len: length of each sequence
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
    Builds time-lag windows *within a single trial* (no cross-boundary).
    X_trial: (T_i, N), Y_trial: (T_i,)
    Returns X_lagged: (T_i - seq_len, seq_len*N), Y_lagged: (T_i - seq_len,)
    """
    T_i, N = X_trial.shape
    if T_i <= seq_len:
        # If not enough data to form 1 window, return empty arrays
        return np.empty((0, seq_len*N), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_lagged_list = []
    Y_lagged_list = []
    for t in range(seq_len, T_i):
        window = X_trial[t-seq_len:t, :].reshape(-1)  # shape: (seq_len*N,)
        X_lagged_list.append(window)
        Y_lagged_list.append(Y_trial[t])

    return np.array(X_lagged_list, dtype=np.float32), np.array(Y_lagged_list, dtype=np.float32)

################################################################################
#   Functions to get per-trial predictions (RNN & Linear) for easy visualization
################################################################################
def get_trialwise_preds_rnn(model, X_trials, seq_len=16, device='cpu'):
    """
    Feeds each trial through the RNN model separately and returns
    a list of length num_trials, each (T_i,) with np.nan for the first seq_len steps.
    """
    model.eval()
    preds_by_trial = []
    with torch.no_grad():
        for i in range(X_trials.shape[0]):
            T_i = X_trials[i].shape[0]
            if T_i <= seq_len:
                # Not enough time points
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue

            # Build windows for just this trial
            X_lag = []
            for t in range(seq_len, T_i):
                X_lag.append(X_trials[i, t-seq_len:t, :])
            X_lag = np.array(X_lag, dtype=np.float32)

            X_lag_t = torch.tensor(X_lag, dtype=torch.float32, device=device)
            y_hat = model(X_lag_t).cpu().numpy().flatten()  # shape (T_i - seq_len,)

            # align
            aligned = np.full((T_i,), np.nan)
            aligned[seq_len:] = y_hat
            preds_by_trial.append(aligned)
    return preds_by_trial

def get_trialwise_preds_linear(model, X_trials, Y_trials, seq_len=16, device='cpu'):
    """
    For the linear model, do the same "trial-by-trial" approach:
      - create windows inside each trial
      - run model predictions
      - align them (first seq_len are NaN)
    Returns a list of length num_trials, each shape (T_i,)
    """
    model.eval()
    preds_by_trial = []
    with torch.no_grad():
        for i in range(X_trials.shape[0]):
            T_i = X_trials[i].shape[0]
            if T_i <= seq_len:
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue

            # Build windows for just this one trial
            X_lag_list = []
            for t in range(seq_len, T_i):
                window = X_trials[i, t-seq_len:t, :].reshape(-1)  # flatten
                X_lag_list.append(window)
            X_lag = np.array(X_lag_list, dtype=np.float32)

            X_lag_t = torch.tensor(X_lag, dtype=torch.float32, device=device)
            y_hat = model(X_lag_t).cpu().numpy().flatten()

            aligned = np.full((T_i,), np.nan)
            aligned[seq_len:] = y_hat
            preds_by_trial.append(aligned)
    return preds_by_trial


################################################################################
#                              3) LOAD DATA
################################################################################
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)

trial_keys = sorted(data['PCA'].keys())
num_trials_total = len(trial_keys)

# Build (num_trials, T, N) arrays for RNN
X_rnn = np.stack([data['PCA'][k][:N].T for k in trial_keys])  # (num_trials, T, N)
Y_rnn = np.stack([data['Force']['y'][k]   for k in trial_keys])  # (num_trials, T)

print("RNN approach shapes:", X_rnn.shape, Y_rnn.shape)
print(f"Number of total trials: {len(X_rnn)}")

################################################################################
#     4) FILTER (OPTIONAL) & Z-SCORE PER TRIAL (RNN approach & linear approach)
################################################################################
# (If you wish to filter, do it trial by trial)
# for i in range(len(Y_rnn)):
#     Y_rnn[i] = apply_lowpass_filter(Y_rnn[i], cutoff_freq=10, sampling_rate=1000, filt_order=5)

# Z-score each trial's X and Y
# You can do per-trial standardization (common approach)
for i in range(len(X_rnn)):
    X_rnn[i] = zscore(X_rnn[i], axis=0)
    Y_rnn[i] = zscore(Y_rnn[i])

################################################################################
# 5) SPLIT TRIALS INTO TRAIN/TEST
################################################################################
num_train_trials = int(num_trials_total * train_split)
X_train_rnn = X_rnn[:num_train_trials]
Y_train_rnn = Y_rnn[:num_train_trials]
X_test_rnn  = X_rnn[num_train_trials:]
Y_test_rnn  = Y_rnn[num_train_trials:]

################################################################################
# 6) BUILD LAGGED DATA FOR RNN (train & test)
################################################################################
X_rnn_train_lag, Y_rnn_train_lag = create_lagged_data_rnn(X_train_rnn, Y_train_rnn, seq_len=k)
X_rnn_test_lag,  Y_rnn_test_lag  = create_lagged_data_rnn(X_test_rnn,  Y_test_rnn,  seq_len=k)

# Convert to Tensors
X_rnn_train_t = torch.tensor(X_rnn_train_lag, dtype=torch.float32)
Y_rnn_train_t = torch.tensor(Y_rnn_train_lag, dtype=torch.float32).unsqueeze(-1)
X_rnn_test_t  = torch.tensor(X_rnn_test_lag,  dtype=torch.float32)
Y_rnn_test_t  = torch.tensor(Y_rnn_test_lag,  dtype=torch.float32).unsqueeze(-1)

train_rnn_dataset = TensorDataset(X_rnn_train_t, Y_rnn_train_t)
train_rnn_loader  = DataLoader(train_rnn_dataset, batch_size=batch_size, shuffle=True)
test_rnn_dataset  = TensorDataset(X_rnn_test_t,  Y_rnn_test_t)
test_rnn_loader   = DataLoader(test_rnn_dataset,  batch_size=batch_size, shuffle=False)


################################################################################
# 7) BUILD LAGGED DATA FOR LINEAR (PER-TRIAL), THEN CONCATENATE
################################################################################
def build_linear_dataset(X_trials, Y_trials, seq_len=16):
    """
    For each trial in X_trials, Y_trials, build windows (per trial),
    then concatenate them across all trials.
    """
    X_list = []
    Y_list = []
    for i in range(len(X_trials)):
        X_lag, Y_lag = create_lagged_data_linear_per_trial(X_trials[i], Y_trials[i], seq_len=seq_len)
        X_list.append(X_lag)
        Y_list.append(Y_lag)
    X_cat = np.concatenate(X_list, axis=0) if len(X_list) > 0 else np.empty((0, seq_len*X_trials[0].shape[1]))
    Y_cat = np.concatenate(Y_list, axis=0) if len(Y_list) > 0 else np.empty((0,))
    return X_cat, Y_cat

X_lin_train_cat, Y_lin_train_cat = build_linear_dataset(X_train_rnn, Y_train_rnn, seq_len=k)
X_lin_test_cat,  Y_lin_test_cat  = build_linear_dataset(X_test_rnn,  Y_test_rnn,  seq_len=k)

X_lin_train_t = torch.tensor(X_lin_train_cat, dtype=torch.float32)
Y_lin_train_t = torch.tensor(Y_lin_train_cat, dtype=torch.float32)
X_lin_test_t  = torch.tensor(X_lin_test_cat,  dtype=torch.float32)
Y_lin_test_t  = torch.tensor(Y_lin_test_cat,  dtype=torch.float32)

train_lin_dataset = TensorDataset(X_lin_train_t, Y_lin_train_t)
train_lin_loader  = DataLoader(train_lin_dataset, batch_size=batch_size, shuffle=True)

################################################################################
# 8) DEFINE MODELS (GRU, LSTM, LINEAR)
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
    """A simple linear model or small MLP on flattened lagged data."""
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

################################################################################
# 9) INSTANTIATE MODELS, OPTIMIZERS, LOSS
################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

input_dim_rnn = X_rnn_train_t.shape[2]  # = N
gru_model = GRUModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)
lstm_model = LSTMModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)

# For linear model: input_dim = k*N
if len(X_train_rnn) > 0:
    input_dim_linear = X_lin_train_cat.shape[1]
else:
    input_dim_linear = k * N  # fallback
linear_model = LinearLagModel(input_dim_linear, hidden_dim=hidden_dim).to(device)

criterion = nn.MSELoss()
gru_optimizer   = optim.Adam(gru_model.parameters(),   lr=learning_rate)
lstm_optimizer  = optim.Adam(lstm_model.parameters(),  lr=learning_rate)
linear_optimizer= optim.Adam(linear_model.parameters(),lr=learning_rate)

################################################################################
# 10) TRAINING LOOPS
################################################################################
def train_rnn(model, loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_rnn(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, Yb)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_linear(model, loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        pred = model(Xb).squeeze()
        loss = criterion(pred, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_linear(model, X_t, Y_t, criterion, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_t = X_t.to(device)
        Y_t = Y_t.to(device)
        pred = model(X_t).squeeze()
        loss = criterion(pred, Y_t)
    return loss.item()

################################################################################
# 11) RUN TRAINING
################################################################################
epochs = num_epochs
gru_train_losses, gru_test_losses = [], []
lstm_train_losses, lstm_test_losses = [], []
lin_train_losses,  lin_test_losses  = [], []

for ep in range(epochs):
    # -- Train --
    gru_tr = train_rnn(gru_model, train_rnn_loader, gru_optimizer, criterion, device)
    lstm_tr= train_rnn(lstm_model, train_rnn_loader, lstm_optimizer, criterion, device)
    lin_tr = train_linear(linear_model, train_lin_loader, linear_optimizer, criterion, device)

    # -- Test --
    gru_te = test_rnn(gru_model, test_rnn_loader, criterion, device)
    lstm_te= test_rnn(lstm_model, test_rnn_loader, criterion, device)
    lin_te = test_linear(linear_model, X_lin_test_t, Y_lin_test_t, criterion, device)

    gru_train_losses.append(gru_tr)
    lstm_train_losses.append(lstm_tr)
    lin_train_losses.append(lin_tr)

    gru_test_losses.append(gru_te)
    lstm_test_losses.append(lstm_te)
    lin_test_losses.append(lin_te)

    if (ep+1) % 20 == 0:
        print(f"Epoch {ep+1}/{epochs} | GRU (tr={gru_tr:.4f}, te={gru_te:.4f}) "
              f"| LSTM (tr={lstm_tr:.4f}, te={lstm_te:.4f}) "
              f"| LIN (tr={lin_tr:.4f}, te={lin_te:.4f})")

print("Training complete.")

################################################################################
# 12) PER-TRIAL PREDICTIONS & METRICS
################################################################################
def compute_metrics(y_true, y_pred):
    # ignore NaNs in y_pred
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    if len(y_true_valid) < 2:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, Corr=np.nan, VarExp=np.nan)
    mse_val = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse_val)
    mae  = mean_absolute_error(y_true_valid, y_pred_valid)
    r2   = r2_score(y_true_valid, y_pred_valid)
    corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1]
    var_resid = np.var(y_true_valid - y_pred_valid)
    var_true  = np.var(y_true_valid)
    var_exp   = 1.0 - (var_resid / var_true) if var_true > 1e-12 else np.nan
    return dict(RMSE=rmse, MAE=mae, R2=r2, Corr=corr, VarExp=var_exp)

def average_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        out[k] = np.mean(vals) if len(vals) > 0 else np.nan
    return out

# Generate predictions per trial, same shape as Y_test_rnn
gru_preds_by_trial  = get_trialwise_preds_rnn(gru_model, X_test_rnn, seq_len=k, device=device)
lstm_preds_by_trial = get_trialwise_preds_rnn(lstm_model, X_test_rnn, seq_len=k, device=device)
linear_preds_by_trial = get_trialwise_preds_linear(linear_model, X_test_rnn, Y_test_rnn, seq_len=k, device=device)

# Evaluate per-trial
gru_pertrial = []
lstm_pertrial= []
lin_pertrial = []
for i in range(len(X_test_rnn)):
    y_true_i = Y_test_rnn[i]
    m1 = compute_metrics(y_true_i, gru_preds_by_trial[i])
    m2 = compute_metrics(y_true_i, lstm_preds_by_trial[i])
    m3 = compute_metrics(y_true_i, linear_preds_by_trial[i])
    gru_pertrial.append(m1)
    lstm_pertrial.append(m2)
    lin_pertrial.append(m3)
    print(f"\n----- Per-Trial Metrics (Test Set {i}) -----")
    print(f"GRU   : {m1}")
    print(f"LSTM  : {m2}")
    print(f"Linear: {m3}")
gru_avg = average_metrics(gru_pertrial)
lstm_avg= average_metrics(lstm_pertrial)
lin_avg = average_metrics(lin_pertrial)

print("\n----- Per-Trial Average Metrics (Test Set) -----")
print("GRU   :", gru_avg)
print("LSTM  :", lstm_avg)
print("Linear:", lin_avg)

# Combined (flatten everything)
def flatten_all(y_rnn, y_pred_by_trial):
    all_true = []
    all_pred = []
    for i in range(len(y_pred_by_trial)):
        all_true.append(y_rnn[i])
        all_pred.append(y_pred_by_trial[i])
    return np.concatenate(all_true), np.concatenate(all_pred)

gru_yT, gru_yP   = flatten_all(Y_test_rnn, gru_preds_by_trial)
lstm_yT, lstm_yP = flatten_all(Y_test_rnn, lstm_preds_by_trial)
lin_yT, lin_yP   = flatten_all(Y_test_rnn, linear_preds_by_trial)

gru_comb = compute_metrics(gru_yT, gru_yP)
lstm_comb= compute_metrics(lstm_yT, lstm_yP)
lin_comb = compute_metrics(lin_yT, lin_yP)

print("\n----- Combined Metrics (All Test Trials Flattened) -----")
print("GRU   :", gru_comb)
print("LSTM  :", lstm_comb)
print("Linear:", lin_comb)

################################################################################
# 13) PLOTTING (Compare up to 9 test trials)
################################################################################
plot_n = min(len(X_test_rnn), 9)
fig, axes = plt.subplots(3, 3, figsize=(15,12), sharex=False)
axes = axes.flatten()
for i in range(plot_n):
    ax = axes[i]
    ax.set_title(f"Test Trial #{i+1}")
    ax.plot(Y_test_rnn[i], label="Actual", color='k')
    ax.plot(gru_preds_by_trial[i], label="GRU", color='r', ls='--')
    ax.plot(lstm_preds_by_trial[i],label="LSTM",color='g', ls='--')
    ax.plot(linear_preds_by_trial[i], label="Linear", color='b', ls=':')
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-scored Force")
plt.tight_layout()
plt.show()

################################################################################
# 14) (OPTIONAL) PLOT TRAINING/TEST LOSS CURVES
################################################################################
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(gru_train_losses,  label="GRU Train", color='r')
# plt.plot(lstm_train_losses, label="LSTM Train",color='g')
# plt.plot(lin_train_losses,  label="Lin Train", color='b')
# plt.title("Training Loss (MSE)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(gru_test_losses,  label="GRU Test",  color='r')
# plt.plot(lstm_test_losses, label="LSTM Test", color='g')
# plt.plot(lin_test_losses,  label="Lin Test",  color='b')
# plt.title("Test Loss (MSE)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.show()
