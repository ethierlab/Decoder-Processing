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
N = 18         # Number of PCA components
k = 18        # Lag length for RNN or linear
hidden_dim = 64
num_epochs = 400
batch_size = 64
learning_rate = 0.001
split = 0.75

cutoff = 10    # lowpass filter cutoff in Hz (optional)
fs = 1000      # sampling frequency in Hz
order = 5      # filter order

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

################################################################################
#                              2) HELPER FUNCTIONS
################################################################################
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, filt_order=5):
    """Applies a low-pass Butterworth filter to 1D data."""
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filt_order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_lagged_data_rnn(X, Y, seq_len=16):
    """
    Create lagged windows for RNN-based approach (GRU or LSTM).
    X: (num_trials, T, N)
    Y: (num_trials, T)
    seq_len: length of each sequence
    Returns:
        X_lagged: (total_windows, seq_len, N)
        Y_lagged: (total_windows,)
    """
    X_lagged, Y_lagged = [], []
    for trial_idx in range(X.shape[0]):
        T = X[trial_idx].shape[0]
        for t in range(seq_len, T):
            X_lagged.append(X[trial_idx, t-seq_len:t, :])  # shape (seq_len, N)
            Y_lagged.append(Y[trial_idx, t])
    return np.array(X_lagged), np.array(Y_lagged)

def create_lagged_data_linear(X, Y, seq_len=16):
    """
    Create lagged vectors for linear approach.
    X: (T, N)  (ALL trials concatenated for train or test)
    Y: (T,)    (ALL trials concatenated for train or test)
    seq_len: how many past time steps to flatten
    Returns:
        X_lagged: (T-seq_len, seq_len*N)
        Y_lagged: (T-seq_len,)
    """
    T, N = X.shape
    if T <= seq_len:
        raise ValueError("Number of time steps must be > seq_len (k).")

    X_lagged, Y_lagged = [], []
    for t in range(seq_len, T):
        X_lagged.append(X[t-seq_len:t, :].flatten())  # shape: (seq_len*N,)
        Y_lagged.append(Y[t])
    return np.array(X_lagged), np.array(Y_lagged)

def get_trialwise_preds_rnn(model, X_trials, seq_len=16):
    """
    Given an RNN model and the *test trials* X_trials: (num_trials, T, N),
    generate predictions per trial, returning a list of length num_trials.
    Each element is (T, ) with np.nan for the first seq_len steps.
    """
    model.eval()
    preds_by_trial = []
    with torch.no_grad():
        for trial_idx in range(X_trials.shape[0]):
            single_X = X_trials[trial_idx]  # shape (T, N)
            T_i = single_X.shape[0]

            # Create lagged windows for just this single trial
            X_lagged = []
            for t in range(seq_len, T_i):
                X_lagged.append(single_X[t-seq_len:t, :])
            X_lagged = np.array(X_lagged, dtype=np.float32)

            # Predict
            if len(X_lagged) == 0:
                # If the trial is shorter than seq_len, skip
                preds_by_trial.append(np.full((T_i,), np.nan))
                continue

            X_lagged_t = torch.tensor(X_lagged, dtype=torch.float32, device=device)
            preds = model(X_lagged_t).cpu().numpy().flatten()

            # Align with original timeline
            aligned_preds = np.full((T_i,), np.nan)
            aligned_preds[seq_len:] = preds
            preds_by_trial.append(aligned_preds)
    return preds_by_trial

def get_trialwise_preds_linear(preds_concat, trial_lengths, seq_len=16):
    """
    Distribute the 'preds_concat' (one big array) into a list of trial-wise arrays,
    inserting np.nan for the first seq_len steps in each trial.
    """
    preds_by_trial = []
    idx_start = 0
    for length in trial_lengths:
        # For this trial of length L, we have L-seq_len predictions
        L = length
        L_pred = max(0, L - seq_len)  # might be zero if L < seq_len
        idx_end = idx_start + L_pred
        # slice out that portion
        trial_preds = preds_concat[idx_start:idx_end]
        idx_start = idx_end

        # align
        aligned_preds = np.full((L,), np.nan)
        if L_pred > 0:
            aligned_preds[seq_len:] = trial_preds
        preds_by_trial.append(aligned_preds)
    return preds_by_trial

################################################################################
#                              3) LOAD DATA
################################################################################
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)

# Sort trial keys for consistent order
trial_keys = sorted(data['PCA'].keys())
num_trials_total = len(trial_keys)

# For the RNN approach, we keep it trial-based
X_rnn = np.stack([data['PCA'][k][:N].T for k in trial_keys])  # (num_trials, T, N)
Y_rnn = np.stack([data['Force']['x'][k]   for k in trial_keys])  # (num_trials, T)

print("RNN approach shapes:", X_rnn.shape, Y_rnn.shape)

# For the linear approach, we will also concatenate all trials
X_linear_list, Y_linear_list = [], []
trial_lengths = []  # Keep track of each trial’s length

for k_t in trial_keys:
    tmp_pca = data['PCA'][k_t][:N].T   # shape (T, N)
    tmp_force = data['Force']['x'][k_t] # shape (T,)
    X_linear_list.append(tmp_pca)
    Y_linear_list.append(tmp_force)
    trial_lengths.append(tmp_pca.shape[0])

X_linear_all = np.concatenate(X_linear_list, axis=0)  # shape (SumOfT, N)
Y_linear_all = np.concatenate(Y_linear_list, axis=0)  # shape (SumOfT,)

################################################################################
#                       4) FILTER (OPTIONAL) & Z-SCORE
################################################################################
# Example: filter Y if you wish
# for i in range(Y_rnn.shape[0]):
#     Y_rnn[i] = apply_lowpass_filter(Y_rnn[i], cutoff, fs, order)

# Z-score for RNN approach
X_rnn_z = zscore(X_rnn, axis=(0,1))  # zscore across trials and time
Y_rnn_z = zscore(Y_rnn, axis=(0,1))

# Z-score for linear approach
# (If you want to do the filtering first, do that before z-scoring)
X_linear_z = zscore(X_linear_all, axis=0)
Y_linear_z = zscore(Y_linear_all, axis=0)

################################################################################
#                5) SPLIT INTO TRAIN/TEST
################################################################################
num_train_trials = int(num_trials_total * split)
# RNN approach: separate by trial
X_train_rnn = X_rnn_z[:num_train_trials]
Y_train_rnn = Y_rnn_z[:num_train_trials]
X_test_rnn  = X_rnn_z[num_train_trials:]
Y_test_rnn  = Y_rnn_z[num_train_trials:]

# Linear approach: we figure out the index in the *concatenated* array
train_end_idx = sum(trial_lengths[:num_train_trials])  # sum up lengths of first 80% trials
X_linear_train = X_linear_z[:train_end_idx]
Y_linear_train = Y_linear_z[:train_end_idx]
X_linear_test  = X_linear_z[train_end_idx:]
Y_linear_test  = Y_linear_z[train_end_idx:]
test_trial_lengths = trial_lengths[num_train_trials:]   # For re-slicing predictions

################################################################################
#   6) PREPARE LAGGED DATA (RNN vs. LINEAR) & BUILD DATALOADERS
################################################################################
# ----- RNN (GRU or LSTM) -----
X_rnn_train_lag, Y_rnn_train_lag = create_lagged_data_rnn(X_train_rnn, Y_train_rnn, seq_len=k)
X_rnn_test_lag,  Y_rnn_test_lag  = create_lagged_data_rnn(X_test_rnn,  Y_test_rnn,  seq_len=k)

X_rnn_train_t = torch.tensor(X_rnn_train_lag, dtype=torch.float32)
Y_rnn_train_t = torch.tensor(Y_rnn_train_lag, dtype=torch.float32).unsqueeze(-1)

X_rnn_test_t = torch.tensor(X_rnn_test_lag, dtype=torch.float32)
Y_rnn_test_t = torch.tensor(Y_rnn_test_lag, dtype=torch.float32).unsqueeze(-1)

train_rnn_dataset = TensorDataset(X_rnn_train_t, Y_rnn_train_t)
train_rnn_loader  = DataLoader(train_rnn_dataset, batch_size=batch_size, shuffle=True)

test_rnn_dataset  = TensorDataset(X_rnn_test_t, Y_rnn_test_t)
test_rnn_loader   = DataLoader(test_rnn_dataset, batch_size=batch_size, shuffle=False)

# ----- Linear -----
X_lin_train_lag, Y_lin_train_lag = create_lagged_data_linear(X_linear_train, Y_linear_train, seq_len=k)
X_lin_test_lag,  Y_lin_test_lag  = create_lagged_data_linear(X_linear_test,  Y_linear_test,  seq_len=k)

X_lin_train_t = torch.tensor(X_lin_train_lag, dtype=torch.float32)
Y_lin_train_t = torch.tensor(Y_lin_train_lag, dtype=torch.float32)
X_lin_test_t  = torch.tensor(X_lin_test_lag,  dtype=torch.float32)
Y_lin_test_t  = torch.tensor(Y_lin_test_lag,  dtype=torch.float32)

train_lin_dataset = TensorDataset(X_lin_train_t, Y_lin_train_t)
train_lin_loader  = DataLoader(train_lin_dataset, batch_size=batch_size, shuffle=True)

################################################################################
#           7) DEFINE MODEL ARCHITECTURES (GRU, LSTM, and Linear)
################################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)        # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]        # last time-step
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LinearLagModel(nn.Module):
    """A purely linear model on lagged data (flattened)."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

################################################################################
#           8) INITIALIZE MODELS, LOSS, OPTIMIZERS
################################################################################
input_dim_rnn = X_rnn_train_t.shape[2]   # = N
gru_model = GRUModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)
lstm_model = LSTMModel(input_size=input_dim_rnn, hidden_size=hidden_dim).to(device)

input_dim_linear = k * X_linear_train.shape[1]  # k*N
linear_model = LinearLagModel(input_dim_linear).to(device)

criterion = nn.MSELoss()
gru_optimizer   = optim.Adam(gru_model.parameters(),   lr=learning_rate)
lstm_optimizer  = optim.Adam(lstm_model.parameters(),  lr=learning_rate)
linear_optimizer= optim.Adam(linear_model.parameters(),lr=learning_rate)

################################################################################
#           9) TRAINING FUNCTIONS (COMMON LOOP)
################################################################################
def train_rnn(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def test_rnn(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            running_loss += loss.item()
    return running_loss / len(loader)

def train_linear(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        pred = model(X_batch).squeeze()
        loss = criterion(pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def test_linear(model, X_t, Y_t, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(X_t.to(device)).squeeze()
        loss = criterion(pred, Y_t.to(device))
    return loss.item()

################################################################################
#  10) MAIN TRAINING LOOP (EPOCHS) FOR GRU, LSTM, AND LINEAR
################################################################################
gru_losses_train, lstm_losses_train, lin_losses_train = [], [], []
gru_losses_test,  lstm_losses_test,  lin_losses_test  = [], [], []

for epoch in range(num_epochs):
    # --- Train ---
    gru_loss_train  = train_rnn(gru_model, train_rnn_loader, gru_optimizer, criterion)
    lstm_loss_train = train_rnn(lstm_model, train_rnn_loader, lstm_optimizer, criterion)
    lin_loss_train  = train_linear(linear_model, train_lin_loader, linear_optimizer, criterion)

    # --- Test ---
    gru_loss_test   = test_rnn(gru_model, test_rnn_loader, criterion)
    lstm_loss_test  = test_rnn(lstm_model, test_rnn_loader, criterion)
    lin_loss_test   = test_linear(linear_model, X_lin_test_t, Y_lin_test_t, criterion)

    gru_losses_train.append(gru_loss_train)
    lstm_losses_train.append(lstm_loss_train)
    lin_losses_train.append(lin_loss_train)

    gru_losses_test.append(gru_loss_test)
    lstm_losses_test.append(lstm_loss_test)
    lin_losses_test.append(lin_loss_test)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: ")
        print(f"  GRU   - train: {gru_loss_train:.4f}, test: {gru_loss_test:.4f}")
        print(f"  LSTM  - train: {lstm_loss_train:.4f}, test: {lstm_loss_test:.4f}")
        print(f"  Linear- train: {lin_loss_train:.4f}, test: {lin_loss_test:.4f}")

print("Training completed for all models.")

################################################################################
#    11) GENERATE PREDICTIONS PER TEST TRIAL (GRU, LSTM, LINEAR)
################################################################################
print("Generating per-trial predictions...")

gru_model.eval()
lstm_model.eval()
linear_model.eval()

# GRU predictions
gru_preds_by_trial = get_trialwise_preds_rnn(gru_model, X_test_rnn, seq_len=k)
# LSTM predictions
lstm_preds_by_trial = get_trialwise_preds_rnn(lstm_model, X_test_rnn, seq_len=k)

# Linear predictions
# We have one big test set (X_linear_test -> X_lin_test_lag -> pred_lin_test)
with torch.no_grad():
    pred_lin_test = linear_model(X_lin_test_t.to(device)).cpu().numpy().flatten()
# Distribute those predictions among the test trials
linear_preds_by_trial = get_trialwise_preds_linear(pred_lin_test, test_trial_lengths, seq_len=k)

#############################################################################
#   12) EVALUATE METRICS (PER TRIAL AND COMBINED) FOR GRU, LSTM, LINEAR
#############################################################################

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    if len(y_true_valid) < 2:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Corr': np.nan, 'VarExp': np.nan}
    
    mse_val = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse_val)
    mae  = mean_absolute_error(y_true_valid, y_pred_valid)
    r2   = r2_score(y_true_valid, y_pred_valid)
    corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1]
    
    var_resid = np.var(y_true_valid - y_pred_valid)
    var_true  = np.var(y_true_valid)
    var_exp   = 1.0 - var_resid / var_true if var_true > 1e-12 else np.nan
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Corr': corr, 'VarExp': var_exp}

def average_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    avg_dict = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        avg_dict[k] = np.mean(vals) if len(vals) > 0 else np.nan
    return avg_dict

# --- 1) GRU per-trial ---
gru_per_trial = []
for i in range(len(gru_preds_by_trial)):
    y_true_i = Y_test_rnn[i]
    y_pred_i = gru_preds_by_trial[i]
    metrics_i = compute_metrics(y_true_i, y_pred_i)
    gru_per_trial.append(metrics_i)
    print(f'GRU metrics for test trials {i} : {metrics_i}')
    
gru_avg = average_metrics(gru_per_trial)

# --- 2) LSTM per-trial ---
lstm_per_trial = []
for i in range(len(lstm_preds_by_trial)):
    y_true_i = Y_test_rnn[i]
    y_pred_i = lstm_preds_by_trial[i]
    metrics_i = compute_metrics(y_true_i, y_pred_i)
    lstm_per_trial.append(metrics_i)
    print(f'GRU metrics for test trials {i} : {metrics_i}')
lstm_avg = average_metrics(lstm_per_trial)

# --- 3) Linear per-trial ---
lin_per_trial = []
for i in range(len(linear_preds_by_trial)):
    y_true_i = Y_test_rnn[i]
    y_pred_i = linear_preds_by_trial[i]  # same shape as y_true_i, with NaNs
    metrics_i = compute_metrics(y_true_i, y_pred_i)
    lin_per_trial.append(metrics_i)
    print(f'Linear metrics for test trials {i} : {metrics_i}')
lin_avg = average_metrics(lin_per_trial)

print("GRU average metrics across test trials :", gru_avg)
print("LSTM average metrics across test trials:", lstm_avg)
print("Linear average metrics across test trials:", lin_avg)

# Combined approach (flatten all test trials):
def flatten_predictions(y_true_rnn, preds_by_trial):
    """
    y_true_rnn: list of arrays [trial_i], each shape (T_i,)
    preds_by_trial: list of arrays [trial_i], same shape, with NaNs
    returns y_true_concat, y_pred_concat (1D arrays)
    """
    all_true = []
    all_pred = []
    for i in range(len(preds_by_trial)):
        all_true.append(y_true_rnn[i])
        all_pred.append(preds_by_trial[i])
    return np.concatenate(all_true), np.concatenate(all_pred)

gru_true_flat, gru_pred_flat = flatten_predictions(Y_test_rnn, gru_preds_by_trial)
gru_combined = compute_metrics(gru_true_flat, gru_pred_flat)

lstm_true_flat, lstm_pred_flat = flatten_predictions(Y_test_rnn, lstm_preds_by_trial)
lstm_combined = compute_metrics(lstm_true_flat, lstm_pred_flat)

lin_true_flat, lin_pred_flat  = flatten_predictions(Y_test_rnn, linear_preds_by_trial)
lin_combined = compute_metrics(lin_true_flat, lin_pred_flat)

print("GRU combined (all test trials) :", gru_combined)
print("LSTM combined (all test trials):", lstm_combined)
print("Linear combined (all test trials):", lin_combined)


################################################################################
#   13) PLOT COMPARISON ON 9 TEST TRIALS
################################################################################
print("Plotting results...")

num_test_trials = X_test_rnn.shape[0]
plot_n = min(num_test_trials, 9)

fig, axes = plt.subplots(3, 3, figsize=(15,12), sharex=False)
axes = axes.flatten()

for i in range(plot_n):
    ax = axes[i]
    ax.set_title(f"Test Trial #{i+1}")

    # Actual test trial from RNN perspective
    actual_y = Y_test_rnn[i]  # shape (T_i,)

    # GRU
    gru_pred = gru_preds_by_trial[i]   # shape (T_i,)
    # LSTM
    lstm_pred = lstm_preds_by_trial[i] # shape (T_i,)
    # Linear
    lin_pred  = linear_preds_by_trial[i]

    ax.plot(actual_y,        label="Actual", color="k")
    ax.plot(gru_pred,        label="GRU",    color="r", linestyle="--")
    ax.plot(lstm_pred,       label="LSTM",   color="g", linestyle="--")
    ax.plot(lin_pred,        label="Linear", color="b", linestyle=":")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-scored Force")

plt.tight_layout()
plt.show()

################################################################################
#   13) (OPTIONAL) PLOT TRAINING LOSSES
################################################################################
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(gru_losses_train,  label="GRU Train",  color='r')
# plt.plot(lstm_losses_train, label="LSTM Train", color='g')
# plt.plot(lin_losses_train,  label="Linear Train", color='b')
# plt.title("Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(gru_losses_test,  label="GRU Test",  color='r')
# plt.plot(lstm_losses_test, label="LSTM Test", color='g')
# plt.plot(lin_losses_test,  label="Linear Test", color='b')
# plt.title("Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend()

# plt.tight_layout()
# plt.show()
