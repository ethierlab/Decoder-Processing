import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

################################################################################
#                           Helper Functions
################################################################################
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_lowpass_filter(data, cutoff_freq, sampling_rate, filt_order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filt_order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_lagged_data_rnn(X, Y, seq_len=16):
    X_lagged, Y_lagged = [], []
    num_trials = X.shape[0]
    for i in range(num_trials):
        T_i = X[i].shape[0]
        for t in range(seq_len, T_i):
            X_lagged.append(X[i, t-seq_len:t, :])
            Y_lagged.append(Y[i, t])
    return np.array(X_lagged), np.array(Y_lagged)

def create_lagged_data_linear_per_trial(X_trial, Y_trial, seq_len=16):
    T_i, N = X_trial.shape
    if T_i <= seq_len:
        return np.empty((0, seq_len*N), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_lagged_list, Y_lagged_list = [], []
    for t in range(seq_len, T_i):
        window = X_trial[t-seq_len:t, :].reshape(-1)
        X_lagged_list.append(window)
        Y_lagged_list.append(Y_trial[t])
    return np.array(X_lagged_list, dtype=np.float32), np.array(Y_lagged_list, dtype=np.float32)

def get_trialwise_preds_rnn(model, X_trials, seq_len=16, device='cpu'):
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
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, Corr=np.nan, VarExp=np.nan, VAF=np.nan)
    mse_val = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse_val)
    mae  = mean_absolute_error(y_true_valid, y_pred_valid)
    r2   = r2_score(y_true_valid, y_pred_valid)
    corr = np.corrcoef(y_true_valid, y_pred_valid)[0,1]
    var_resid = np.var(y_true_valid - y_pred_valid)
    var_true  = np.var(y_true_valid)
    var_exp   = 1.0 - (var_resid / var_true) if var_true > 1e-12 else np.nan
    vaf = 1 - (np.var(y_true_valid - y_pred_valid) / np.var(y_true_valid))
    return dict(RMSE=rmse, MAE=mae, R2=r2, Corr=corr, VarExp=var_exp, VAF=vaf)

def average_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        out[k] = np.mean(vals) if len(vals) > 0 else np.nan
    return out

def flatten_all(y_rnn, y_pred_by_trial):
    all_true, all_pred = [], []
    for i in range(len(y_pred_by_trial)):
        all_true.append(y_rnn[i])
        all_pred.append(y_pred_by_trial[i])
    return np.concatenate(all_true), np.concatenate(all_pred)

################################################################################
#                           1) PARAMETERS
################################################################################
# --- Random Seed ---
SEED = 42
set_seed(SEED)

# FILE_PATH = 'projected_data_test.pkl'
FILE_PATH = "Jango_dataset.pkl"
N = 2         # Number of PCA components
k = 11        # Lag length for RNN or Linear
hidden_dim = 5
num_epochs = 150
batch_size = 64
learning_rate = 0.001
train_split = 0.75  # Fraction of trials for training
reduction = 'PCA'   # Reduction method used (PCA, UMAP, t-SNE)

# Weight file paths
weight_file_GRU   = "gru_weights_rat.pth"
weight_file_LSTM  = "lstm_weights_rat.pth"
weight_file_lin   = "linear_weights_rat.pth"
weight_file_LiGRU = "ligru_weights_rat.pth"

# Which decoders to run
run_gru   = True
run_lstm  = True
run_lin   = True
run_ligru = True  # Enable Li-GRU

################################################################################
#                        2) LOAD DATA & PREPROCESSING
################################################################################
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)
trial_keys = sorted(data[reduction].keys())
num_trials_total = len(trial_keys)
X_rnn = np.stack([data[reduction][k][:N].T for k in trial_keys])  # (num_trials, T, N)
Y_rnn = np.stack([data['Force']['y'][k] for k in trial_keys])       # (num_trials, T)
print("RNN approach shapes:", X_rnn.shape, Y_rnn.shape)
print(f"Total trials: {num_trials_total}")

# Optional low-pass filtering (if needed)
# for i in range(len(Y_rnn)):
#     Y_rnn[i] = apply_lowpass_filter(Y_rnn[i], cutoff_freq=10, sampling_rate=1000, filt_order=5)

# Z-score normalization per trial
for i in range(len(X_rnn)):
    X_rnn[i] = zscore(X_rnn[i], axis=0)
    Y_rnn[i] = zscore(Y_rnn[i])

################################################################################
#                         3) TRAIN/TEST SPLIT
################################################################################
num_train_trials = int(num_trials_total * train_split)
X_train_rnn = X_rnn[:num_train_trials]
Y_train_rnn = Y_rnn[:num_train_trials]
X_test_rnn  = X_rnn[num_train_trials:]
Y_test_rnn  = Y_rnn[num_train_trials:]

################################################################################
#                    4) CREATE LAGGED DATA FOR RNN & LINEAR
################################################################################
X_rnn_train_lag, Y_rnn_train_lag = create_lagged_data_rnn(X_train_rnn, Y_train_rnn, seq_len=k)
X_rnn_test_lag,  Y_rnn_test_lag  = create_lagged_data_rnn(X_test_rnn, Y_test_rnn, seq_len=k)
X_rnn_train_t = torch.tensor(X_rnn_train_lag, dtype=torch.float32)
Y_rnn_train_t = torch.tensor(Y_rnn_train_lag, dtype=torch.float32).unsqueeze(-1)
X_rnn_test_t  = torch.tensor(X_rnn_test_lag, dtype=torch.float32)
Y_rnn_test_t  = torch.tensor(Y_rnn_test_lag, dtype=torch.float32).unsqueeze(-1)
train_rnn_dataset = TensorDataset(X_rnn_train_t, Y_rnn_train_t)
train_rnn_loader  = DataLoader(train_rnn_dataset, batch_size=batch_size, shuffle=True)
test_rnn_dataset  = TensorDataset(X_rnn_test_t, Y_rnn_test_t)
test_rnn_loader   = DataLoader(test_rnn_dataset, batch_size=batch_size, shuffle=False)

def build_linear_dataset(X_trials, Y_trials, seq_len=16):
    X_list, Y_list = [], []
    for i in range(len(X_trials)):
        X_lag, Y_lag = create_lagged_data_linear_per_trial(X_trials[i], Y_trials[i], seq_len=seq_len)
        X_list.append(X_lag)
        Y_list.append(Y_lag)
    if len(X_list) > 0:
        X_cat = np.concatenate(X_list, axis=0)
        Y_cat = np.concatenate(Y_list, axis=0)
    else:
        X_cat = np.empty((0, seq_len*X_trials[0].shape[1]))
        Y_cat = np.empty((0,))
    return X_cat, Y_cat

X_lin_train_cat, Y_lin_train_cat = build_linear_dataset(X_train_rnn, Y_train_rnn, seq_len=k)
X_lin_test_cat,  Y_lin_test_cat  = build_linear_dataset(X_test_rnn, Y_test_rnn, seq_len=k)
X_lin_train_t = torch.tensor(X_lin_train_cat, dtype=torch.float32)
Y_lin_train_t = torch.tensor(Y_lin_train_cat, dtype=torch.float32)
X_lin_test_t  = torch.tensor(X_lin_test_cat, dtype=torch.float32)
Y_lin_test_t  = torch.tensor(Y_lin_test_cat, dtype=torch.float32)
train_lin_dataset = TensorDataset(X_lin_train_t, Y_lin_train_t)
train_lin_loader  = DataLoader(train_lin_dataset, batch_size=batch_size, shuffle=True)

################################################################################
#                         5) DEFINE MODELS
################################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LinearLagModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LinearLagModel, self).__init__()
        self.linear_hidden = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.act(x)
        return self.linear_out(x)

# --- Li-GRU Implementation ---
class LiGRUCell(nn.Module):
    """
    A simplified GRU cell (Li-GRU) that removes the reset gate and uses ReLU
    for the candidate activation.
    """
    def __init__(self, input_size, hidden_size):
        super(LiGRUCell, self).__init__()
        # Update gate parameters
        self.x2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        # Candidate hidden state parameters (using ReLU)
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.x2z(x) + self.h2z(h))
        h_candidate = torch.relu(self.x2h(x) + self.h2h(h))
        h_new = (1 - z) * h + z * h_candidate
        return h_new

class LiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # For simplicity, we implement a single-layer Li-GRU
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x has shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

################################################################################
#                6) INITIALIZE MODELS, OPTIMIZERS, LOSS, DEVICE
################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
gru_model   = GRUModel(input_size=X_rnn_train_t.shape[2], hidden_size=hidden_dim).to(device)
lstm_model  = LSTMModel(input_size=X_rnn_train_t.shape[2], hidden_size=hidden_dim).to(device)
linear_model= LinearLagModel(input_dim=X_lin_train_cat.shape[1], hidden_dim=hidden_dim).to(device)
ligru_model = LiGRUModel(input_size=X_rnn_train_t.shape[2], hidden_size=hidden_dim).to(device)

gru_optimizer   = optim.Adam(gru_model.parameters(), lr=learning_rate)
lstm_optimizer  = optim.Adam(lstm_model.parameters(), lr=learning_rate)
linear_optimizer= optim.Adam(linear_model.parameters(), lr=learning_rate)
ligru_optimizer = optim.Adam(ligru_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# --- Parameter Counting Function ---
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################################################################################
#              7) TRAIN/TEST FUNCTIONS & TRAINING LOOP
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
        X_t, Y_t = X_t.to(device), Y_t.to(device)
        pred = model(X_t).squeeze()
        loss = criterion(pred, Y_t)
    return loss.item()

gru_train_losses, gru_test_losses = [], []
lstm_train_losses, lstm_test_losses = [], []
lin_train_losses,  lin_test_losses  = [], []
ligru_train_losses, ligru_test_losses = [], []

for ep in range(num_epochs):
    if run_gru:
        gru_tr = train_rnn(gru_model, train_rnn_loader, gru_optimizer, criterion, device)
        gru_train_losses.append(gru_tr)
    if run_lstm:
        lstm_tr = train_rnn(lstm_model, train_rnn_loader, lstm_optimizer, criterion, device)
        lstm_train_losses.append(lstm_tr)
    if run_lin:
        lin_tr = train_linear(linear_model, train_lin_loader, linear_optimizer, criterion, device)
        lin_train_losses.append(lin_tr)
    if run_ligru:
        ligru_tr = train_rnn(ligru_model, train_rnn_loader, ligru_optimizer, criterion, device)
        ligru_train_losses.append(ligru_tr)
        
    if run_gru:
        gru_te = test_rnn(gru_model, test_rnn_loader, criterion, device)
        gru_test_losses.append(gru_te)
    if run_lstm:
        lstm_te = test_rnn(lstm_model, test_rnn_loader, criterion, device)
        lstm_test_losses.append(lstm_te)
    if run_lin:
        lin_te = test_linear(linear_model, X_lin_test_t, Y_lin_test_t, criterion, device)
        lin_test_losses.append(lin_te)
    if run_ligru:
        ligru_te = test_rnn(ligru_model, test_rnn_loader, criterion, device)
        ligru_test_losses.append(ligru_te)
        
    if (ep+1) % 20 == 0:
        message = f"Epoch {ep+1}/{num_epochs} | "
        if run_gru:
            message += f"GRU (tr={gru_tr:.4f}, te={gru_te:.4f}) "
        if run_lstm:
            message += f"| LSTM (tr={lstm_tr:.4f}, te={lstm_te:.4f}) "
        if run_lin:
            message += f"| Linear (tr={lin_tr:.4f}, te={lin_te:.4f}) "
        if run_ligru:
            message += f"| Li-GRU (tr={ligru_tr:.4f}, te={ligru_te:.4f}) "
        print(message)

print("Training complete.")

# --- Save weights for each decoder ---
if run_gru:
    torch.save(gru_model.state_dict(), weight_file_GRU)
    print(f"GRU weights saved to {weight_file_GRU}")
if run_lstm:
    torch.save(lstm_model.state_dict(), weight_file_LSTM)
    print(f"LSTM weights saved to {weight_file_LSTM}")
if run_lin:
    torch.save(linear_model.state_dict(), weight_file_lin)
    print(f"Linear weights saved to {weight_file_lin}")
if run_ligru:
    torch.save(ligru_model.state_dict(), weight_file_LiGRU)
    print(f"Li-GRU weights saved to {weight_file_LiGRU}")

# --- Count and display parameters ---
if run_gru:
    print("GRU model parameters:", count_params(gru_model))
if run_lstm:
    print("LSTM model parameters:", count_params(lstm_model))
if run_lin:
    print("Linear model parameters:", count_params(linear_model))
if run_ligru:
    print("Li-GRU model parameters:", count_params(ligru_model))

################################################################################
#                8) PREDICTIONS & METRICS COMPUTATION
################################################################################
if run_gru:
    gru_preds_by_trial = get_trialwise_preds_rnn(gru_model, X_test_rnn, seq_len=k, device=device)
if run_lstm:
    lstm_preds_by_trial = get_trialwise_preds_rnn(lstm_model, X_test_rnn, seq_len=k, device=device)
if run_lin:
    linear_preds_by_trial = get_trialwise_preds_linear(linear_model, X_test_rnn, Y_test_rnn, seq_len=k, device=device)
if run_ligru:
    ligru_preds_by_trial = get_trialwise_preds_rnn(ligru_model, X_test_rnn, seq_len=k, device=device)

gru_pertrial, lstm_pertrial, lin_pertrial, ligru_pertrial = [], [], [], []
for i in range(len(X_test_rnn)):
    y_true_i = Y_test_rnn[i]
    if run_gru:
        m1 = compute_metrics(y_true_i, gru_preds_by_trial[i])
        gru_pertrial.append(m1)
    if run_lstm:
        m2 = compute_metrics(y_true_i, lstm_preds_by_trial[i])
        lstm_pertrial.append(m2)
    if run_lin:
        m3 = compute_metrics(y_true_i, linear_preds_by_trial[i])
        lin_pertrial.append(m3)
    if run_ligru:
        m4 = compute_metrics(y_true_i, ligru_preds_by_trial[i])
        ligru_pertrial.append(m4)
    print(f"\n----- Trial {i} Metrics -----")
    if run_gru:
        print("GRU   :", m1)
    if run_lstm:
        print("LSTM  :", m2)
    if run_lin:
        print("Linear:", m3)
    if run_ligru:
        print("Li-GRU:", m4)

if run_gru:
    gru_avg = average_metrics(gru_pertrial)
    print("\nGRU average metrics:", gru_avg)
if run_lstm:
    lstm_avg = average_metrics(lstm_pertrial)
    print("LSTM average metrics:", lstm_avg)
if run_lin:
    lin_avg = average_metrics(lin_pertrial)
    print("Linear average metrics:", lin_avg)
if run_ligru:
    ligru_avg = average_metrics(ligru_pertrial)
    print("Li-GRU average metrics:", ligru_avg)

if run_gru:
    gru_yT, gru_yP = flatten_all(Y_test_rnn, gru_preds_by_trial)
    gru_comb = compute_metrics(gru_yT, gru_yP)
    print("\nGRU combined metrics:", gru_comb)
if run_lstm:
    lstm_yT, lstm_yP = flatten_all(Y_test_rnn, lstm_preds_by_trial)
    lstm_comb = compute_metrics(lstm_yT, lstm_yP)
    print("LSTM combined metrics:", lstm_comb)
if run_lin:
    lin_yT, lin_yP = flatten_all(Y_test_rnn, linear_preds_by_trial)
    lin_comb = compute_metrics(lin_yT, lin_yP)
    print("Linear combined metrics:", lin_comb)
if run_ligru:
    ligru_yT, ligru_yP = flatten_all(Y_test_rnn, ligru_preds_by_trial)
    ligru_comb = compute_metrics(ligru_yT, ligru_yP)
    print("Li-GRU combined metrics:", ligru_comb)

################################################################################
#                           9) PLOTTING RESULTS
################################################################################
plot_n = min(len(X_test_rnn), 9)
fig, axes = plt.subplots(3, 3, figsize=(15,12), sharex=False)
axes = axes.flatten()
for i in range(plot_n):
    ax = axes[i]
    ax.set_title(f"Test Trial #{i+1}")
    ax.plot(Y_test_rnn[i], label="Actual", color='k')
    if run_gru:
        ax.plot(gru_preds_by_trial[i], label="GRU", color='r', ls='--')
    if run_lstm:
        ax.plot(lstm_preds_by_trial[i], label="LSTM", color='g', ls='--')
    if run_lin:
        ax.plot(linear_preds_by_trial[i], label="Linear", color='b', ls=':')
    if run_ligru:
        ax.plot(ligru_preds_by_trial[i], label="Li-GRU", color='m', ls='-.')
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-scored Force")
plt.tight_layout()
plt.savefig('decoder_res.png', dpi=700)
plt.show()
