import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from scipy.stats import zscore
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
import itertools
import os

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

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_lagged_data(X, Y, k):
    X_lagged, Y_lagged = [], []
    num_trials = X.shape[0]
    for trial in range(num_trials):
        T = X.shape[1]
        for t in range(k, T):
            X_lagged.append(X[trial, t-k:t, :])
            Y_lagged.append(Y[trial, t])
    return np.array(X_lagged), np.array(Y_lagged)

def load_data_and_preprocess(file_path, N, k, train_split=0.8, cutoff=None, fs=None, order=5):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    reduction = 'PCA'
    trials = sorted(data[reduction].keys())
    X = np.stack([data[reduction][trial][:N].T for trial in trials])
    Y = np.stack([data['Force']['x'][trial] for trial in trials])
    if cutoff is not None and fs is not None:
        Y = np.array([apply_lowpass_filter(y, cutoff, fs, order) for y in Y])
    for i in range(len(X)):
        X[i] = zscore(X[i], axis=0)
        Y[i] = zscore(Y[i])
    num_trials = X.shape[0]
    num_train = int(num_trials * train_split)
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test,  Y_test  = X[num_train:], Y[num_train:]
    X_train_lag, Y_train_lag = create_lagged_data(X_train, Y_train, k)
    X_test_lag, Y_test_lag = create_lagged_data(X_test, Y_test, k)
    X_train_t = torch.tensor(X_train_lag, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_lag, dtype=torch.float32).unsqueeze(-1)
    X_test_t  = torch.tensor(X_test_lag, dtype=torch.float32)
    Y_test_t  = torch.tensor(Y_test_lag, dtype=torch.float32).unsqueeze(-1)
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    test_dataset  = TensorDataset(X_test_t, Y_test_t)
    input_dim = X_train_t.shape[2]
    return train_dataset, test_dataset, (X_test, Y_test), input_dim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def count_lstm_params(input_size, hidden_size, num_layers=1):
    model = LSTMModel(input_size, hidden_size, num_layers)
    return sum(p.numel() for p in model.parameters())

def compute_r2_vaf(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    vaf = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    return r2, vaf

def train_and_evaluate_lstm(file_path, N, k, hidden_dim, num_epochs, batch_size,
                            learning_rate, train_split=0.8, cutoff=None, fs=None, order=5,
                            device='cpu'):
    train_dataset, test_dataset, test_raw, input_dim = load_data_and_preprocess(
        file_path, N, k, train_split, cutoff, fs, order)
    X_test_raw, Y_test_raw = test_raw
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size=input_dim, hidden_size=hidden_dim, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            pred = model(batch_X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    test_loss = np.mean((all_preds - all_targets)**2)
    r2, vaf = compute_r2_vaf(all_targets, all_preds)
    param_count = count_lstm_params(input_dim, hidden_dim, num_layers=1)
    return test_loss, r2, vaf, param_count

################################################################################
#                           Experiment Loop (with checkpointing)
################################################################################
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    file_path = 'Jango_dataset.pkl'
    output_file = 'experiment_results_lstm_seeds.pkl'
    
    # Parameter grid
    N_values = list(range(1, 17))
    k_values = list(range(10, 17))
    hidden_dims = list(range(1, 65))
    batch_sizes = [64]
    num_epochs_list = [250]
    learning_rates = [0.001]
    train_split = 0.8
    cutoff, fs, order = None, None, 5
    num_runs = 25  # number of runs per parameter combination

    parameter_combinations = list(itertools.product(N_values, k_values, hidden_dims,
                                                     batch_sizes, num_epochs_list, learning_rates))
    
    # Load previous results (and use them as checkpoint)
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = []
    
    total_experiments = len(parameter_combinations) * num_runs
    
    # Determine where to resume based on the number of experiments already saved.
    num_completed = len(results)
    start_config_index = num_completed // num_runs  # which parameter combination to start at
    start_run_index = num_completed % num_runs        # which run in that configuration

    print(f"Resuming from experiment #{num_completed + 1} (config index {start_config_index}, run {start_run_index + 1}).")
    exp_counter = num_completed

    # Loop over parameter combinations starting from the checkpoint
    for config_index in range(start_config_index, len(parameter_combinations)):
        config = parameter_combinations[config_index]
        N, k, hidden_dim, batch_size, num_epochs, learning_rate = config
        # For the first configuration, resume from the saved run; otherwise, start at 0.
        current_run_start = start_run_index if config_index == start_config_index else 0

        for run in range(current_run_start, num_runs):
            current_seed = random.randint(0, 1000000)
            set_seed(current_seed)
            print(f"LSTM Config: {config_index+1}/{len(parameter_combinations)} "
                  f"(N={N}, k={k}, hidden_dim={hidden_dim}, batch_size={batch_size}, "
                  f"num_epochs={num_epochs}, lr={learning_rate}) | "
                  f"Run {run+1}/{num_runs} | Seed {current_seed}")
            try:
                test_loss, r2, vaf, param_count = train_and_evaluate_lstm(
                    file_path, N, k, hidden_dim, num_epochs, batch_size, learning_rate,
                    train_split, cutoff, fs, order, device)
                exp_result = {
                    'N': N,
                    'k': k,
                    'hidden_dim': hidden_dim,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'seed': current_seed,
                    'test_loss': test_loss,
                    'R2': r2,
                    'VAF': vaf,
                    'num_params': param_count
                }
                results.append(exp_result)
                exp_counter += 1
                with open(output_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Experiment {exp_counter}/{total_experiments} saved.")
            except Exception as e:
                print(f"Experiment failed: {e}")
                continue

    print(f"All LSTM experiments completed. Results saved to {output_file}.")
