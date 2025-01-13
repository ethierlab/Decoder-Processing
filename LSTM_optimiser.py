import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.stats import zscore
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
import itertools
import os

#######################################
# Functions
#######################################
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def create_lagged_data(X, Y=None, k=None):
    X_lagged = []
    Y_lagged = []

    for trial in range(X.shape[0]):
        T = X.shape[1]
        for t in range(k, T):
            X_lagged.append(X[trial, t-k:t, :])
            if Y is not None:
                Y_lagged.append(Y[trial, t])
    X_lagged = np.array(X_lagged)
    Y_lagged = np.array(Y_lagged) if Y is not None else None
    return X_lagged, Y_lagged

#######################################
# LSTM Model
#######################################
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM returns (output, (h_n, c_n)) by default
        out, (h, c) = self.lstm(x)
        # We only need the last time step’s output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_data_and_preprocess(file_path, N, k, train_split=0.8, cutoff=None, fs=None, order=5):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract PCA components and force
    trials = sorted(data['PCA'].keys())
    X = np.stack([data['PCA'][trial][:N].T for trial in trials])  # (num_trials, T, N)
    Y = np.stack([data['Force']['x'][trial] for trial in trials]) # (num_trials, T)

    # Optional low-pass filtering:
    if cutoff is not None and fs is not None:
        Y = np.array([apply_lowpass_filter(y, cutoff, fs, order) for y in Y])

    # Z-score normalization
    X_zscored = zscore(X, axis=(0,1))
    Y_zscored = zscore(Y, axis=(0,1))

    # Split into train/test by trials
    num_trials = X_zscored.shape[0]
    num_train = int(num_trials * train_split)
    X_train_raw = X_zscored[:num_train]
    Y_train_raw = Y_zscored[:num_train]
    X_test_raw = X_zscored[num_train:]
    Y_test_raw = Y_zscored[num_train:]

    # Create lagged data
    X_train_lagged, Y_train_lagged = create_lagged_data(X_train_raw, Y_train_raw, k=k)
    X_test_lagged, Y_test_lagged = create_lagged_data(X_test_raw, Y_test_raw, k=k)

    # Convert to tensors
    X_train_t = torch.tensor(X_train_lagged, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_lagged, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test_lagged, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test_lagged, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    return train_dataset, test_dataset, (X_test_raw, Y_test_raw), X_train_t.shape[2]

def compute_r2_vaf(y_true, y_pred):
    # R^2 calculation
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)

    # VAF calculation
    # VAF = 1 - var(error)/var(y_true)
    vaf = 1 - (np.var(y_true - y_pred) / np.var(y_true))

    return r2, vaf

#######################################
# Train and Evaluate with LSTM
#######################################
def train_and_evaluate_lstm(
    file_path, 
    N, 
    k, 
    hidden_dim, 
    num_epochs, 
    batch_size, 
    learning_rate, 
    train_split=0.8, 
    cutoff=None, 
    fs=None, 
    order=5, 
    device='cpu'
):
    # Load and preprocess data
    train_dataset, test_dataset, test_raw, input_dim = load_data_and_preprocess(
        file_path, N, k, train_split, cutoff, fs, order
    )
    X_test_raw, Y_test_raw = test_raw

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = LSTMModel(input_size=input_dim, hidden_size=hidden_dim, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Calculate number of trainable parameters (for complexity argument)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   -> LSTM Model with {num_trainable_params} trainable parameters.")
    print(f"   -> Starting training for {num_epochs} epochs...")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # (Optional) Print progress every certain number of epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"     Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            pred = model(batch_X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute test loss, R2, VAF
    test_loss = np.mean((all_preds - all_targets)**2)
    r2, vaf = compute_r2_vaf(all_targets, all_preds)

    print(f"   -> Final Test Loss: {test_loss:.6f}, R2: {r2:.3f}, VAF: {vaf:.3f}")
    return test_loss, r2, vaf, num_trainable_params

#######################################
# Running All Experiments with Incremental Saving (Example)
#######################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    file_path = 'Jango_dataset.pkl'
    output_file = 'experiment_results_lstm.pkl'

    # Parameter grids
    N_values = [i for i in range(1, 17)]        # Different numbers of PCA components
    k_values = [i for i in range(1, 17)]        # Different time lags
    hidden_dims = [16, 32, 64]  # Different hidden dimensions
    batch_sizes = [32,64]      # Different batch sizes
    num_epochs_list = [250] # Adjusted slightly for a quicker run
    learning_rates = [0.001, 0.0005]  # Different learning rates

    train_split = 0.8

    # Optional filtering parameters (set to None to skip filtering)
    cutoff = None
    fs = None
    order = 5

    # Initialize or load existing results
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {output_file}.")
    else:
        results = []
        print(f"No existing results found. Starting fresh.")

    # Create all combinations of parameters
    parameter_combinations = list(itertools.product(
        N_values,
        k_values,
        hidden_dims,
        batch_sizes,
        num_epochs_list,
        learning_rates
    ))

    total_experiments = len(parameter_combinations)
    completed_experiments = len(results)

    print(f"Total experiments to run: {total_experiments}")
    print(f"Experiments already completed: {completed_experiments}")

    # Iterate over all parameter combinations
    for idx, (N, k, hidden_dim, batch_size, num_epochs, learning_rate) in enumerate(parameter_combinations, start=1):
        # Skip already completed experiments
        if idx <= completed_experiments:
            print(f"Skipping experiment {idx}/{total_experiments} as it's already completed.")
            continue

        print(f"\nRunning experiment {idx}/{total_experiments} with parameters:")
        print(f"   -> N={N}, k={k}, hidden_dim={hidden_dim}, batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}")

        try:
            test_loss, r2, vaf, n_params = train_and_evaluate_lstm(
                file_path=file_path,
                N=N,
                k=k,
                hidden_dim=hidden_dim,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                train_split=train_split,
                cutoff=cutoff,
                fs=fs,
                order=order,
                device=device
            )

            # Compile experiment results
            exp_result = {
                'N': N,
                'k': k,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'train_split': train_split,
                'num_params': n_params,
                'test_loss': test_loss,
                'R2': r2,
                'VAF': vaf
            }

            results.append(exp_result)

            # Save results incrementally
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Experiment {idx}/{total_experiments} completed. Results saved.")

        except Exception as e:
            print(f"Experiment {idx}/{total_experiments} failed with error: {e}")
            # Optionally, you can choose to continue or break
            continue

    print(f"\nAll experiments completed. Final results saved to {output_file}.")

if __name__ == "__main__":
    main()
