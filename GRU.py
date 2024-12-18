import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
#######################################
# Parameters
#######################################
file_path = 'Jango_dataset.pkl'
# file_path = 'projected_data_test.pkl'
N = 16       # Number of components
representation = 'PCA'
batch_size = 64


# Filtering parameters
cutoff = 10    # cutoff frequency in Hz
fs = 1000       # sampling frequency in Hz
order = 5       # order of the Butterworth filter

#######################################
# Functions
#######################################
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# def create_sliding_windows(X, Y, window_size, step):
#     # X, Y: (num_trials, T, input_dim/output_dim)
#     windows_X = []
#     windows_Y = []
#     for trial_idx in range(X.shape[0]):
#         T = X.shape[1]
#         for start in range(0, T - window_size + 1, step):
#             end = start + window_size
#             windows_X.append(X[trial_idx, start:end, :])
#             windows_Y.append(Y[trial_idx, start:end, :])

#     # Convert to arrays
#     return np.array(windows_X), np.array(windows_Y)

#######################################
# Load Data
#######################################
with open(file_path, 'rb') as f:
    data = pickle.load(f)

trials = data[representation]
force_trials = data['Force']['x'] # Add ['x'] or ['y'] for Jango dataset
trial_indices = sorted(trials.keys())

X = []
y = []
for idx in trial_indices:
    X_trial = trials[idx][:N]
    y_trial = force_trials[idx]

    X.append(X_trial)
    y.append(y_trial)
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape, "Y shape:", y.shape)
#######################################
#Z-score Normalization
#######################################
X_zscored = np.array([zscore(trial, axis=1) for trial in X])
Y_zscored = np.array([zscore(trial) for trial in y])
# Y_zscored = zscore(Y_filtered, axis=0)

#######################################
#Z-score Split Data into Train/Test
#######################################

X_train, X_test, y_train, y_test = train_test_split(
        X_zscored, Y_zscored, test_size=0.2, random_state=42, shuffle=False
    )
print("X_train shape:", X_train.shape, "Y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "Y_test shape:", y_test.shape)

#######################################
# Convert to Tensors and DataLoader
#######################################
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)  

X_train = X_train.permute(0, 2, 1)  # (batch_size, T, N_component)
X_test = X_test.permute(0, 2, 1)  # (batch_size, T, N_component)
print("X_train tensor shape:", X_train.shape, "Y_train tensor shape:", y_train.shape)
print("X_test  tensor shape:", X_test.shape, "Y_test tensor shape:", y_test.shape)

# small_X = X_train[:10]
# small_Y = y_train[:10]

dataset = TensorDataset(X_train, y_train)
# dataset = TensorDataset(small_X, small_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


#######################################
# Define the Model
#######################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)  # (batch, seq, hidden)
        out = torch.relu(out)
        out = self.fc(out)
        return out


hidden_dim = 64
num_epochs = 500

learning_rate = 1e-4
input_dim = X_train.shape[2]
num_layers = 1
model = GRUModel(input_dim, hidden_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#######################################
# Step 8: Train the Model
#######################################
model.train()

# Initialize a list to store gradient norms
gradient_norms = []
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.1)
for epoch in range(num_epochs): 
    for batch_X, batch_Y in dataloader:
        # batch_X Should be (batch_size, window_size, input_dim)
        # batch_Y Should be (batch_size, window_size, 1)
        # print("Batch_Y Mean:", batch_Y.mean().item())
        # print("Batch_Y Std:", batch_Y.std().item())
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        # Compute gradient norms
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm of the gradient
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5  # Take the square root to get the L2 norm
        gradient_norms.append(total_norm)  # Store the gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional gradient clipping
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
        for name, param in model.named_parameters():
            print(f"{name}: Mean of weights: {param.data.mean():.4f}, Gradient Norm: {param.grad.norm() if param.grad is not None else 'No gradient'}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("Training completed.")


#######################################
# Step 9: Evaluate and Plot
#######################################
model.eval()
with torch.no_grad():
    pred = model(X_train).detach().cpu().numpy()
# print(pred.shape)
# print(pred)
plot_length = 1000
plt.figure(figsize=(12, 6))
plt.plot(y[0], label="Actual", linestyle='-', alpha=0.7)
plt.plot(pred[0], label="Predicted", linestyle='--', alpha=0.7)  # Predicted force signal
plt.xlabel("Time Steps (after lagging)")
plt.ylabel("Z-scored (Filtered & Converted) Force")
plt.title("Actual vs. Predicted Force (Filtered)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot gradient norms
plt.figure(figsize=(10, 6))
plt.plot(gradient_norms, label="Gradient Norms")
plt.xlabel("Training Steps")
plt.ylabel("Gradient Norm (L2)")
plt.title("Gradient Norms During Training")
plt.legend()
plt.grid(True)
plt.show()