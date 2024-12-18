import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
import os

#######################################
# Parameters
#######################################
file_path = 'Jango_dataset.pkl'
N = 16          # Number of PCA components
k = 16          # Lag length (sequence length)
hidden_dim = 64
num_epochs = 500
batch_size = 64
learning_rate = 0.001

# Filtering parameters (optional)
cutoff = 10    # Cutoff frequency in Hz
fs = 1000      # Sampling frequency in Hz
order = 5      # Order of the Butterworth filter

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
    """
    Create sliding windows of past 'k' time steps.
    X: (num_trials, T, N)  → Input features
    Y: (num_trials, T) or None → Target outputs (optional)
    k:                     → Number of past steps to include
    Returns:
        X_lagged: (total_windows, k, N)
        Y_lagged: (total_windows,) or None if Y is None
    """
    X_lagged = []
    Y_lagged = [] if Y is not None else None

    for trial in range(X.shape[0]):
        T = X.shape[1]
        for t in range(k, T):
            X_lagged.append(X[trial, t-k:t, :])  # Past k steps
            if Y is not None:
                Y_lagged.append(Y[trial, t])     # Predict current step

    X_lagged = np.array(X_lagged)
    if Y is not None:
        Y_lagged = np.array(Y_lagged)
    return X_lagged, Y_lagged

#######################################
# Step 1: Load Data
#######################################
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Extract PCA components and force data
trials = sorted(data['PCA'].keys())
X = np.stack([data['PCA'][trial][:N].T for trial in trials])  # Shape: (num_trials, T, N)
Y = np.stack([data['Force']['x'][trial] for trial in trials])  # Shape: (num_trials, T)

print("Original X shape:", X.shape, "Original Y shape:", Y.shape)

#######################################
# Step 2: Filter and Normalize Data (Optional filtering)
#######################################
# Y_filtered = np.array([apply_lowpass_filter(y, cutoff, fs, order) for y in Y])

# Z-score normalization
X_zscored = zscore(X, axis=(0, 1))
Y_zscored = zscore(Y, axis=(0, 1)) # or Y_filtered if you used filtering

#######################################
# Step 3: Split data by trials before lagging
# This ensures entire trials remain intact for testing
#######################################
num_trials = X_zscored.shape[0]
num_train = int(num_trials * 0.8)  # 80% train, 20% test
X_train_raw = X_zscored[:num_train]
Y_train_raw = Y_zscored[:num_train]
X_test_raw = X_zscored[num_train:]
Y_test_raw = Y_zscored[num_train:]

print("Train trials shape:", X_train_raw.shape, "Test trials shape:", X_test_raw.shape)

#######################################
# Step 4: Create Lagged Data from the raw sets
#######################################
X_train_lagged, Y_train_lagged = create_lagged_data(X_train_raw, Y_train_raw, k=k)
X_test_lagged, Y_test_lagged = create_lagged_data(X_test_raw, Y_test_raw, k=k)

print("Lagged X_train shape:", X_train_lagged.shape, "Lagged Y_train shape:", Y_train_lagged.shape)
print("Lagged X_test shape:", X_test_lagged.shape, "Lagged Y_test shape:", Y_test_lagged.shape)

#######################################
# Step 5: Convert to Tensors and DataLoader
#######################################
X_train_t = torch.tensor(X_train_lagged, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_lagged, dtype=torch.float32).unsqueeze(-1)

X_test_t = torch.tensor(X_test_lagged, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test_lagged, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_t, Y_test_t)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#######################################
# Step 6: Define the GRU Model
#######################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predict single output

    def forward(self, x):
        out, _ = self.gru(x)    # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]     # Take the last time step's output
        out = self.fc(out)      
        return out

input_dim = X_train_t.shape[2]
model = GRUModel(input_dim, hidden_dim, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Using device:', device)

#######################################
# Step 7: Train the Model
#######################################
model.train()
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training completed.")

#######################################
# Step 8: Save the model so you don't have to retrain every time
#######################################
# Save model state_dict
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to model_weights.pth.")

# Optionally, save a full checkpoint with optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'loss': losses[-1]
}
torch.save(checkpoint, 'checkpoint.pth')
print("Full checkpoint saved to checkpoint.pth.")

#######################################
# Step 9: Evaluate on the Test Set
#######################################
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

#######################################
# Step 10: Visualize Results for a Single Test Trial
#######################################
# Let's pick the first test trial (trial_idx=0 in test set)
trial_idx = 0
trial_X = X_test_raw[trial_idx]  # Shape: (T, N)
trial_Y = Y_test_raw[trial_idx]  # Shape: (T,)

# Re-create lagged data for this single test trial
trial_X_lagged, trial_Y_lagged = create_lagged_data(trial_X[np.newaxis,:,:], 
                                                    trial_Y[np.newaxis,:], 
                                                    k=k)
trial_X_tensor = torch.tensor(trial_X_lagged, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    trial_pred = model(trial_X_tensor).cpu().numpy().flatten()

# Align predictions to the original trial length
aligned_pred = np.zeros_like(trial_Y)
aligned_pred[k:] = trial_pred

# Plot actual vs predicted force (Z-scored units)
plt.figure(figsize=(12, 6))
plt.plot(trial_Y, label="Actual Force (Z-scored)")
plt.plot(aligned_pred, label="Predicted Force (Model)")
plt.xlabel("Time Steps")
plt.ylabel("Force (Z-scored)")
plt.title(f"Test Trial {trial_idx + 1}: Actual vs Predicted Force")
plt.legend()
plt.tight_layout()
plt.show()

# Also plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

#######################################
# Step 11: Loading the Model Later (Example)
#######################################
# If later you want to load the model without retraining:
# model = GRUModel(input_dim, hidden_dim, num_layers=1)
# model.load_state_dict(torch.load('model_weights.pth', map_location=device))
# model.to(device)
# model.eval()
# # Now you can run model predictions without retraining.
