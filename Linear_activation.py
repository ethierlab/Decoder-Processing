import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt

#######################################
# Parameters
#######################################
file_path = 'Jango_dataset.pkl'
# file_path = 'projected_data_test.pkl'
N = 16       # Number of PCA components
k = 16       # Lag length
hidden_dim = 64
num_epochs = 500
batch_size = 64
learning_rate = 0.001

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

def create_lagged_data(X, k):
    # X: (T, N), Y: (T,)
    # Returns lagged_X: (T-k, k*N), lagged_Y: (T-k,)
    T, N = X.shape
    if T <= k:
        raise ValueError("T must be greater than k")

    lagged_X = np.array([X[t - k:t].flatten() for t in range(k, T)], dtype=np.float32)
    return lagged_X

#######################################
# Step 1: Load Data
#######################################
with open(file_path, 'rb') as f:
    data = pickle.load(f)

#######################################
# Step 2: Concatenate All Trials
#######################################

# Concatenate PCA components into a TxN array (T: time steps, N: features)
X = np.hstack([data['PCA'][key] for key in sorted(data['PCA'].keys())])
X = X[:N, :].T

# Concatenate force data into a Tx1 array
Y_force = np.hstack([data['Force']['x'][key] for key in sorted(data['Force']['x'].keys())]).T


print("X shape:", X.shape, "Y shape:", Y_force.shape)
# Example: X shape: (19232, 16), Y shape: (19232,)

#######################################
# Step 3: Filter and Convert Y Data
#######################################
# Apply low-pass filter to Y before normalization
# Convert to grams
# Y_filtered = apply_lowpass_filter(Y_force, cutoff, fs, order=order)
# Y_filtered = (Y_filtered - 294) * 1.95  # Convert to grams if desired

#######################################
# Step 4: Z-score Normalization
#######################################
X_zscored = zscore(X, axis=0)
Y_zscored = zscore(Y_force, axis=0)
# Y_zscored = zscore(Y_filtered, axis=0)

#######################################
# Step 5: Create Lagged Data
#######################################
lagged_X_np = create_lagged_data(X_zscored, k)
lagged_Y_np = Y_force[k:].squeeze()
print("Lagged X shape:", lagged_X_np.shape, "Lagged Y shape:", lagged_Y_np.shape)


#######################################
# Step 6: Convert to Tensors and DataLoader
#######################################
lagged_X = torch.tensor(lagged_X_np, dtype=torch.float32)
lagged_Y = torch.tensor(lagged_Y_np, dtype=torch.float32)

dataset = TensorDataset(lagged_X, lagged_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#######################################
# Step 7: Define the Model
#######################################
class TimeLaggedNonLinearOutputModel(nn.Module):
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

input_dim = k * N
model = TimeLaggedNonLinearOutputModel(input_dim, hidden_dim=hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#######################################
# Step 8: Train the Model
#######################################
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_Y in dataloader:
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed.")

#######################################
# Step 9: Evaluate and Plot
#######################################
model.eval()
with torch.no_grad():
    pred = model(lagged_X).squeeze().detach().numpy()

plot_length = -1
plt.figure(figsize=(12, 6))
plt.plot(lagged_Y_np[:plot_length], label="Actual", linestyle='-', alpha=0.7)
plt.plot(pred[:plot_length], label="Predicted", linestyle='--', alpha=0.7)
plt.xlabel("Time Steps (after lagging)")
plt.ylabel("Z-scored (Filtered & Converted) Force")
plt.title("Actual vs. Predicted Force (Filtered)")
plt.legend()
plt.tight_layout()
plt.show()
