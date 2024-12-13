import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load Data
file_path = 'projected_data_test.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Preprocess Data
N = 16  # Number of PCA components
# Concatenate PCA components into a TxN array (T: time steps, N: features)
X = np.hstack([data['PCA'][key] for key in sorted(data['PCA'].keys())])
X = X[:N, :].T

# Concatenate force data into a Tx1 array
Y_force = np.hstack([data['Force'][key] for key in sorted(data['Force'].keys())]).T

print(f"X shape: {X.shape}, Y_force shape: {Y_force.shape}")

# Create Lagged Data
def create_lagged_data(X, k=3):
    """
    Generate lagged inputs from a data matrix X.

    Args:
        X (np.ndarray): Input data of shape (T, N), where T is the number of time steps and N is the number of features.
        k (int): Number of lags.

    Returns:
        np.ndarray: Lagged input data of shape (T-k, k*N).
    """
    T, N = X.shape
    if T <= k:
        raise ValueError("Number of time steps (T) must be greater than the number of lags (k).")

    lagged_X = np.array([X[t - k:t].flatten() for t in range(k, T)], dtype=np.float32)
    return lagged_X

# Generate lagged and normalized inputs/targets
k = 16  # Number of time lags
lagged_X_np = zscore(create_lagged_data(X, k), axis=0)
lagged_Y_np = zscore(Y_force[k:], axis=0)

# Convert to PyTorch tensors
lagged_X = torch.tensor(lagged_X_np, dtype=torch.float32)
lagged_Y = torch.tensor(lagged_Y_np.squeeze(), dtype=torch.float32)

print(f"Lagged X shape: {lagged_X.shape}, Lagged Y shape: {lagged_Y.shape}")

# Define Neural Network Model
class TimeLaggedNonLinearOutputModel(nn.Module):
    """
    A neural network for predicting outputs based on time-lagged inputs.

    Args:
        input_dim (int): Dimension of the input layer.
        hidden_dim (int): Dimension of the hidden layer.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x) * 5  # Scale output to a desired range
        return x

# Initialize Model and Training Setup
input_dim = k * N
hidden_dim = 64
model = TimeLaggedNonLinearOutputModel(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 2000
for epoch in range(num_epochs):
    predictions = model(lagged_X).squeeze()
    loss = criterion(predictions, lagged_Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed.")

# Gradient Check
for name, param in model.named_parameters():
    grad_norm = param.grad.norm().item() if param.grad is not None else 0
    print(f"Gradient for {name}: {grad_norm:.6f}")

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(lagged_Y.numpy()[0:601-k], label="Actual Y", linestyle='dashdot', linewidth=2, alpha=0.7)
plt.plot(predictions.detach().numpy()[0:601-k], label="Predicted Y", linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Actual vs. Predicted Y")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lagged_Y.numpy(), label="Actual Y", linestyle='-', linewidth=2, alpha=0.7)
plt.plot(model(lagged_X).detach().numpy(), label="Predicted Y", linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Actual vs. Predicted Y")
plt.legend()
plt.show()