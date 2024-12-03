import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Data Loading and Preparation
with open("projected_data_test.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Data Loading and Preparation with Preprocessing
def prepare_data(saved_data, representation, cutoff, fs, order=5):
    X = []
    y = []
    trials = saved_data[representation] # Shape: (n_component, common_times)
    force_trials = saved_data['Force'] 
    trial_indices = sorted(trials.keys())

    for idx in trial_indices:

        X_trial = trials[idx][0:8] # Shape: (n_component, common_times)
        y_trial = force_trials[idx]  # Shape: (1, common_times)
        # Preprocess the force signal
        # Apply low-pass filter
        y_filtered = apply_lowpass_filter(y_trial.squeeze(), cutoff, fs, order)
        y_filtered = y_filtered.reshape(1, -1)  # Reshape back to (1, common_times)
        # Convert to grams
        y_filtered = (y_filtered - 294) * 1.95  # Convert to grams

        X.append(X_trial)
        y.append(y_filtered)

    X = np.array(X)  # Shape: (num_trials, n_component, common_times)
    y = np.array(y)  # Shape: (num_trials, 1, common_times)
    # Normalize inputs and outputs
    X_normalized = np.zeros_like(X)
    for trial in range(X.shape[0]):
        for comp in range(X.shape[1]):
            X_normalized[trial, comp, :] = normalize_signal(X[trial, comp, :])

    y_normalized = np.zeros_like(y)
    for trial in range(y.shape[0]):
        y_normalized[trial, :] = normalize_signal(y[trial, :])
        print(y_normalized[trial, :])
    return X_normalized, y_normalized

# Define the low-pass filter function
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def normalize_signal(signal):

    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)  # Handle constant signals
    normalized = 2 * (signal - min_val) / (max_val - min_val) - 1
    
    # Clip values to ensure within [-1, 1] due to floating-point precision
    return np.clip(normalized, -1, 1)

# Test if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Data initialization
X = []

trials = saved_data['PCA'] 
trial_indices = sorted(trials.keys())

for idx in trial_indices:
    X_trial = trials[idx][0:8]   # Shape: (n_component, common_times)
    X.append(X_trial)

# Convert to numpy arrays
X = np.array(X)  # Shape: (num_trials, n_components, common_times)
# Normalize X
X_normalized = np.zeros_like(X)
for trial in range(X.shape[0]):
    for comp in range(X.shape[1]):
        X_normalized[trial, comp, :] = normalize_signal(X[trial, comp, :])
print('X shape:', X.shape)
# Parameters for the weight distribution
T = 7  # Number of time lags
w_start = 0.1
w_end = 1.0

# Generate linear weights over time lags
n_components = X.shape[1]
w_base = np.linspace(w_start, w_end, T)  # Shape: (T,)
w_true = np.tile(w_base, (n_components, 1))  # Shape: (n_components, T)
# Define noise parameters
mu = 0        # Mean of the noise
sigma = 0.05  # Standard deviation of the noise


num_trials, n_components, common_times = X.shape

# Initialize y
y = np.zeros((num_trials, common_times))


# Apply convolution and add noise
for trial in range(num_trials):
    y_conv = np.zeros(common_times)
    for c in range(n_components):
        x_c = X[trial, c, :]  # Shape: (common_times,)
        w_c = w_true[c]    # Shape: (T,)
        y_conv_c = np.convolve(x_c, w_c, mode='same')  # Shape: (common_times,)
        y_conv += y_conv_c

    # Normalize the signal after convolution
    y_conv = normalize_signal(y_conv)
    
    # Add noise (if desired)
    noise = np.random.normal(mu, sigma, size=y_conv.shape)
    y_conv_with_noise = y_conv + noise

    # Normalize again after adding noise
    y[trial, :] = normalize_signal(y_conv_with_noise)



print('y shape:', y.shape)

print(X.shape, y.shape)
# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)        # Shape: (num_trials,n_component, common_times)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)        # Shape: (num_trials, common_times)
y_tensor = y_tensor.unsqueeze(1) # Shape: (num_trials,n_comoponent (1), common_times)
print(X_tensor.shape)
# Add a dimention for reusability
# X_tensor = X_tensor.unsqueeze(1) # Shape: (num_trials,n_comoponent (1), common_times)


# Preparing Real Data
cutoff = 10  # Example cutoff frequency
fs = 1000    # Example sampling frequency

real_X, real_y = prepare_data(saved_data, representation="PCA", cutoff=cutoff, fs=fs)
print('real X shape:', real_X.shape)
print('real y shape:', real_y.shape)
# Convert real data to tensors
real_X_tensor = torch.tensor(real_X, dtype=torch.float32).to(device)
real_y_tensor = torch.tensor(real_y, dtype=torch.float32).to(device)


# Define the WienerCascadeDecoder with Conv
class WienerCascadeDecoderConv(nn.Module):
    def __init__(self, n_components, kernel_size):
        super(WienerCascadeDecoderConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=n_components,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.nonlinear = nn.Tanh()  # Options: nn.ReLU(), nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlinear(out)
        return out


# Define the WienerCascadeDecoder with Linear
class WienerCascadeDecoderLinear(nn.Module):
    def __init__(self, n_components, kernel_size, input_length):
        super(WienerCascadeDecoderLinear, self).__init__()
        self.linear = nn.Linear(n_components * input_length, input_length, bias=False)
        self.nonlinear = nn.Tanh()  # Options: nn.ReLU(), nn.LeakyReLU()

    def forward(self, x):
        batch_size, n_components, input_length = x.size()
        x_flattened = x.view(batch_size, -1)  # Flatten to (batch_size, n_components * input_length)
        out = self.linear(x_flattened)
        out = self.nonlinear(out)
        return out.view(batch_size, 1, input_length)  # Reshape to (batch_size, 1, input_length)

print("Input shape:", X_tensor.shape)    
print("Target shape:", y_tensor.shape)

# Initialize the model
n_components = X_tensor.shape[1]
kernel_size = T
input_length = X_tensor.shape[2]

# Initialize the model
model_conv = WienerCascadeDecoderConv(n_components=n_components, kernel_size=T).to(device)
model_linear = WienerCascadeDecoderLinear(n_components=n_components, kernel_size=T, input_length=input_length).to(device)

# model = WienerCascadeDecoder(n_components=n_components, kernel_size=kernel_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001, weight_decay=1e-5)
optimizer_linear = torch.optim.Adam(model_linear.parameters(), lr=0.001, weight_decay=1e-5)

# optimizer = optim.RMSprop(model.parameters(),lr=0.01)

def train_model(model, optimizer, X_tensor, y_tensor, num_epochs=1000):
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_values, model


# Train on fake data
fake_model_conv = WienerCascadeDecoderConv(n_components=n_components, kernel_size=T).to(device)
fake_model_linear = WienerCascadeDecoderLinear(n_components=n_components, kernel_size=T, input_length=input_length).to(device)

fake_optimizer_conv = torch.optim.Adam(fake_model_conv.parameters(), lr=0.001, weight_decay=1e-5)
fake_optimizer_linear = torch.optim.Adam(fake_model_linear.parameters(), lr=0.001, weight_decay=1e-5)

fake_loss_conv, fake_model_conv = train_model(fake_model_conv, fake_optimizer_conv, X_tensor, y_tensor, num_epochs=500)
fake_loss_linear, fake_model_linear = train_model(fake_model_linear, fake_optimizer_linear, X_tensor, y_tensor, num_epochs=500)

# Train on real data
real_model_conv = WienerCascadeDecoderConv(n_components=n_components, kernel_size=T).to(device)
real_model_linear = WienerCascadeDecoderLinear(n_components=n_components, kernel_size=T, input_length=input_length).to(device)

real_optimizer_conv = torch.optim.Adam(real_model_conv.parameters(), lr=0.001, weight_decay=1e-5)
real_optimizer_linear = torch.optim.Adam(real_model_linear.parameters(), lr=0.001, weight_decay=1e-5)

real_loss_conv, real_model_conv = train_model(real_model_conv, real_optimizer_conv, real_X_tensor, real_y_tensor, num_epochs=500)
real_loss_linear, real_model_linear = train_model(real_model_linear, real_optimizer_linear, real_X_tensor, real_y_tensor, num_epochs=500)

# Evaluate models
fake_model_conv.eval()
fake_model_linear.eval()
real_model_conv.eval()
real_model_linear.eval()

with torch.no_grad():
    fake_predicted_conv = fake_model_conv(X_tensor)
    fake_predicted_linear = fake_model_linear(X_tensor)
    real_predicted_conv = real_model_conv(real_X_tensor)
    real_predicted_linear = real_model_linear(real_X_tensor)



outputs = model_conv(X_tensor)
outputs = model_linear(X_tensor)

print("Output shape conv:", outputs.shape)
print("Output shape lin:", outputs.shape)
# Plot
trial_idx = 0
component_idx = 0

# Extract signal over time
signal_over_time = y_tensor[trial_idx,component_idx , :].cpu().numpy()

# Fake data predictions
fake_actual_y = y_tensor[trial_idx, component_idx, :].cpu().numpy()
fake_predicted_y_conv = fake_predicted_conv[trial_idx, component_idx, :].cpu().numpy()
fake_predicted_y_linear = fake_predicted_linear[trial_idx, component_idx, :].cpu().numpy()

# Real data predictions
real_actual_y = real_y_tensor[trial_idx, component_idx, :].cpu().numpy()
real_predicted_y_conv = real_predicted_conv[trial_idx, component_idx, :].cpu().numpy()
real_predicted_y_linear = real_predicted_linear[trial_idx, component_idx, :].cpu().numpy()

# Plot results
plt.figure(figsize=(12, 6))

# Fake Data
# plt.subplot(1, 2, 1)
plt.plot(fake_actual_y, label="Fake Actual Signal", color='black', linewidth=1)
plt.plot(fake_predicted_y_conv, label="Fake Conv1d Prediction", alpha=0.7)
plt.plot(fake_predicted_y_linear, label="Fake Linear Prediction", alpha=0.7)
plt.xlabel("Time Steps")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Fake Data: Actual vs Predicted")
plt.show()
# Real Data
# plt.subplot(1, 2, 2)
plt.figure(figsize=(12, 6))
plt.plot(real_actual_y, label="Real Actual Signal", color='black', linewidth=1)
plt.plot(real_predicted_y_conv, label="Real Conv1d Prediction", alpha=0.7)
plt.plot(real_predicted_y_linear, label="Real Linear Prediction", alpha=0.7)
plt.xlabel("Time Steps")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Real Data: Actual vs Predicted")
plt.show()
# Plot Loss functions
# Plot all loss curves
plt.figure(figsize=(12, 6))
plt.plot(fake_loss_conv, label="Fake Conv1d Loss", alpha=0.8, linestyle="--")
plt.plot(fake_loss_linear, label="Fake Linear Loss", alpha=0.8, linestyle=":")
plt.plot(real_loss_conv, label="Real Conv1d Loss", alpha=0.8, linestyle="-")
plt.plot(real_loss_linear, label="Real Linear Loss", alpha=0.8, linestyle="-.")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curves.png")
plt.show()

