import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader

# Data Loading and Preparation
with open("projected_data_test.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Data Loading and Preparation with Preprocessing
def prepare_data(saved_data, representation, cutoff, fs, n_components=16, order=5):
    X = []
    y = []
    trials = saved_data[representation] # Shape: (n_component, common_times)
    force_trials = saved_data['Force'] 
    trial_indices = sorted(trials.keys())

    for idx in trial_indices:
        X_trial = trials[idx][0:n_components] # Shape: (n_component, common_times)
        y_trial = force_trials[idx]  # Shape: (1, common_times)

        # Preprocess the force signal
        y_filtered = apply_lowpass_filter(y_trial.squeeze(), cutoff, fs, order)
        y_filtered = y_filtered.reshape(1, -1)  # Reshape back to (1, common_times)
        # Convert to grams
        y_filtered = (y_filtered - 294) * 1.95

        X.append(X_trial)
        y.append(y_filtered)

    X = np.array(X)  # Shape: (num_trials, n_component, common_times)
    y = np.array(y)  # Shape: (num_trials, 1, common_times)

    # Z-score normalize inputs
    X_zscored = np.zeros_like(X)
    for trial in range(X.shape[0]):
        for comp in range(X.shape[1]):
            X_zscored[trial, comp, :] = z_score(X[trial, comp, :])

    # Z-score normalize outputs
    y_zscored = np.zeros_like(y)
    for trial in range(y.shape[0]):
        y_zscored[trial, 0, :] = z_score(y[trial, 0, :])

    return X_zscored, y_zscored

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
# Z-score normalization function
def z_score(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0:
        return np.zeros_like(signal)
    return (signal - mean_val) / std_val
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    normalized = 2 * (signal - min_val) / (max_val - min_val) - 1
    return np.clip(normalized, -1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

X = []
n_components = 16
trials = saved_data['PCA'] 
trial_indices = sorted(trials.keys())

for idx in trial_indices:
    X_trial = trials[idx][0:n_components]
    X.append(X_trial)

X = np.array(X)
X_normalized = np.zeros_like(X)
for trial in range(X.shape[0]):
    for comp in range(X.shape[1]):
        X_normalized[trial, comp, :] = normalize_signal(X[trial, comp, :])
print('X shape:', X.shape)

# # Parameters for the weight distribution
# T = 16  # Number of time lags
# w_start = 0.1
# w_end = 1.0

# n_components = X.shape[1]
# w_base = np.linspace(w_start, w_end, T)  # Shape: (T,)
# w_true = np.tile(w_base, (n_components, 1))  # Shape: (n_components, T)

# mu = 0
# sigma = 0.05

num_trials, n_components, common_times = X.shape
y = np.zeros((num_trials, common_times))

# for trial in range(num_trials):
#     y_conv = np.zeros(common_times)
#     for c in range(n_components):
#         x_c = X[trial, c, :]
#         w_c = w_true[c]
#         y_conv_c = np.convolve(x_c, w_c, mode='same')
#         y_conv += y_conv_c

#     y_conv = normalize_signal(y_conv)
#     noise = np.random.normal(mu, sigma, size=y_conv.shape)
#     y_conv_with_noise = y_conv + noise
#     y[trial, :] = normalize_signal(y_conv_with_noise)

# print('y shape:', y.shape)
# print(X.shape, y.shape)

# ---------------------- NEW PART FOR LAG ----------------------
lag = 50  
X_list = []
y_list = []

for trial_idx in range(num_trials):
    for t in range(lag, common_times):
        X_window = X[trial_idx, :, t-lag:t]  # (n_components, lag)
        y_target = y[trial_idx, t]           # scalar
        X_list.append(X_window)
        y_list.append([y_target])

X_array = np.array(X_list)  # (num_samples, n_components, lag)
y_array = np.array(y_list)  # (num_samples, 1)

X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_array, dtype=torch.float32).to(device)
print("X_tensor shape (with lag):", X_tensor.shape)
print("y_tensor shape (with lag):", y_tensor.shape)
# --------------------------------------------------------------

cutoff = 10
fs = 1000

real_X, real_y = prepare_data(saved_data, representation="PCA", cutoff=cutoff, fs=fs)
print('real X shape:', real_X.shape)
print('real y shape:', real_y.shape)

# --- Repeat lag transformation for real data ---
real_X_list = []
real_y_list = []
r_num_trials, r_n_components, r_common_times = real_X.shape

for trial_idx in range(r_num_trials):
    for t in range(lag, r_common_times):
        X_window = real_X[trial_idx, :, t-lag:t]  # (n_components, lag)
        # real_y is (num_trials,1,common_times)
        # To get corresponding y, take y at time t:
        y_target = real_y[trial_idx, 0, t]
        real_X_list.append(X_window)
        real_y_list.append([y_target])

real_X_array = np.array(real_X_list)
real_y_array = np.array(real_y_list)
dataset = TensorDataset(X_tensor, y_tensor) 
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

real_X_tensor = torch.tensor(real_X_array, dtype=torch.float32).to(device)
real_y_tensor = torch.tensor(real_y_array, dtype=torch.float32).to(device)
print("real_X_tensor shape (with lag):", real_X_tensor.shape)
print("real_y_tensor shape (with lag):", real_y_tensor.shape)

# ---------------------- UPDATED MODEL ----------------------
class WienerCascadeDecoderLinear(nn.Module):
    def __init__(self, n_components, lag, hidden_dim=64):
        super().__init__()
        self.input_dim = n_components * lag  # Correctly calculate input dimension
        self.hidden_layer = nn.Linear(self.input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)
        # self.nonlinear = nn.Tanh()

    def forward(self, x):
        batch_size, n_components, lag = x.size()  # Ensure the input has the expected shape
        x_flat = x.view(batch_size, -1)  # Flatten to (batch_size, n_components * lag)
        x = self.hidden_layer(x_flat)
        x = self.activation(x)
        x = self.output_layer(x)
        # x = self.nonlinear(x)  # Apply non-linear squashing
        return x
# -----------------------------------------------------------

print("Input shape:", X_tensor.shape)    
print("Target shape:", y_tensor.shape)

# Initialize the models using the new lag-based input
model_linear = WienerCascadeDecoderLinear(n_components=n_components, lag=lag).to(device)  # <-- now using lag

criterion = nn.MSELoss()

def train_model(model, optimizer, X_tensor, y_tensor, num_epochs=1000):
    loss_values = []
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return loss_values, model

fake_model_linear = WienerCascadeDecoderLinear(n_components=n_components, lag=lag).to(device)


fake_optimizer_linear = torch.optim.Adam(fake_model_linear.parameters(), lr=0.001, weight_decay=1e-5)


fake_loss_linear, fake_model_linear = train_model(fake_model_linear, fake_optimizer_linear, X_tensor, y_tensor, num_epochs=500)

real_model_linear = WienerCascadeDecoderLinear(n_components=n_components, lag=lag).to(device)

real_optimizer_linear = torch.optim.Adam(real_model_linear.parameters(), lr=0.001, weight_decay=1e-5)

real_loss_linear, real_model_linear = train_model(real_model_linear, real_optimizer_linear, real_X_tensor, real_y_tensor, num_epochs=500)


fake_model_linear.eval()
real_model_linear.eval()

with torch.no_grad():
    # Note: now the model outputs are (batch_size,1), but we can reshape if needed
    # For plotting, we might need to index a specific sample.
    # We'll just pick the first sample for demonstration.

    fake_predicted_linear = fake_model_linear(X_tensor)  # shape: (num_samples,1)

    real_predicted_linear = real_model_linear(real_X_tensor)

# For plotting, we need to consider that we no longer have a full trial in one go.
# Each sample is just one time step. To reconstruct a trial, you'd need to stitch predictions back together.
# Here, let's just plot the first trial's reconstructed prediction by mapping samples back to time steps.
# This is more complex now because of the lag windowing. For simplicity, we'll just plot the first N samples.

# num_trials, common_times, lag
# from your data before you created X_array, y_array
num_samples_per_trial = common_times - lag

# Let's plot the force for the first 3 trials
trials_to_plot = [0]  # or any three trials you want

plt.figure(figsize=(12, 6))
colors = ['black', 'red', 'blue']  # choose colors for different trials
for i, t_idx in enumerate(trials_to_plot):
    start_idx = t_idx * num_samples_per_trial
    end_idx = (t_idx + 1) * num_samples_per_trial
    
    # Plot actual force for this trial
    # Shape of y_tensor: (num_samples, 1)
    actual_force = y_tensor[start_idx:end_idx, 0].cpu().numpy()
    plt.plot(actual_force, label=f"Trial {t_idx} Actual", color=colors[i], linewidth=1)

plt.xlabel("Time Steps (after lagging)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Actual Force for Trials")
plt.show()

# Similarly, plot predictions for these 3 trials
plt.figure(figsize=(12, 6))
for i, t_idx in enumerate(trials_to_plot):
    start_idx = t_idx * num_samples_per_trial
    end_idx = (t_idx + 1) * num_samples_per_trial

    actual_force = y_tensor[start_idx:end_idx, 0].cpu().numpy()
    predicted_force_linear = fake_predicted_linear[start_idx:end_idx, 0].cpu().numpy()

    # Plot actual
    plt.plot(actual_force, label=f"Trial {t_idx} Actual", color='black', linewidth=1)
    # Plot predictions
    plt.plot(predicted_force_linear, label=f"Trial {t_idx} Linear Prediction", alpha=0.7)

plt.xlabel("Time Steps (after lagging)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Fake Data: Actual vs Predicted for 3 Trials")
plt.show()

# For real data, do the same using real_y_tensor and real_predicted_conv/real_predicted_linear
plt.figure(figsize=(12, 6))
for i, t_idx in enumerate(trials_to_plot):
    start_idx = t_idx * num_samples_per_trial
    end_idx = (t_idx + 1) * num_samples_per_trial

    actual_force = real_y_tensor[start_idx:end_idx, 0].cpu().numpy()
    predicted_force_linear = real_predicted_linear[start_idx:end_idx, 0].cpu().numpy()

    plt.plot(actual_force, label=f"Trial {t_idx} Actual", color='black', linewidth=1)
    plt.plot(predicted_force_linear, label=f"Trial {t_idx} Linear Prediction", alpha=0.7)

plt.xlabel("Time Steps (after lagging)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.title("Real Data: Actual vs Predicted for 3 Trials")
plt.show()