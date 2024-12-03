import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Data Loading and Preparation
with open("projected_data_test.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Data Loading and Preparation with Preprocessing
def prepare_data(saved_data, representation, cutoff, fs, order=5):
    X = []
    y = []
    trials = saved_data[representation]
    force_trials = saved_data['Force']
    trial_indices = sorted(trials.keys())

    for idx in trial_indices:
        X_trial = trials[idx].T  # Shape: (len(common_times), n_components)
        y_trial = force_trials[idx].T  # Shape: (len(common_times), 1)

        # Preprocess the force signal
        # Apply low-pass filter
        y_filtered = apply_lowpass_filter(y_trial.squeeze(), cutoff, fs, order)
        y_filtered = y_filtered.reshape(-1, 1)  # Reshape back to (len(common_times), 1)

        # Convert to grams
        y_filtered = (y_filtered - 294) * 1.95  # Convert to grams

        X.append(X_trial)
        y.append(y_filtered)

    X = np.array(X)  # Shape: (num_trials, len(common_times), n_components)
    y = np.array(y)  # Shape: (num_trials, len(common_times), 1)
    return X, y

# Define the low-pass filter function
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# Data Augmentation
def augment_data(X, y, segment_length, overlap_percentage):
    augmented_X = []
    augmented_y = []

    step_size = int(segment_length * (1 - overlap_percentage))
    if step_size <= 0:
        step_size = 1

    num_trials, trial_length, num_features = X.shape

    for trial_idx in range(num_trials):
        trial_X = X[trial_idx]
        trial_y = y[trial_idx]

        for start in range(0, trial_length - segment_length + 1, step_size):
            end = start + segment_length
            augmented_X.append(trial_X[start:end, :])
            augmented_y.append(trial_y[start:end, :])

    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    return augmented_X, augmented_y

# Model Definitions
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0):
        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class WienerFilterDecoder(nn.Module):
    def __init__(self, input_size):
        super(WienerFilterDecoder, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        x = x.view(-1, input_size)
        out = self.linear(x)
        out = out.view(batch_size, seq_len, 1)
        return out

# Training Function
def prepare_torch_data(augmented_X, augmented_y, device):
    X = torch.tensor(augmented_X, dtype=torch.float32).to(device)
    y = torch.tensor(augmented_y, dtype=torch.float32).to(device)
    return X, y

def train_decoders(augmented_X, augmented_y, selected_decoders, decoder_params, device='cpu'):

    X_train, X_test, y_train, y_test = train_test_split(
        augmented_X, augmented_y, test_size=0.2, random_state=42
    )

    X_train, y_train = prepare_torch_data(X_train, y_train, device)
    X_test, y_test = prepare_torch_data(X_test, y_test, device)

    models = {}
    input_size = X_train.shape[2]

    for decoder_name in selected_decoders:
        params = decoder_params.get(decoder_name, {})
        if decoder_name == 'wiener':
            model = WienerFilterDecoder(input_size=input_size).to(device)
        elif decoder_name == 'gru':
            model = GRUDecoder(
                input_size=input_size,
                hidden_size=params.get('units', 100),
                num_layers=params.get('num_layers', 1),
                dropout=params.get('dropout', 0.0)
            ).to(device)
        elif decoder_name == 'lstm':
            model = LSTMDecoder(
                input_size=input_size,
                hidden_size=params.get('units', 100),
                num_layers=params.get('num_layers', 1),
                dropout=params.get('dropout', 0.0)
            ).to(device)
        else:
            continue
        models[decoder_name] = model

    criterion = nn.MSELoss()

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} decoder...")
        params = decoder_params.get(name, {})
        num_epochs = params.get('num_epochs', 10)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        optimizer_name = params.get('optimizer', 'Adam')

        # Choose optimizer
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            print(f"Unknown optimizer {optimizer_name}, defaulting to Adam")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % (num_epochs // 5) == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            test_loss = criterion(predictions, y_test)
            print(f"{name} decoder test loss: {test_loss.item():.4f}")

        predictions = predictions.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        results[name] = {
            'predictions': predictions.squeeze(-1),
            'actual': y_test_np.squeeze(-1),
            'test_loss': test_loss.item()
        }

    return results


# Visualization
def plot_results_sequence(results, samples_per_figure=9):
    for name, data in results.items():
        predictions = data['predictions']
        actual = data['actual']

        num_test_samples = predictions.shape[0]
        num_figures = int(np.ceil(num_test_samples / samples_per_figure))

        for fig_num in range(num_figures):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            start_idx = fig_num * samples_per_figure
            end_idx = min((fig_num + 1) * samples_per_figure, num_test_samples)
            sample_indices = range(start_idx, end_idx)

            for ax, idx in zip(axes, sample_indices):
                ax.plot(actual[idx], label='Actual Force')
                ax.plot(predictions[idx], label='Predicted Force')
                ax.set_title(f'Sample {idx}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Force')
                ax.legend()

            for ax in axes[len(sample_indices):]:
                ax.axis('off')

            plt.suptitle(f'{name} Decoder - Predicted vs Actual Force (Samples {start_idx} to {end_idx - 1})', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


# Choose representation
representation = 'PCA'  # or 'UMAP', 't-SNE'

# Set parameters for the low-pass filter
cutoff_frequency = 10  # Cutoff frequency in Hz
sampling_rate = 1017.3   # Sampling rate in Hz
filter_order = 5       # Order of the Butterworth filter

# Prepare data with preprocessing
X, y = prepare_data(
    saved_data,
    representation,
    cutoff=cutoff_frequency,
    fs=sampling_rate,
    order=filter_order
)

# Choose segment length and overlap percentage
segment_length = 400 # Segment length in number of time steps
overlap_percentage = 0.5 # Overlap percentage
augment = False  # Set to True or False based on your choice

if augment:
    # Augment the data
    augmented_X, augmented_y = augment_data(X, y, segment_length, overlap_percentage)
else:
    augmented_X, augmented_y = X, y


# Decoder Parameters
default_decoder_params = {
    'wiener': {
        'num_epochs': 10000,
        'learning_rate': 0.05,
        'batch_size': 32
    },
    'gru': {
        'units': 50,
        'num_layers': 1,
        'dropout': 0,
        'num_epochs': 10000,
        'learning_rate': 0.00001,
        'batch_size': 32
    },
    'lstm': {
        'units': 50,
        'num_layers': 1,
        'dropout': 0,
        'num_epochs': 10000,
        'learning_rate': 0.0001,
        'batch_size': 32
    }
}

# Parameters to Vary
params_to_vary = {
    'optimizer': ['SGD', 'Adam', 'RMSprop'],
    'num_epochs': [1000, 5000, 10000],
    'learning_rate': [0.1, 0.01, 0.001],
    'units': [25, 50, 100],
    'num_layers': [1, 2, 3]
}


# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Train Decoders
selected_decoders = ['wiener', 'gru', 'lstm']
best_results = {}

for decoder in selected_decoders:
    print(f"\nParameter sweep for {decoder} decoder...")

    default_params = default_decoder_params[decoder]

    best_loss = float('inf')
    best_params = deepcopy(default_params)

    for param in params_to_vary:
        values = params_to_vary[param]

        for value in values:
            print(f"\nTraining {decoder} with {param} = {value}")

            # Create a copy of default parameters
            current_params = deepcopy(default_params)
            # Update the parameter being varied
            current_params[param] = value

            # For other parameters, keep them at default values
            decoder_params = {decoder: current_params}

            results = train_decoders(augmented_X, augmented_y, [decoder], decoder_params, device=device)

            # Get test_loss
            test_loss = results[decoder]['test_loss']

            print(f"Test loss for {param} = {value}: {test_loss}")

            # Update best parameters if test_loss improved
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = deepcopy(current_params)
                print(f"New best test loss: {best_loss}")

    # Store best parameters for the decoder
    best_results[decoder] = {
        'best_loss': best_loss,
        'best_params': best_params
    }

# Display Best Results
for decoder in selected_decoders:
    print(f"\nBest results for {decoder} decoder:")
    print(f"Test Loss: {best_results[decoder]['best_loss']}")
    print(f"Parameters: {best_results[decoder]['best_params']}")

# Plot the best results
for decoder in selected_decoders:
    # Use best parameters to retrain and get predictions
    best_params = {decoder: best_results[decoder]['best_params']}
    results = train_decoders(augmented_X, augmented_y, [decoder], best_params, device=device)
    plot_results_sequence(results)