import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
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

    # Compute global min and max for y in a single loop
    global_min = float('inf')
    global_max = float('-inf')

    for idx in trial_indices:
        X_trial = trials[idx][1:4]
        y_trial = force_trials[idx]

        # Convert to grams BEFORE filtering
        y_converted = (y_trial.squeeze() - 294) * 1.95

        # Apply low-pass filter
        y_filtered = apply_lowpass_filter(y_converted, cutoff, fs, order)

        # Update global min and max
        global_min = min(global_min, np.min(y_filtered))
        global_max = max(global_max, np.max(y_filtered))

        X.append(X_trial)
        y.append(y_filtered.reshape(1, -1))

    # Normalize y using symmetric scaling
    X = np.array(X)
    y = np.array(y)
    y_normalized = np.zeros_like(y)
    for trial in range(y.shape[0]):
        y_normalized[trial, :] = normalize_signal(y[trial, :], global_min, global_max)

    # Normalize X component-wise
    X_normalized = np.zeros_like(X)
    for trial in range(X.shape[0]):
        for comp in range(X.shape[1]):
            X_min = np.min(X[trial, comp, :])
            X_max = np.max(X[trial, comp, :])
            if X_max != X_min:
                X_normalized[trial, comp, :] = (
                    2 * (X[trial, comp, :] - X_min) / (X_max - X_min) - 1
                )
            else:
                X_normalized[trial, comp, :] = 0
        # After normalizing y
    print("y_normalized mean:", np.mean(y_normalized))
    print("y_normalized std:", np.std(y_normalized))
    print("y_normalized min:", np.min(y_normalized))
    print("y_normalized max:", np.max(y_normalized))

    # After normalizing X
    print("X_normalized mean:", np.mean(X_normalized))
    print("X_normalized std:", np.std(X_normalized))
    print("X_normalized min:", np.min(X_normalized))
    print("X_normalized max:", np.max(X_normalized))
    return X_normalized, y_normalized

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

def normalize_signal(signal, global_min, global_max):
    range_max = max(abs(global_min), abs(global_max))  # Use symmetric range around zero
    return signal / range_max  # Scale by the max absolute value


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

class WienerCascadeDecoder(nn.Module):
    def __init__(self, n_components, input_length):
        super(WienerCascadeDecoder, self).__init__()
        self.linear = nn.Linear(n_components * input_length, input_length, bias=False)
        self.nonlinear = nn.Tanh()  # Options: nn.ReLU(), nn.LeakyReLU() 

    def forward(self, x):
        batch_size, n_components, input_length = x.size()
        x_flattened = x.view(batch_size, -1)  # Flatten to (batch_size, n_components * input_length)
        out = self.linear(x_flattened)
        out = self.nonlinear(out) 
        return out.view(batch_size, 1, input_length)  # Reshape to (batch_size, 1, input_length)

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
    input_size = X_train.shape[1]
    input_length = X_train.shape[2]
    
    for decoder_name in selected_decoders:
        params = decoder_params.get(decoder_name, {})
        if decoder_name == 'wiener':
            model = WienerCascadeDecoder(input_size, input_length=input_length).to(device)
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
    perm = False
    for name, model in models.items():
        print(f"\nTraining {name} decoder...")
        params = decoder_params.get(name, {})
        num_epochs = params.get('num_epochs', 10)
        learning_rate = params.get('learning_rate', 0.001)

        # Dynamically set batch size based on dataset size
        batch_size = min(params.get('batch_size', 32), len(X_train))  # Ensure batch size <= dataset size
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Avoid incomplete batches
        )
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        if name in ['gru', 'lstm'] and perm == False:
            X_train = X_train.permute(0, 2, 1)  # (batch_size, seq_length, input_size)
            X_test = X_test.permute(0, 2, 1)  # (batch_size, seq_length, input_size)
            perm = True
        
        model.train()
        print(model)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                # Add these lines
                print("Outputs mean:", outputs.detach().cpu().numpy().mean())
                print("Targets mean:", y_batch.detach().cpu().numpy().mean())
                # print("Sample outputs:", outputs.detach().cpu().numpy()[0, :5])  # Print first 5 values
                # print("Sample targets:", y_batch.detach().cpu().numpy()[0, :5])  # Print first 5 values
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            test_loss = criterion(predictions, y_test)
            print(f"{name} decoder test loss: {test_loss.item():.4f}")

        predictions = predictions.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        results[name] = {
            'predictions': predictions.squeeze(),
            'actual': y_test_np.squeeze()
        }

    return results

def visualize_filtering(y_original, y_converted, y_filtered, trial_idx=0):
    plt.figure(figsize=(12, 6))
    plt.plot(y_original[trial_idx], label='Original Signal (Raw)', alpha=0.7)
    plt.plot(y_converted[trial_idx], label='Converted Signal (Grams)', alpha=0.7)
    plt.plot(y_filtered[trial_idx], label='Filtered Signal', alpha=0.7, linewidth=2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
    plt.title(f'Trial {trial_idx} - Filtering Process')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()



def visualize_predictions(y_test, predictions, epoch, decoder_name):
    num_samples = y_test.shape[0]
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(y_test[i].squeeze(), label="Actual Force", alpha=0.7)
        plt.plot(predictions[i].squeeze(), label="Predicted Force", alpha=0.7)
        plt.title(f"Sample {i + 1} - Epoch {epoch}")
        plt.xlabel("Time Steps")
        plt.ylabel("Force")
        plt.legend()
    plt.suptitle(f"{decoder_name} - Predictions vs. Actual at Epoch {epoch}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Visualization
def plot_results_sequence(results, samples_per_figure=9):
    for name, data in results.items():
        predictions = data['predictions']
        print(predictions.shape)
        actual = data['actual']
        # print(f"{name} Decoder - Predictions shape: {predictions.shape}, Actual shape: {actual.shape}")
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
cutoff_frequency = 50  # Cutoff frequency in Hz
sampling_rate = 1017.3   # Sampling rate in Hz
filter_order = 4       # Order of the Butterworth filter

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

# Augment the data
augmented_X, augmented_y = augment_data(X, y, segment_length, overlap_percentage)

# Decoder Parameters
decoder_params = {
    'wiener': {
        'num_epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    'gru': {
        'units': 50,
        'num_layers': 2,
        'dropout': 0,
        'num_epochs': 1000,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    'lstm': {
        'units': 50,
        'num_layers': 2,
        'dropout': 0,
        'num_epochs': 1000,
        'learning_rate': 0.001,
        'batch_size': 32
    }
}

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Train Decoders
selected_decoders = ['wiener']
# results = train_decoders(augmented_X, augmented_y, selected_decoders, decoder_params, device=device)
results=  train_decoders(X, y, selected_decoders, decoder_params, device=device)
# Plot the results
plot_results_sequence(results)
