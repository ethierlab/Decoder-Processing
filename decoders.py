import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from Neural_Decoding import WienerFilterDecoder, GRUDecoder, LSTMDecoder

# Load data from a pickle file (customize the filename)
with open("data.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Choose which representation to use ('PCA', 'UMAP', or 't-SNE')
representation = 'PCA'  # Change as needed ('PCA', 'UMAP', 't-SNE')

# Set parameters for augmentation
new_data_length = 200  # Length of new data segments (number of time bins)
overlap_percentage = 0.5  # Overlap percentage between segments (e.g., 0.5 for 50% overlap)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare the data
def prepare_data(saved_data, representation):
    X = []
    y = []
    trials = saved_data[representation]
    force_trials = saved_data['Force']
    trial_indices = sorted(trials.keys())

    for idx in trial_indices:
        X.append(trials[idx])  # Shape: (n_components, len(common_times))
        y.append(force_trials[idx])  # Shape: (1, len(common_times))

    # Convert lists to numpy arrays
    X = np.array(X)  # Shape: (num_trials, n_components, len(common_times))
    y = np.array(y)  # Shape: (num_trials, 1, len(common_times))

    # Transpose to get time steps in the correct dimension
    # Desired shape for PyTorch RNNs: (num_trials, seq_len, input_size)
    X = np.transpose(X, (0, 2, 1))  # Now shape: (num_trials, len(common_times), n_components)
    y = np.transpose(y, (0, 2, 1))  # Now shape: (num_trials, len(common_times), 1)

    return X, y

X, y = prepare_data(saved_data, representation)

# Data augmentation function to generate new forward-moving trials from each trial
def augment_data(X, y, new_data_length, overlap_percentage):
    augmented_X = []
    augmented_y = []

    bin_size = new_data_length  # Number of time steps in the new data segments
    overlap = overlap_percentage  # Overlap percentage between 0 and <1

    step_size = int(bin_size * (1 - overlap))
    if step_size <= 0:
        step_size = 1  # Ensure step_size is at least 1

    num_trials, trial_length, num_features = X.shape

    for i in range(num_trials):
        trial_X = X[i]
        trial_y = y[i]

        for start in range(0, trial_length - bin_size + 1, step_size):
            end = start + bin_size
            augmented_X.append(trial_X[start:end, :])
            augmented_y.append(trial_y[start:end, :])

    # Convert to tensors
    augmented_X = torch.tensor(augmented_X, dtype=torch.float32)
    augmented_y = torch.tensor(augmented_y, dtype=torch.float32)

    return augmented_X, augmented_y



# Augment the data
augmented_X, augmented_y = augment_data(X, y, new_data_length, overlap_percentage)

# Function for model selection and training with flexible parameters
def train_decoder(augmented_X, augmented_y, selected_decoders, decoder_params):
    models = {}
    results = {}

    input_size = augmented_X.shape[2]  # Number of features
    output_size = augmented_y.shape[2]  # Output dimension

    # Split into training and testing sets (e.g., 80% train, 20% test)
    total_samples = augmented_X.shape[0]
    train_size = int(0.8 * total_samples)

    X_train = augmented_X[:train_size].to(device)
    y_train = augmented_y[:train_size].to(device)
    X_test = augmented_X[train_size:].to(device)
    y_test = augmented_y[train_size:].to(device)

    # Initialize decoders based on selection and user-specified parameters
    if "wiener" in selected_decoders:
        models["wiener"] = WienerFilterDecoder()

    if "gru" in selected_decoders:
        gru_params = decoder_params.get("gru", {})
        models["gru"] = GRUDecoder(
            input_size=input_size,
            hidden_size=gru_params.get("hidden_size", 50),
            output_size=output_size,
            num_layers=gru_params.get("num_layers", 1),
            dropout=gru_params.get("dropout", 0.0)
        ).to(device)

    if "lstm" in selected_decoders:
        lstm_params = decoder_params.get("lstm", {})
        models["lstm"] = LSTMDecoder(
            input_size=input_size,
            hidden_size=lstm_params.get("hidden_size", 50),
            output_size=output_size,
            num_layers=lstm_params.get("num_layers", 1),
            dropout=lstm_params.get("dropout", 0.0)
        ).to(device)

    # Train each selected model
    for name, model in models.items():
        print(f"\nTraining {name} decoder on {device} with parameters: {decoder_params.get(name, {})}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            'predictions': predictions.cpu().detach().numpy(),
            'actual': y_test.cpu().detach().numpy()
        }
        print(f"{name} training complete.")

    return results

# Specify the decoders to use: 'wiener', 'gru', and/or 'lstm'
selected_decoders = ["wiener", "gru", "lstm"]

# Define decoder parameters
decoder_params = {
    "gru": {"hidden_size": 100, "num_layers": 2, "dropout": 0.1},
    "lstm": {"hidden_size": 100, "num_layers": 2, "dropout": 0.1}
}

# Train the selected decoders and retrieve results using augmented data
results = train_decoder(augmented_X, augmented_y, selected_decoders, decoder_params)

# Plotting predicted vs actual results for each decoder
def plot_results(results):
    for name, data in results.items():
        predictions = data['predictions']
        actual = data['actual']

        # Flatten the predictions and actual values for plotting
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        actual_flat = actual.reshape(-1, actual.shape[-1])

        plt.figure(figsize=(12, 6))
        plt.plot(actual_flat, label='Actual Force')
        plt.plot(pred_flat, label='Predicted Force')
        plt.title(f'{name} Decoder - Predicted vs Actual Force')
        plt.xlabel('Time Steps')
        plt.ylabel('Force')
        plt.legend()
        plt.show()

plot_results(results)
