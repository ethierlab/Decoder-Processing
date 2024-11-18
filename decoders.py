import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from Neural_Decoding import WienerCascadeDecoder, GRUDecoder, LSTMDecoder

# Load data from a pickle file (customize the filename)
with open("projected_data_test.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Choose which representation to use ('PCA', 'UMAP', or 't-SNE')
representation = 'PCA'  # Change as needed ('PCA', 'UMAP', 't-SNE')

# Prepare data
def prepare_data(saved_data, representation):
    X = []
    y = []
    trials = saved_data[representation]
    force_trials = saved_data['Force']
    trial_indices = sorted(trials.keys())

    for idx in trial_indices:
        X.append(trials[idx])  # Shape: (n_components, len(common_times))
        y.append(force_trials[idx])  # Shape: (Force, len(common_times))

    # Convert to NumPy arrays
    X = np.array(X)  # Shape: [num_trials, n_components, len(common_times)]
    y = np.array(y)  # Shape: [num_trials, Force, len(common_times)]

    print("Shapes after loading:")
    print(f"X shape: {X.shape}")  # Expected: [num_trials, n_components, len(common_times)]
    print(f"y shape: {y.shape}")  # Expected: [num_trials, Force, len(common_times)]

    # Transpose
    X = np.transpose(X, (0, 2, 1))  # Shape: [num_trials, len(common_times), n_components]
    y = np.transpose(y, (0, 2, 1))  # Shape: [num_trials, len(common_times), Force]

    print("Shapes after transposing:")
    print(f"X shape: {X.shape}")  # Expected: [num_trials, len(common_times), n_components]
    print(f"y shape: {y.shape}")  # Expected: [num_trials, len(common_times), Force]
    # If y has last dimension of size 1, squeeze it
    y = y.squeeze(-1)  # Shape: [num_trials, len(common_times)]
    print("Shape of y after squeezing:")
    print(f"y shape: {y.shape}")  # Expected: [num_trials, len(common_times)]

    return X, y

# **Call the prepare_data function to parse the data**
X, y = prepare_data(saved_data, representation)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation function to generate new forward-moving trials from each trial
def augment_data(X, y, segment_length, overlap_percentage):
    augmented_X = []
    augmented_y = []

    step_size = int(segment_length * (1 - overlap_percentage))
    if step_size <= 0:
        step_size = 1  # Ensure step_size is at least 1 to avoid infinite loops

    num_trials, trial_length, num_features = X.shape


    for i in range(num_trials):
        trial_X = X[i]
        trial_y = y[i]

        # Generate segments from this trial
        for start in range(0, trial_length - segment_length + 1, step_size):
            end = start + segment_length
            augmented_X.append(trial_X[start:end, :])
            augmented_y.append(trial_y[start:end])

    # Convert lists to NumPy arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    print("After data augmentation:")
    print(f"augmented_X shape: {augmented_X.shape}")  # Expected: [num_augmented_samples, new_data_length, num_features]
    print(f"augmented_y shape: {augmented_y.shape}")  # Expected: [num_augmented_samples, new_data_length]
    print(f"Number of augmented samples: {augmented_X.shape[0]}")
    return augmented_X, augmented_y

# Set parameters for augmentation
segment_length = 100  # Number of time steps in each snippet
overlap_percentage = 0.5  # 50% overlap between snippets

# Augment the data
augmented_X, augmented_y = augment_data(X, y, segment_length, overlap_percentage)

# Function for model selection and training with flexible parameters
def train_decoder(augmented_X, augmented_y, selected_decoders, decoder_params):
    models = {}
    results = {}

    # Split into training and testing sets
    total_samples = augmented_X.shape[0]
    train_size = int(0.8 * total_samples)

    X_train = augmented_X[:train_size]
    y_train = augmented_y[:train_size]
    X_test = augmented_X[train_size:]
    y_test = augmented_y[train_size:]


    print(f"augmented_X shape: {X_train.shape}")  # Expected: [num_augmented_samples, new_data_length, num_features]
    print(f"augmented_y shape: {y_train.shape}") 
    print(f"augmented_X shape: {X_test.shape}")  # Expected: [num_augmented_samples, new_data_length, num_features]
    print(f"augmented_y shape: {y_test.shape}") 
    # Initialize decoders based on selection and user-specified parameters
    if "wiener" in selected_decoders:
        models["wiener"] = WienerCascadeDecoder()

    if "gru" in selected_decoders:
        gru_params = decoder_params.get("gru", {})
        models["gru"] = GRUDecoder(
            units=gru_params.get("units", 100),
            dropout=gru_params.get("dropout", 0),
            num_epochs=gru_params.get("num_epochs", 1000),
            verbose=gru_params.get("verbose", 1)
        )

    if "lstm" in selected_decoders:
        lstm_params = decoder_params.get("lstm", {})
        models["lstm"] = LSTMDecoder(
            units=lstm_params.get("units", 100),
            dropout=lstm_params.get("dropout", 0.1),
            num_epochs=lstm_params.get("num_epochs", 10),
            verbose=lstm_params.get("verbose", 1)
        )

    # Train each selected model
    for name, model in models.items():
        print(f"\nTraining {name} decoder with parameters: {decoder_params.get(name, {})}")

        if name == "wiener":
            # For WienerFilterDecoder, flatten X_train and X_test
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            model.fit(X_train_flat, y_train)
            predictions = model.predict(X_test_flat)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        # **Add print statements to check predictions and actual data**
        print(f"{name} predictions shape: {predictions.shape}")
        print(f"{name} actual data shape: {y_test.shape}")
        print(f"{name} predictions sample values: {predictions[:5]}")
        print(f"{name} actual data sample values: {y_test[:5]}")

        results[name] = {
            'predictions': predictions,
            'actual': y_test
        }
        print(f"{name} training complete.")

    return results


# Specify the decoders to use: 'wiener', 'gru', and/or 'lstm'
selected_decoders = [ "lstm"]

# Define decoder parameters
decoder_params = {
    "gru": {"units": 32, "dropout": 0, "num_epochs": 200, "verbose": 1},
    "lstm": {"units": 100, "dropout": 0.1, "num_epochs": 100, "verbose": 1}
}


# **Call the train_decoder function to train the models**
results = train_decoder(augmented_X, augmented_y, selected_decoders, decoder_params)

# Plotting predicted vs actual results for each decoder
def plot_results(results, samples_per_figure=9):
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

            # Hide any unused subplots
            for ax in axes[len(sample_indices):]:
                ax.axis('off')

            plt.suptitle(f'{name} Decoder - Predicted vs Actual Force (Samples {start_idx} to {end_idx - 1})', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


# **Call the plot_results function to visualize the results**
plot_results(results)
