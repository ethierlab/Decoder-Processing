import pickle
import torch
from neural_decoding import WienerFilterDecoder, GRUDecoder, LSTMDecoder

# Load data from a pickle file (customize the filename)
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function for model selection and training with flexible parameters
def train_decoder(data, selected_decoders, decoder_params):
    models = {}
    results = {}

    # Initialize decoders based on selection and user-specified parameters
    if "wiener" in selected_decoders:
        models["wiener"] = WienerFilterDecoder()  # Wiener decoder typically has fewer hyperparameters
    
    if "gru" in selected_decoders:
        gru_params = decoder_params.get("gru", {})
        models["gru"] = GRUDecoder(
            input_size=data['input_size'],
            hidden_size=gru_params.get("hidden_size", 50),
            output_size=data['output_size'],
            num_layers=gru_params.get("num_layers", 1),
            dropout=gru_params.get("dropout", 0.0)
        ).to(device)
        
    if "lstm" in selected_decoders:
        lstm_params = decoder_params.get("lstm", {})
        models["lstm"] = LSTMDecoder(
            input_size=data['input_size'],
            hidden_size=lstm_params.get("hidden_size", 50),
            output_size=data['output_size'],
            num_layers=lstm_params.get("num_layers", 1),
            dropout=lstm_params.get("dropout", 0.0)
        ).to(device)

    # Train each selected model
    for name, model in models.items():
        print(f"Training {name} decoder on {device} with parameters: {decoder_params.get(name, {})}")
        X_train, y_train = torch.tensor(data['X_train']).to(device), torch.tensor(data['y_train']).to(device)
        X_test = torch.tensor(data['X_test']).to(device)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = predictions.cpu()  # Move back to CPU for storage/analysis
        print(f"{name} training complete.")

    return results

# Specify the decoders to use: 'wiener', 'gru', and/or 'lstm'
selected_decoders = ["wiener", "gru", "lstm"]

# Define decoder parameters
decoder_params = {
    "gru": {"hidden_size": 100, "num_layers": 2, "dropout": 0.1},
    "lstm": {"hidden_size": 100, "num_layers": 2, "dropout": 0.1}
}

# Train the selected decoders and retrieve results
results = train_decoder(data, selected_decoders, decoder_params)
