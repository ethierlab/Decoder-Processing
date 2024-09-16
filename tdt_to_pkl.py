import tdt
import pickle
import os

# Chemin du fichier TDT (à ajuster selon votre fichier)
BLOCK_PATH = "T:/Projects/ForcePredictBMI/296/18-07-2024/Recording/296-240718-125555"
SAVING_PATH = "C:/Users/Ethier Lab/Desktop"

def extract_tdt_signals(block_path):
    """Extract specific signals from a TDT block."""
    # Lire le block de données
    data = tdt.read_block(block_path)
    
    # Extraire les signaux spécifiques
    signals = {
        'Levier': data.streams.Lev_.data,  # Données du signal Lev_
        'Event Time': data.scalars.T0__.ts    # Données du signal T0__
    }
    
    return signals

def save_to_pickle(data, save_path, filename="tdt_signals.pkl"):
    """Save the dictionary to a pickle file."""
    with open(os.path.join(save_path, filename), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # Extraire les signaux et les sauvegarder
    signals_data = extract_tdt_signals(BLOCK_PATH)
    save_to_pickle(signals_data, SAVING_PATH)