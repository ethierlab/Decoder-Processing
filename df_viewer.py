import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_combined_df(file_path):
    # Load the combined DataFrame from the pickle file
    try:
        combined_df = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print("Combined DataFrame loaded.")
    print("Shape:", combined_df.shape)
    print("\nFirst few rows:")
    print(combined_df.head())

    # For visualization, we use the first row (trial) as an example.
    trial = combined_df.iloc[0]

    # Extract the time axis. We assume this is stored under 'time_frame'
    if 'time_frame' not in trial:
        print("No 'time_frame' found in the trial data.")
        return

    time_frame = trial['time_frame']
    # Ensure time_frame is a 1D numpy array
    time_frame = np.array(time_frame).flatten()

    # # --- Visualize EMG Data ---
    # if 'EMG' in trial:
    #     emg_df = trial['EMG']
    #     # We assume emg_df is a DataFrame with columns as channel names
    #     plt.figure(figsize=(12, 6))
    #     for col in emg_df.columns:
    #         # Plot only a subset of the data (first 1000 samples) for clarity if needed
    #         plt.plot(time_frame[:1000], emg_df[col].values[:1000], label=col)
    #     plt.title("EMG Data (Trial 1)")
    #     plt.xlabel("Time")
    #     plt.ylabel("EMG amplitude")
    #     plt.legend(loc='upper right')
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("No EMG data available in this trial.")

    # # --- Visualize Spike Counts ---
    # if 'spike_counts' in trial:
    #     spike_counts = trial['spike_counts']
    #     # Assume spike_counts is a 2D array (units x samples)
    #     plt.figure(figsize=(12, 6))
    #     # Plot first unit for demonstration; adjust index as needed
    #     plt.plot(time_frame[:1000], spike_counts[0, :1000], label="Unit 1")
    #     plt.title("Spike Counts (Trial 1)")
    #     plt.xlabel("Time")
    #     plt.ylabel("Spike Count")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("No spike counts available in this trial.")

    # --- Visualize Force Data ---
    if 'force' in trial:
        force_data = trial['force']['x']
        # Assume force_data is a 2D array (channels x samples)
        plt.figure(figsize=(12, 6))
        # Plot first channel for demonstration; adjust index as needed
        plt.plot(time_frame[:1000], force_data[0, :1000], label="Force Channel 1")
        plt.title("Force Data (Trial 1)")
        plt.xlabel("Time")
        plt.ylabel("Force")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No force data available in this trial.")

if __name__ == '__main__':
    # Specify the path to your combined pickle file
    file_path = "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Spike_ISO_2012/combined.pkl"
    visualize_combined_df(file_path)
