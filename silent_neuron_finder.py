import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Settings: Update the file path as needed ---
combined_pickle_path = "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Spike_ISO_2012/combined.pkl"

# --- Load the Combined DataFrame ---
combined_df = pd.read_pickle(combined_pickle_path)
print("Combined DataFrame shape:", combined_df.shape)

# Iterate over each row and inspect the trial_start_time field
for idx, row in combined_df.iterrows():
    tst = row['trial_start_time']
    
    # Check if tst is an array or list and if it has more than one element.
    if isinstance(tst, (np.ndarray, list)):
        if len(tst) > 1:
            print(f"Row {idx} has multiple recordings (trial_start_time has length {len(tst)}): {tst}")
    else:
        # If it's not an array, then it's likely a single recording.
        pass
print('loop complete')
# --- Ensure the date column is in datetime format ---
if not np.issubdtype(combined_df['date'].dtype, np.datetime64):
    combined_df['date'] = pd.to_datetime(combined_df['date'], format="%Y/%m/%d")

# --- Identify Day 0 (earliest date) ---
day0_date = combined_df['date'].min()
day0_df = combined_df[combined_df['date'] == day0_date]
print(f"Found {len(day0_df)} row(s) for Day 0 (date: {day0_date}).")

if len(day0_df) == 0:
    print("No Day 0 data found.")
else:
    # --- Extract spike_counts DataFrame from Day 0 ---
    spike_df = day0_df.iloc[0]['spike_counts']
    print("Spike DataFrame shape (samples x neurons):", spike_df.shape)
    
    # --- Compute per-neuron (column) standard deviations ---
    neuron_std = spike_df.std(axis=0)
    
    # --- Identify silent neurons (std == 0) ---
    silent_neurons = neuron_std[neuron_std == 0].index.tolist()
    print("Silent neurons on Day 0:", silent_neurons)
    
    # --- Plot a histogram of standard deviations for all neurons ---
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(neuron_std, bins=20, edgecolor='black')
    plt.xlabel("Standard Deviation")
    plt.ylabel("Number of Neurons")
    plt.title("Histogram of Neuron Standard Deviations (Day 0)")
    
    # Annotate the histogram for silent neurons
    # Find the bin corresponding to 0
    for neuron in silent_neurons:
        # For each silent neuron, annotate its name near x=0.
        plt.text(0, n[0] * 0.9, f"{neuron}", rotation=90, verticalalignment='center', color='red')
    
    plt.show()
    
    # --- Optionally, plot time-series for a few silent neurons (if any) ---
    if silent_neurons:
        plt.figure(figsize=(10, 4))
        for neuron in silent_neurons[:5]:  # Plot first 5 silent neurons if there are many
            plt.plot(spike_df[neuron], label=neuron)
        plt.xlabel("Time")
        plt.ylabel("Spike Count")
        plt.title("Time-Series for a Few Silent Neurons (Day 0)")
        plt.legend()
        plt.show()
