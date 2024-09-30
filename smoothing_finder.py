import tkinter as tk
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Function to load the pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def bin_and_smooth_spike_times(spike_times, bin_size=50, sigma=None):
    max_time = int(np.ceil(max(spike_times))) + 1  # To include the last spike
    num_bins = max_time // bin_size + 1
    binned_counts, bin_edges = np.histogram(spike_times, bins=num_bins, range=(0, max_time))
    
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
    
    return binned_counts, bin_edges[:-1]  # Return the binned (and optionally smoothed) values and the bin edges

# Function to generate a heatmap with optional zooming
def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins, zoom_start=None, zoom_end=None):
    plt.figure(figsize=(12, 8))
    
    # Determine the range to display based on zoom
    if zoom_start is not None and zoom_end is not None:
        zoom_indices = (time_bins >= zoom_start) & (time_bins <= zoom_end)
        time_bins = time_bins[zoom_indices]
        smoothed_spikes_matrix = smoothed_spikes_matrix[:, zoom_indices]
    
    plt.imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot', extent=[time_bins[0], time_bins[-1], len(subunit_names), 0])
    plt.colorbar(label='Spike Density')
    plt.yticks(ticks=np.arange(len(subunit_names)), labels=subunit_names)
    plt.xlabel('Time (ms)')
    plt.ylabel('Subunits')
    plt.title('Heatmap of Smoothed Spike Times Across Subunits')
    plt.show()

# Function to handle subunit selection
def get_subunit_selection():
    subunit_1_selected = subunit_1_var.get()
    subunit_2_selected = subunit_2_var.get()
    selected_subunits = []
    if subunit_1_selected:
        selected_subunits.append("Subunit 1")
    if subunit_2_selected:
        selected_subunits.append("Subunit 2")
    return selected_subunits

# Function to z-score normalize the data
def z_score_normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    # Check if std_dev is not zero to avoid division by zero
    if std_dev == 0:
        return data  # No need to z-score if std_dev is zero
    return (data - mean) / std_dev


# Function to parse the input and get the selected channels
def get_channel_selection():
    input_text = channel_entry.get()
    channels = parse_channels(input_text)
    return channels

# Function to parse the input channels based on user format
def parse_channels(input_text):
    channels = set()
    input_text = input_text.replace(' ', '')  # Remove spaces
    if input_text:
        for part in input_text.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                channels.update(range(start, end + 1))
            else:
                channels.add(int(part))
    return sorted(channels)

# Function to handle the final selection and data processing
def handle_selection():
    selected_subunits = get_subunit_selection()
    selected_channels = get_channel_selection()
    print(f"Selected Subunits: {selected_subunits}")
    print(f"Selected Channels: {selected_channels}")

    # Get sigma, bin_size, z-score, zoom_start, and zoom_end values from GUI
    try:
        bin_size = int(bin_size_entry.get())
        sigma = float(sigma_entry.get())
        zoom_start = float(zoom_start_entry.get()) if zoom_start_entry.get() else None
        zoom_end = float(zoom_end_entry.get()) if zoom_end_entry.get() else None
        # Get z-score value from GUI
        apply_z_score = z_score_var.get()

    except ValueError:
        print("Invalid input for bin size, sigma, or zoom values. Please enter valid numbers.")
        return

    # Load the experiment data
    experiment_data = load_pickle('experiment_data.pkl')

    # Initialize the list to store the smoothed data for each subunit
    smoothed_spikes_matrix = []
    subunit_names = []

    # Determine the maximum time length for aligning the heatmap
    max_time_length = 0

    # Flag to check if Subunit 2 is missing
    subunit_2_missing = False

    # Process each channel and ID based on user selection
    for channel, ids in experiment_data['data'].items():
        channel_number = int(channel.split('Channel')[1])
        
        if channel_number not in selected_channels:
            continue  # Skip if channel is not selected

        # Check if Subunit 2 is missing for this channel
        if "Subunit 2" in selected_subunits and not any("#2" in id_ for id_ in ids.keys()):
            print(f"Subunit 2 not available for Channel {channel_number}")
            subunit_2_missing = True

        for id_, values in ids.items():
            if ("Subunit 1" in selected_subunits and "#1" in id_) or \
               ("Subunit 2" in selected_subunits and "#2" in id_):

                spike_times = values['spike_times']

                # Binning of combined spike_times for the subunit
                # binned_counts, time_bins = bin_spike_times(spike_times, bin_size=bin_size)

                # # Smoothing the binned data
                # smoothed_spikes = smooth_binned_spike_times(binned_counts, sigma=sigma)
                # Binning and Smoothing of combined spike_times for the subunit
                smoothed_spikes, time_bins = bin_and_smooth_spike_times(spike_times, bin_size=bin_size, sigma=sigma)
                # Apply Z-Score normalization if checked
                if apply_z_score:
                    smoothed_spikes = z_score_normalize(smoothed_spikes)

                # Update the maximum time length
                if len(smoothed_spikes) > max_time_length:
                    max_time_length = len(smoothed_spikes)

                # Store the smoothed data
                smoothed_spikes_matrix.append(smoothed_spikes)

                # Store the short name of the subunit, e.g., 'ch17#1'
                subunit_name = f"{channel.split('Channel')[1]}#{id_.split('#')[1]}"
                subunit_names.append(subunit_name)

    # Align all matrices to the same length by padding with zeros
    for i in range(len(smoothed_spikes_matrix)):
        if len(smoothed_spikes_matrix[i]) < max_time_length:
            smoothed_spikes_matrix[i] = np.pad(smoothed_spikes_matrix[i],
                                               (0, max_time_length - len(smoothed_spikes_matrix[i])),
                                               'constant')

    # Convert to numpy matrix
    smoothed_spikes_matrix = np.array(smoothed_spikes_matrix)

    # Generate aligned time bins
    time_bins_aligned = np.arange(max_time_length) * bin_size
    # Check for time range validity and display heatmap with optional zoom
    if zoom_start is not None and zoom_end is not None:
        if zoom_start < 0 or zoom_end > time_bins_aligned[-1]:
            print(f"Time index out of range. Maximum time is {time_bins_aligned[-1]} ms.")
            return

    # Display the heatmap with optional zoom
    plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins_aligned, zoom_start, zoom_end)

    # If Subunit 2 was selected but not available in any channel
    if subunit_2_missing:
        print("Note: Subunit 2 is not available in one or more selected channels.")

# Create the main window
root = tk.Tk()
root.title("Interactive Selection Tool")

# Section for subunit selection
subunit_label = tk.Label(root, text="Select Subunits:")
subunit_label.pack()

# Variables to store checkbox values
subunit_1_var = tk.IntVar(value=1)  # 1 means selected by default
subunit_2_var = tk.IntVar(value=1)

# Checkboxes for subunit selection
subunit_1_checkbox = tk.Checkbutton(root, text="Subunit 1", variable=subunit_1_var)
subunit_2_checkbox = tk.Checkbutton(root, text="Subunit 2", variable=subunit_2_var)

# Display checkboxes
subunit_1_checkbox.pack()
subunit_2_checkbox.pack()

# Section for channel selection
channel_label = tk.Label(root, text="Enter Channels (e.g., 1-3, 5, 7):")
channel_label.pack()


channel_entry = tk.Entry(root)
channel_entry.insert(0, '1-32')  
channel_entry.pack()

# Section for z-score normalization option
z_score_var = tk.IntVar(value=0)  # 0 means not selected by default
z_score_checkbox = tk.Checkbutton(root, text="Apply Z-Score Normalization", variable=z_score_var)
z_score_checkbox.pack()


# Section for bin size input
bin_size_label = tk.Label(root, text="Enter Bin Size (ms):")
bin_size_label.pack()

bin_size_entry = tk.Entry(root)
bin_size_entry.insert(0, "50")  # Default value
bin_size_entry.pack()

# Section for sigma input
sigma_label = tk.Label(root, text="Enter Sigma (for Gaussian filter):")
sigma_label.pack()

sigma_entry = tk.Entry(root)
sigma_entry.insert(0, "1.0")  # Default value
sigma_entry.pack()

# Section for zoom start input
zoom_start_label = tk.Label(root, text="Zoom Start Time (ms):")
zoom_start_label.pack()

zoom_start_entry = tk.Entry(root)
zoom_start_entry.pack()

# Section for zoom end input
zoom_end_label = tk.Label(root, text="Zoom End Time (ms):")
zoom_end_label.pack()

zoom_end_entry = tk.Entry(root)
zoom_end_entry.pack()

# Button to handle the selection and display results
selection_button = tk.Button(root, text="Get Selection and Process", command=handle_selection)
selection_button.pack()

# Run the main loop
root.mainloop()
