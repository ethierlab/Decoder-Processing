import tkinter as tk
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
# Global variables to store processed data and figures
smoothed_spikes_matrix = None
raw_binned_counts_matrix = None
subunit_names = None
time_bins_aligned = None
figures = []
save_button = None


# Function to load TDT data
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def bin_and_smooth_spike_times(spike_times, bin_size=5, sigma=None):
    spike_times = np.array(spike_times)
    # Convert spike_times to milliseconds
    spike_times_ms = spike_times * 1000

    # Calculate max_time in ms
    max_time = int(np.ceil(max(spike_times_ms)))  # ms

    # Create bin edges
    bin_edges = np.arange(0, max_time + bin_size, bin_size)

    # Perform histogram binning
    binned_counts, _ = np.histogram(spike_times_ms, bins=bin_edges)

    # Convert binned counts to float for smoothing
    binned_counts = binned_counts.astype(float)

    # Apply Gaussian smoothing if sigma is provided
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)

    return binned_counts, bin_edges[:-1]

# Function to generate a heatmap with optional zooming
def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins, zoom_start=None, zoom_end=None):
    global figures
    global save_button
    fig = plt.figure(figsize=(12, 8))
    fig_label = "Heatmap"

    # Determine the range to display based on zoom
    if zoom_start is not None and zoom_end is not None:
        zoom_indices = (time_bins >= zoom_start) & (time_bins <= zoom_end)
        time_bins = time_bins[zoom_indices]
        smoothed_spikes_matrix = smoothed_spikes_matrix[:, zoom_indices]

    plt.imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot', 
               extent=[time_bins[0], time_bins[-1], len(subunit_names), 0])
    plt.colorbar(label='Spike Density')
    plt.yticks(ticks=np.arange(len(subunit_names)), labels=subunit_names)
    plt.xlabel('Time (ms)')
    plt.ylabel('Subunits')
    plt.title('Heatmap of Smoothed Spike Times Across Subunits')
    plt.show()
    figures.append((fig_label, fig))
    figure_listbox.insert(tk.END, fig_label)
    save_button.config(state="normal")

def plot_raw_signal_below_heatmap(smoothed_spikes_matrix, raw_binned_counts_matrix, subunit_names, time_bins,
                                  selected_channel, selected_subunit):
    global figures
    global save_button

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig_label = f"Channel_{selected_channel}_Subunit_{selected_subunit.replace(' ', '_')}"
    # Plot the main heatmap as usual
    ax1.imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot',
               extent=[time_bins[0], time_bins[-1], len(subunit_names), 0])
    ax1.set_yticks(ticks=np.arange(len(subunit_names)), labels=subunit_names)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Subunits')
    ax1.set_title('Heatmap of Smoothed Spike Times Across Subunits')

    # Extract subunit number from selected_subunit
    subunit_number = selected_subunit.split(' ')[1]  # '1' or '2'
    subunit_name_to_find = f"{selected_channel}#{subunit_number}"

    if subunit_name_to_find in subunit_names:
        idx = subunit_names.index(subunit_name_to_find)
        raw_counts = raw_binned_counts_matrix[idx]
        smoothed_counts = smoothed_spikes_matrix[idx]

        # Plot the raw and smoothed counts
        ax2.plot(time_bins, raw_counts, label=f'Raw Channel {selected_channel} Subunit {subunit_number}',
                 linestyle='--')
        ax2.plot(time_bins, smoothed_counts, label=f'Smoothed Channel {selected_channel} Subunit {subunit_number}')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Spike Count')
        ax2.set_title(f'Raw and Smoothed Signals for Channel {selected_channel} Subunit {subunit_number}')
        ax2.legend()
    else:
        print(f"Subunit {subunit_number} of Channel {selected_channel} not found in data.")

    plt.tight_layout()
    plt.show()
    figures.append((fig_label, fig))
    figure_listbox.insert(tk.END, fig_label)
    save_button.config(state="normal")

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

def save_plots():
    global figures

    if not figures:
        print("No plots to save.")
        return

    # Get selected indices from the listbox
    selected_indices = figure_listbox.curselection()
    if not selected_indices:
        print("No figures selected to save.")
        return

    # Use current working directory
    directory = os.getcwd()
    print(f"Saving selected plots to current directory: {directory}")

    # Save each selected figure
    for idx in selected_indices:
        fig_label, fig = figures[idx]
        filename = f"{fig_label}.png"
        filepath = os.path.join(directory, filename)
        try:
            fig.savefig(filepath)
            print(f"Saved {filepath}")
        except Exception as e:
            print(f"Failed to save {filepath}: {e}")

    print("Selected plots have been saved.")
# Function to compute PETHs
def compute_peths(smoothed_spikes_matrix, time_bins_aligned, t_0_times, bin_size, pre_event_time, post_event_time):
    num_subunits = smoothed_spikes_matrix.shape[0]
    time_bins_peth = np.arange(-pre_event_time, post_event_time + bin_size, bin_size)
    peths_mean = []
    peths_std = []
    for i in range(num_subunits):
        subunit_spikes = smoothed_spikes_matrix[i]
        trials = []
        for t0 in t_0_times:
            window_start = t0 - pre_event_time
            window_end = t0 + post_event_time
            idx_start = np.searchsorted(time_bins_aligned, window_start)
            idx_end = np.searchsorted(time_bins_aligned, window_end)
            # Handle boundary conditions
            if idx_start < 0 or idx_end > len(subunit_spikes):
                continue
            trial_data = subunit_spikes[idx_start:idx_end]
            expected_length = len(time_bins_peth)
            # Pad or trim to match expected length
            if len(trial_data) < expected_length:
                trial_data = np.pad(trial_data, (0, expected_length - len(trial_data)), 'constant')
            elif len(trial_data) > expected_length:
                trial_data = trial_data[:expected_length]
            trials.append(trial_data)
        if trials:
            trials = np.array(trials)
            peth_mean = np.mean(trials, axis=0)
            peth_std = np.std(trials, axis=0)
        else:
            peth_mean = np.zeros(len(time_bins_peth))
            peth_std = np.zeros(len(time_bins_peth))
        peths_mean.append(peth_mean)
        peths_std.append(peth_std)
    peths_mean = np.array(peths_mean)
    peths_std = np.array(peths_std)
    return peths_mean, peths_std, time_bins_peth

# Function to plot PETHs
def plot_peths(peths_mean, peths_std, time_bins_peth, subunit_names):
    global figures
    global save_button
    
    num_subunits = len(subunit_names)
    for i in range(num_subunits):
        fig = plt.figure()
        fig_label = f"PETH_Subunit_{subunit_label.replace('#', '_')}"
        mean = peths_mean[i]
        std = peths_std[i]
        plt.plot(time_bins_peth, mean, label='Mean')
        plt.fill_between(time_bins_peth, mean - std, mean + std, alpha=0.3, label='Std Dev')
        plt.title(f'PETH for Subunit {subunit_names[i]}')
        plt.xlabel('Time relative to event (ms)')
        plt.ylabel('Spike Density')
        plt.axvline(0, color='red', linestyle='--')  # Add vertical line at time zero
        plt.legend()
        plt.show()
        figures.append(fig)
        figure_listbox.insert(tk.END, fig_label)
        save_button.config(state="normal")

# Function to compute and plot PETHs for the selected subunit
def compute_and_plot_peths():
    global smoothed_spikes_matrix
    global time_bins_aligned
    global subunit_names

    # Get TDT file path
    tdt_file = tdt_file_entry.get()

    # Load tdt_signals and t_0_times
    try:
        tdt_signals = load_data(tdt_file)
        t_0_times = tdt_signals['Event Time'] * 1000  # Convert to ms if necessary
    except Exception as e:
        print(f"Error loading TDT signals: {e}")
        return

    # Get pre-event and post-event times
    try:
        pre_event_time = float(pre_event_entry.get())
        post_event_time = float(post_event_entry.get())
    except ValueError:
        print("Invalid input for pre-event or post-event time.")
        return

    # Get bin size
    try:
        bin_size = int(bin_size_entry.get())
    except ValueError:
        print("Invalid input for bin size.")
        return

    # Get selected subunit for PETH plotting
    selected_subunit = peth_subunit_var.get()
    try:
        subunit_idx = subunit_names.index(selected_subunit)
    except ValueError:
        print(f"Selected subunit {selected_subunit} not found.")
        return

    # Compute PETH for the selected subunit
    peths_mean, peths_std, time_bins_peth = compute_peths(
        smoothed_spikes_matrix[subunit_idx:subunit_idx+1],  # Pass only the selected subunit
        time_bins_aligned,
        t_0_times,
        bin_size,
        pre_event_time,
        post_event_time
    )

    # Plot PETH for the selected subunit
    plot_peths(peths_mean, peths_std, time_bins_peth, [selected_subunit])  # Pass single subunit name in a list

# Function to handle the final selection and data processing
def handle_selection():
    global smoothed_spikes_matrix
    global raw_binned_counts_matrix
    global subunit_names
    global time_bins_aligned
    
    
    # Clear previous figures and listbox entries
    figures.clear()
    figure_listbox.delete(0, tk.END)
    save_button.config(state="disabled")

    selected_subunits = get_subunit_selection()
    selected_channels = get_channel_selection()
    print(f"Selected Subunits: {selected_subunits}")
    print(f"Selected Channels: {selected_channels}")

    # Get sigma, bin_size, z-score, zoom_start, and zoom_end values from GUI
    try:
        bin_size = int(bin_size_entry.get())  # Bin size in ms
        smoothing_length = float(smoothing_length_entry.get())  # Smoothing length in ms
        zoom_start = float(zoom_start_entry.get()) if zoom_start_entry.get() else None
        zoom_end = float(zoom_end_entry.get()) if zoom_end_entry.get() else None
        # Get z-score value from GUI
        apply_z_score = z_score_var.get()
        sigma = smoothing_length / bin_size
    except ValueError:
        print("Invalid input for bin size, sigma, or zoom values. Please enter valid numbers.")
        return

    # Load the experiment data
    experiment_data = load_data('experiment_data.pkl')

    # Initialize the lists to store the data for each subunit
    smoothed_spikes_matrix = []
    raw_binned_counts_matrix = []
    subunit_names = []

    # Determine the maximum time length for aligning the heatmap
    max_time_length = 0

    # Process each channel and ID based on user selection
    for channel, ids in experiment_data['data'].items():
        channel_number = int(channel.split('Channel')[1])

        if channel_number not in selected_channels:
            continue  # Skip if channel is not selected

        for id_, values in ids.items():
            subunit_number = id_.split('#')[1]
            if f"Subunit {subunit_number}" not in selected_subunits:
                continue  # Skip if subunit is not selected

            spike_times = values['spike_times']

            # Binning of spike_times for the subunit without smoothing to get raw counts
            binned_counts, time_bins = bin_and_smooth_spike_times(spike_times, bin_size=bin_size, sigma=None)  # sigma=None for raw counts

            # Convert binned counts to float
            binned_counts = binned_counts.astype(float)

            # Store the raw binned counts
            raw_binned_counts_matrix.append(binned_counts)

            # Apply Gaussian smoothing to the binned counts
            smoothed_spikes = gaussian_filter1d(binned_counts, sigma=sigma)

            # Apply Z-Score normalization if checked
            if apply_z_score:
                smoothed_spikes = z_score_normalize(smoothed_spikes)

            # Update the maximum time length
            if len(smoothed_spikes) > max_time_length:
                max_time_length = len(smoothed_spikes)

            # Store the smoothed data
            smoothed_spikes_matrix.append(smoothed_spikes)

            # Store the short name of the subunit, e.g., '17#1'
            subunit_name = f"{channel_number}#{subunit_number}"
            subunit_names.append(subunit_name)

    # Align all matrices to the same length by padding with zeros
    for i in range(len(smoothed_spikes_matrix)):
        if len(smoothed_spikes_matrix[i]) < max_time_length:
            smoothed_spikes_matrix[i] = np.pad(smoothed_spikes_matrix[i],
                                               (0, max_time_length - len(smoothed_spikes_matrix[i])),
                                               'constant')
        if len(raw_binned_counts_matrix[i]) < max_time_length:
            raw_binned_counts_matrix[i] = np.pad(raw_binned_counts_matrix[i],
                                                 (0, max_time_length - len(raw_binned_counts_matrix[i])),
                                                 'constant')

    # Convert to numpy matrices
    smoothed_spikes_matrix = np.array(smoothed_spikes_matrix)
    raw_binned_counts_matrix = np.array(raw_binned_counts_matrix)

    # Generate aligned time bins
    time_bins_aligned = np.arange(max_time_length) * bin_size

    # Check for time range validity and display heatmap with optional zoom
    if zoom_start is not None and zoom_end is not None:
        if zoom_start < 0 or zoom_end > time_bins_aligned[-1]:
            print(f"Time index out of range. Maximum time is {time_bins_aligned[-1]} ms.")
            return

    # Display the heatmap with optional zoom
    plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins_aligned, zoom_start, zoom_end)

    # Populate the plotting selection dropdowns
    available_channels = sorted(set([int(name.split('#')[0]) for name in subunit_names]))
    available_subunits = sorted(set(['Subunit ' + name.split('#')[1] for name in subunit_names]))

    # Update the channel dropdown menu
    plot_channel_var.set(available_channels[0])  # Set default value
    plot_channel_menu['menu'].delete(0, 'end')
    for channel in available_channels:
        plot_channel_menu['menu'].add_command(label=channel, command=tk._setit(plot_channel_var, channel))
    plot_channel_menu.config(state="normal")  # Enable the dropdown

    # Update the subunit dropdown menu
    plot_subunit_var.set(available_subunits[0])  # Set default value
    plot_subunit_menu['menu'].delete(0, 'end')
    for subunit in available_subunits:
        plot_subunit_menu['menu'].add_command(label=subunit, command=tk._setit(plot_subunit_var, subunit))
    plot_subunit_menu.config(state="normal")  # Enable the dropdown

    # Enable the plot button
    plot_button.config(state="normal")

    # Enable the PETH button after data is processed
    peth_button.config(state="normal")

    # Update the PETH subunit dropdown menu
    peth_subunit_var.set(subunit_names[0])  # Set default value
    peth_subunit_menu['menu'].delete(0, 'end')
    for subunit in subunit_names:
        peth_subunit_menu['menu'].add_command(label=subunit, command=tk._setit(peth_subunit_var, subunit))
    peth_subunit_menu.config(state="normal")  # Enable the dropdown

    # Inform the user to select the subunit to plot PETH
    print("Select the subunit to plot PETH from the dropdown menu.")

# Function to plot the selected channel and subunit
def plot_selected_channel():
    selected_channel = int(plot_channel_var.get())
    selected_subunit = plot_subunit_var.get()  # e.g., 'Subunit 1'

    # Proceed to plot the selected channel and subunit
    plot_raw_signal_below_heatmap(
        smoothed_spikes_matrix,
        raw_binned_counts_matrix,
        subunit_names,
        time_bins_aligned,
        selected_channel,
        selected_subunit
    )

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

# Create the main window
root = tk.Tk()
root.title("Interactive Selection Tool")

# Now that the root window is created, we can initialize Tkinter variables
# Variables to store checkbox values
subunit_1_var = tk.IntVar(value=1)  # 1 means selected by default
subunit_2_var = tk.IntVar(value=1)

# Variables to store the user's choices for plotting
plot_channel_var = tk.StringVar()
plot_subunit_var = tk.StringVar()

# Variable for PETH subunit selection
peth_subunit_var = tk.StringVar()

# Section for subunit selection
subunit_label = tk.Label(root, text="Select Subunits:")
subunit_label.pack()

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
bin_size_entry.insert(0, "5")  # Default value
bin_size_entry.pack()

# Section for smoothing length input
smoothing_length_label = tk.Label(root, text="Smoothing Length (ms):")
smoothing_length_label.pack()

smoothing_length_entry = tk.Entry(root)
smoothing_length_entry.insert(0, "50")  # Default smoothing length in ms
smoothing_length_entry.pack()

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
selection_button = tk.Button(root, text="Process Data", command=handle_selection)
selection_button.pack()

# Placeholder dropdowns for selecting the channel and subunit to plot
plot_channel_label = tk.Label(root, text="Select Channel to Plot:")
plot_channel_label.pack()
plot_channel_menu = tk.OptionMenu(root, plot_channel_var, [])
plot_channel_menu.config(state="disabled")  # Initially disabled
plot_channel_menu.pack()

plot_subunit_label = tk.Label(root, text="Select Subunit to Plot:")
plot_subunit_label.pack()
plot_subunit_menu = tk.OptionMenu(root, plot_subunit_var, [])
plot_subunit_menu.config(state="disabled")  # Initially disabled
plot_subunit_menu.pack()

# Button to plot the selected channel and subunit
plot_button = tk.Button(root, text="Plot Selected Channel and Subunit", command=plot_selected_channel)
plot_button.config(state="disabled")
plot_button.pack()

# Placeholder dropdown for selecting the subunit to plot PETH
peth_subunit_label = tk.Label(root, text="Select Subunit for PETH Plot:")
peth_subunit_label.pack()
peth_subunit_menu = tk.OptionMenu(root, peth_subunit_var, [])
peth_subunit_menu.config(state="disabled")  # Initially disabled
peth_subunit_menu.pack()

# Section for TDT file input
tdt_file_label = tk.Label(root, text="Enter TDT File Path:")
tdt_file_label.pack()
tdt_file_entry = tk.Entry(root)
tdt_file_entry.insert(0, "tdt_signals.pkl")  # Default file name
tdt_file_entry.pack()

# Section for pre-event time input
pre_event_label = tk.Label(root, text="Pre-Event Time (ms):")
pre_event_label.pack()
pre_event_entry = tk.Entry(root)
pre_event_entry.insert(0, "1000")  # Default pre-event time in ms
pre_event_entry.pack()

# Section for post-event time input
post_event_label = tk.Label(root, text="Post-Event Time (ms):")
post_event_label.pack()
post_event_entry = tk.Entry(root)
post_event_entry.insert(0, "1000")  # Default post-event time in ms
post_event_entry.pack()

# Button to compute and plot PETHs
peth_button = tk.Button(root, text="Compute and Plot PETHs", command=compute_and_plot_peths)
peth_button.config(state="disabled")  # Initially disabled
peth_button.pack()

# Listbox to display generated figure labels
figure_listbox_label = tk.Label(root, text="Generated Figures:")
figure_listbox_label.pack()

figure_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=10)
figure_listbox.pack()

# Button to save selected plots
save_button = tk.Button(root, text="Save Selected Plots", command=save_plots)
save_button.config(state="disabled")  # Initially disabled
save_button.pack()

# Run the main loop
root.mainloop()
