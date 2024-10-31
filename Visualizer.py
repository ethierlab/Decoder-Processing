import tkinter as tk
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Charger les données du fichier pickle
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Apply low-pass filter to the data
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Function to bin and smooth spike times
def bin_and_smooth_spike_times(spike_times, bin_size=50, sigma=None):
    # Convert bin size from milliseconds to seconds
    bin_size_sec = bin_size / 1000.0
    
    # Determine the maximum time in the dataset (in seconds)
    max_time = int(np.ceil(max(spike_times))) + 1  # Include the last spike

    # Calculate the number of bins based on the bin size in seconds
    num_bins = int(max_time // bin_size_sec) + 1
    
    # Print the number of bins being created
    print(f"Number of bins created: {num_bins} (bin size = {bin_size} ms)")
    
    # Bin the spike times
    binned_counts, bin_edges = np.histogram(spike_times, bins=num_bins, range=(0, max_time))
    
    binned_counts = binned_counts.astype(float)
    # Apply Gaussian smoothing if sigma is provided
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
    
    return binned_counts, bin_edges[:-1]  # Return smoothed counts and bin edges (exclude the last edge)

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

# Function to plot the heatmap with zoom by adjusting axis limits
def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins, fig, axs, zoom_start=None, zoom_end=None):
    axs[0].clear()

    # Plot the full heatmap
    im = axs[0].imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot', 
                       extent=[time_bins[0], time_bins[-1], len(subunit_names), 0])

    # Set labels and titles
    # axs[0].set_title('Heatmap')
    axs[0].set_xlabel('Temps (ms)')
    axs[0].set_ylabel('Subunits')
    axs[0].set_yticks(np.arange(len(subunit_names)))  # Set the tick positions
    axs[0].set_yticklabels(subunit_names)  # Set the labels for the ticks
    # Add the colorbar
    fig.colorbar(im, ax=axs[0], label='Densité de Spike')

    # Set x-axis limits for zooming if zoom_start and zoom_end are provided
    if zoom_start is not None and zoom_end is not None:
        axs[0].set_xlim(zoom_start*1000, zoom_end*1000)



# Function to plot the force signal
def plot_force(tdt_signals, axs, zoom_start=None, zoom_end=None):
    force_signal = tdt_signals['Levier']
    
    force_sampling_rate = 1017.3  # 1017.3 Hz for the force signal
    total_time_force = len(force_signal) * (1 / force_sampling_rate)
    time_force = np.arange(0, total_time_force, 1 / force_sampling_rate)
    force_signal = (force_signal - 294) * 1.95  # Convert to grams
    cutoff = 30  # Cutoff frequency for the low-pass filter
    force_signal = apply_lowpass_filter(force_signal, cutoff, force_sampling_rate)
    
    # Apply zoom if zoom_start and zoom_end are provided
    if zoom_start is not None and zoom_end is not None:
        zoom_indices = (time_force >= zoom_start) & (time_force <= zoom_end)
        if np.sum(zoom_indices) > 0:  # Ensure there is data to plot within the zoom range
            time_force = time_force[zoom_indices]
            force_signal = force_signal[zoom_indices]

    axs[1].clear()
    axs[1].plot(time_force, force_signal, label='Signal de Force')
    # axs[1].set_title('Signal de Force')
    axs[1].set_xlabel('Temps (s)')
    axs[1].set_ylabel('Force (g)')
    axs[1].grid(True)

# Function to plot kinematic positions (X and Y)
def plot_kinematics(kinematics_data, t_0_times, axs, plot_x=True, plot_y=True, selected_positions=None, zoom_start=None, zoom_end=None):
    if selected_positions is None:
        selected_positions = ['start', 'middle', 'tip', 'angle_left', 'angle_right']

    position_colors = {'start': 'blue', 'middle': 'green', 'tip': 'red', 'angle_left': 'purple', 'angle_right': 'orange'}
    time_step_kinematics = 1 / 200  # 200 Hz for kinematics data
    axs[2].clear()

    for key in selected_positions:
        color = position_colors.get(key, 'black')
        for trial_index, trial_data_x in enumerate(kinematics_data['x'][key]):
            trial_data_y = kinematics_data['y'][key][trial_index]
            if trial_index >= len(t_0_times):
                continue
            start_time = t_0_times[trial_index] - 1  # T_0 - 1 second
            time = np.arange(start_time, start_time + len(trial_data_x) * time_step_kinematics, time_step_kinematics)

            # Apply zoom if zoom_start and zoom_end are provided
            if zoom_start is not None and zoom_end is not None:
                zoom_indices = (time >= zoom_start) & (time <= zoom_end)
                zoom_indices = np.where(zoom_indices)[0]  # Get the integer indices where condition is True

                if len(zoom_indices) > 0:  # Ensure there is data to plot within the zoom range
                    time = time[zoom_indices]
                    trial_data_x = np.array(trial_data_x)[zoom_indices]
                    trial_data_y = np.array(trial_data_y)[zoom_indices]

            # Plot X and Y data based on user selection
            if plot_x:
                axs[2].plot(time, trial_data_x, label=f'X {key}', color=color, alpha=0.6, linestyle='--')
            if plot_y:
                axs[2].plot(time, trial_data_y, label=f'Y {key}', color=color, alpha=0.6)

    # axs[2].set_title('Positions Kinématiques X et Y')
    axs[2].set_xlabel('Temps (s)')
    axs[2].set_ylabel('Position')
    axs[2].grid(True)


    # Set x-axis limits for zooming if zoom_start and zoom_end are provided
    if zoom_start is not None and zoom_end is not None:
        axs[2].set_xlim(zoom_start, zoom_end)





# Function to handle subunit selection and data processing
def handle_selection():
    # Create a new figure and subplots to avoid colorbar duplication
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    # Ensure axs is a list, even if it's a single axis
    if not isinstance(axs, np.ndarray):
        axs = [axs]  # Wrap axs in a list if it's a single axis
    
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
        plot_x = x_var.get() == 1  # Check if the X-axis is selected
        plot_y = y_var.get() == 1  # Check if the Y-axis is selected
        selected_positions = [pos for pos, var in position_vars.items() if var.get() == 1]
        # Get z-score value from GUI
        apply_z_score = z_score_var.get()

    except ValueError:
        print("Invalid input for bin size, sigma, or zoom values. Please enter valid numbers.")
        return

    # Load the experiment data
    experiment_data = load_pickle('experiment_data.pkl')

    # Initialize lists for smoothed data and subunit names
    smoothed_spikes_matrix = []
    subunit_names = []

    # Variable to store maximum time length for aligning the heatmap
    max_time_length = 0
    subunit_2_missing = False

    # Process each channel and its IDs
    for channel, ids in experiment_data['data'].items():
        channel_number = int(channel.split('Channel')[1])

        if channel_number not in selected_channels:
            continue  # Skip if the channel is not selected

        if "Subunit 2" in selected_subunits and not any("#2" in id_ for id_ in ids.keys()):
            print(f"Subunit 2 not available for Channel {channel_number}")
            subunit_2_missing = True

        for id_, values in ids.items():
            if ("Subunit 1" in selected_subunits and "#1" in id_) or \
               ("Subunit 2" in selected_subunits and "#2" in id_):

                spike_times = values['spike_times']
                smoothed_spikes, time_bins = bin_and_smooth_spike_times(spike_times, bin_size=bin_size, sigma=sigma)
                
                if apply_z_score:
                    smoothed_spikes = z_score_normalize(smoothed_spikes)

                if len(smoothed_spikes) > max_time_length:
                    max_time_length = len(smoothed_spikes)

                smoothed_spikes_matrix.append(smoothed_spikes)
                subunit_name = f"{channel.split('Channel')[1]}#{id_.split('#')[1]}"
                subunit_names.append(subunit_name)
                
    print(subunit_names)
    # Align matrices by padding with zeros to match the maximum time length
    for i in range(len(smoothed_spikes_matrix)):
        if len(smoothed_spikes_matrix[i]) < max_time_length:
            smoothed_spikes_matrix[i] = np.pad(smoothed_spikes_matrix[i],
                                               (0, max_time_length - len(smoothed_spikes_matrix[i])),
                                               'constant')

    # Convert to numpy matrix and generate time bins
    smoothed_spikes_matrix = np.array(smoothed_spikes_matrix)
    time_bins_aligned = np.arange(max_time_length) * bin_size

    # Check for time range validity
    if zoom_start is not None and zoom_end is not None:
        if zoom_start < 0 or zoom_end > time_bins_aligned[-1]:
            print(f"Time index out of range. Maximum time is {time_bins_aligned[-1]} ms.")
            return

    # Display the heatmap
    plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins_aligned, fig, axs, zoom_start, zoom_end)

    if subunit_2_missing:
        print("Note: Subunit 2 is not available in one or more selected channels.")

    # Load additional data files
    tdt_signals = load_pickle('tdt_signals.pkl')
    kinematics_data = load_pickle('kinematics.pkl')

    # Get T_0 times for TDT signals
    t_0_times = tdt_signals['Event Time']

    # Plot the force signal with zoom support
    plot_force(tdt_signals, axs, zoom_start=zoom_start, zoom_end=zoom_end)

    # Plot the kinematic positions with zoom support
    plot_kinematics(kinematics_data, t_0_times, axs, plot_x=plot_x, plot_y=plot_y, selected_positions=selected_positions, zoom_start=zoom_start, zoom_end=zoom_end)

    # Show the figure
    plt.show()


# Création de la fenêtre principale avec Tkinter
root = tk.Tk()
root.title("Outil de Sélection Interactive")

# Variables to store checkbox values
subunit_1_var = tk.IntVar(value=1)  # 1 means selected by default
subunit_2_var = tk.IntVar(value=1)
x_var = tk.IntVar(value=1)  # X-axis selected by default
y_var = tk.IntVar(value=1)  # Y-axis selected by default
z_score_var = tk.IntVar(value=0)  # Z-score checkbox is deselected by default

# Frame for Channels section
frame_channels = tk.Frame(root)
frame_channels.pack(padx=10, pady=10, side=tk.LEFT)

channel_label = tk.Label(frame_channels, text="Entrer les Canaux (ex: 1-3, 5, 7):")
channel_label.pack()

channel_entry = tk.Entry(frame_channels)
channel_entry.insert(0, '1-32')  # Valeur par défaut
channel_entry.pack()

# Subunit Selection
subunit_label = tk.Label(frame_channels, text="Select Subunits:")
subunit_label.pack()

subunit_1_checkbox = tk.Checkbutton(frame_channels, text="Subunit 1", variable=subunit_1_var)
subunit_2_checkbox = tk.Checkbutton(frame_channels, text="Subunit 2", variable=subunit_2_var)
subunit_1_checkbox.pack()
subunit_2_checkbox.pack()

# Zoom Section
zoom_start_label = tk.Label(frame_channels, text="Début du Zoom (s):")
zoom_start_label.pack()

zoom_start_entry = tk.Entry(frame_channels)
zoom_start_entry.pack()

zoom_end_label = tk.Label(frame_channels, text="Fin du Zoom (s):")
zoom_end_label.pack()

zoom_end_entry = tk.Entry(frame_channels)
zoom_end_entry.pack()

# Frame for Processing section
frame_processing = tk.Frame(root)
frame_processing.pack(padx=10, pady=10, side=tk.LEFT)

# Bin size
bin_size_label = tk.Label(frame_processing, text="Entrer la Taille des Bins (ms):")
bin_size_label.pack()

bin_size_entry = tk.Entry(frame_processing)
bin_size_entry.insert(0, "50")  # Valeur par défaut
bin_size_entry.pack()

# Sigma input
sigma_label = tk.Label(frame_processing, text="Entrer le Sigma (pour le filtre gaussien):")
sigma_label.pack()

sigma_entry = tk.Entry(frame_processing)
sigma_entry.insert(0, "1.0")  # Valeur par défaut
sigma_entry.pack()

# Z-score checkbox
z_score_checkbox = tk.Checkbutton(frame_processing, text="Appliquer la Normalisation Z-Score", variable=z_score_var)
z_score_checkbox.pack()

# Apply button (right below the Z-score checkbox)
apply_button = tk.Button(frame_processing, text="Obtenir la Sélection et Traiter", command=handle_selection)
apply_button.pack(pady=10)

# Frame for Kinematics section
frame_kinematics = tk.Frame(root)
frame_kinematics.pack(padx=10, pady=10, side=tk.LEFT)

# Axes Selection (X, Y)
x_checkbox = tk.Checkbutton(frame_kinematics, text="Afficher X", variable=x_var)
y_checkbox = tk.Checkbutton(frame_kinematics, text="Afficher Y", variable=y_var)
x_checkbox.pack()
y_checkbox.pack()

# Position checkboxes
position_vars = {}
position_labels = ['start', 'middle', 'tip', 'angle_left', 'angle_right']

for pos in position_labels:
    position_vars[pos] = tk.IntVar(value=1)  # Default selected
    position_checkbox = tk.Checkbutton(frame_kinematics, text=pos, variable=position_vars[pos])
    position_checkbox.pack()

# Start the GUI main loop
root.mainloop()
