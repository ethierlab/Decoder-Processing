import tkinter as tk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter1d
import os

# Unified function to load pickle files
def load_pickle_file(file_path):
    """Load a pickle file and return its contents."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None

# Load the necessary data files
loaded_data = {
    'kinematics': load_pickle_file('kinematics.pkl'),
    'tdt_signals': load_pickle_file('tdt_signals.pkl'),
    'experiment_data': load_pickle_file('experiment_data.pkl')  # Include the heatmap data
}

# Verify that all required data has been loaded
if not all(loaded_data.values()):
    print("Some required data files could not be loaded. Please check the file paths.")
else:
    print("All required data files have been successfully loaded.")

def get_trial_data_for_plot(kinematics_data, trial_number, time_step=0.005):
    """
    Extract kinematics data and corresponding time array for a specific trial.

    Parameters:
    - kinematics_data: Dictionary containing x and y coordinates for different parts.
    - trial_number: Index of the trial to extract.
    - time_step: Time interval between data points (default: 0.005 seconds).

    Returns:
    - extracted_data: Dictionary containing x and y coordinates for the selected trial.
    - time_array: Array of time points aligned with T_0.
    """
    extracted_data = {'x': {}, 'y': {}}
    trial_length = len(kinematics_data['x']['start'][trial_number])
    time_array = np.linspace(-1.0, 2.0, trial_length)  # Create the time array once here

    for key in kinematics_data['x'].keys():
        # Extract x and y data for the specified trial number
        extracted_data['x'][key] = kinematics_data['x'][key][trial_number]
        extracted_data['y'][key] = kinematics_data['y'][key][trial_number]

    return extracted_data, time_array

def generate_heatmap_data_and_subunit_names(selected_channels, selected_subunits, bin_size, sigma, apply_z_score):
    """Generate the smoothed spikes matrix, time bins, and subunit names."""
    smoothed_spikes_matrix = []
    subunit_names = []

    # Process each channel and ID based on user selection
    for channel, ids in loaded_data['experiment_data']['data'].items():
        channel_number = int(channel.split('Channel')[1])

        if channel_number not in selected_channels:
            continue  # Skip if channel is not selected

        for id_, values in ids.items():
            if ("Subunit 1" in selected_subunits and "#1" in id_) or \
               ("Subunit 2" in selected_subunits and "#2" in id_):

                spike_times = values['spike_times']

                # Binning and Smoothing of combined spike_times for the subunit
                smoothed_spikes, time_bins = bin_and_smooth_spike_times(spike_times, bin_size=bin_size, sigma=sigma)

                # Apply Z-Score normalization if checked
                if apply_z_score:
                    smoothed_spikes = z_score_normalize(smoothed_spikes)

                # Store the smoothed data
                smoothed_spikes_matrix.append(smoothed_spikes)

                # Store the short name of the subunit, e.g., 'ch17#1'
                subunit_name = f"{channel.split('Channel')[1]}#{id_.split('#')[1]}"
                subunit_names.append(subunit_name)

    # Align all matrices to the same length by padding with zeros
    max_time_length = max(len(s) for s in smoothed_spikes_matrix)
    for i in range(len(smoothed_spikes_matrix)):
        if len(smoothed_spikes_matrix[i]) < max_time_length:
            smoothed_spikes_matrix[i] = np.pad(smoothed_spikes_matrix[i],
                                               (0, max_time_length - len(smoothed_spikes_matrix[i])),
                                               'constant')

    # Convert to numpy matrix
    smoothed_spikes_matrix = np.array(smoothed_spikes_matrix)

    # Generate aligned time bins
    time_bins_aligned = np.arange(max_time_length) * bin_size
    return smoothed_spikes_matrix, time_bins_aligned, subunit_names

def bin_and_smooth_spike_times(spike_times, bin_size=50, sigma=None):
    max_time = int(np.ceil(max(spike_times))) + 1  # To include the last spike
    num_bins = max_time // bin_size + 1
    binned_counts, bin_edges = np.histogram(spike_times, bins=num_bins, range=(0, max_time))
    
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
    
    return binned_counts, bin_edges[:-1]  # Return the binned (and optionally smoothed) values and the bin edges

def z_score_normalize(data):
    """Apply z-score normalization to a 1D array."""
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:  # Avoid division by zero
        return data - mean
    return (data - mean) / std_dev

def filter_heatmap_data_for_zoom(smoothed_spikes_matrix, time_bins, zoom_start, zoom_end):
    """Filter the heatmap data to include only the specified zoom window."""
    zoom_indices = (time_bins >= zoom_start) & (time_bins <= zoom_end)
    filtered_matrix = smoothed_spikes_matrix[:, zoom_indices]
    filtered_time_bins = time_bins[zoom_indices]
    return filtered_matrix, filtered_time_bins


def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins):
    axs[0].clear()  # Clear the previous plot
    axs[0].imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot', extent=[time_bins[0], time_bins[-1], len(subunit_names), 0])
    axs[0].set_yticks(ticks=np.arange(len(subunit_names)))
    axs[0].set_yticklabels(labels=subunit_names)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Subunits')
    axs[0].set_title('Heatmap of Smoothed Spike Times Across Subunits')
    fig.canvas.draw()


def plot_combined_kinematics(data, ax, time_array):
    position_colors = {
        'start': 'blue',
        'middle': 'green',
        'tip': 'red',
        'angle_left': 'purple',
        'angle_right': 'orange'
    }

    for key in data['x'].keys():
        color = position_colors[key]
        x_data = data['x'][key]
        y_data = data['y'][key]
        
        for trial_index in range(len(x_data)):
            # Check if data length matches time_array length
            if len(time_array) == len(x_data[trial_index]):
                ax.plot(time_array, x_data[trial_index], label=f'X-{key}' if trial_index == 0 else "", color=color, linestyle='-', alpha=0.6)
                ax.plot(time_array, y_data[trial_index], label=f'Y-{key}' if trial_index == 0 else "", color=color, linestyle='--', alpha=0.6)
            else:
                print(f"Warning: Trial {trial_index} for key '{key}' has mismatched data lengths. Skipping plot for this trial.")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')  # General position label for both x and y
    ax.legend(loc="upper right")
    ax.grid(True)

# Function to get channel selection from entry
def get_channel_selection():
    try:
        channel_input = channel_entry.get()
        selected_channels = []
        if '-' in channel_input:
            parts = channel_input.split(',')
            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_channels.extend(range(start, end + 1))
                else:
                    selected_channels.append(int(part))
        else:
            selected_channels = list(map(int, channel_input.split(',')))
    except ValueError:
        print("Invalid channel input. Please enter channels as numbers separated by commas.")
        return []
    return selected_channels

# Function to get subunit selection from checkboxes
def get_subunit_selection():
    subunits = []
    if subunit1_var.get():
        subunits.append("Subunit 1")
    if subunit2_var.get():
        subunits.append("Subunit 2")
    return subunits

# Function to get T0 value for a specific trial
def get_t0_for_trial(trial_number):
    t_0_values = loaded_data['experiment_data']['t_0']
    if trial_number < len(t_0_values):
        return t_0_values[trial_number]
    print(f"Trial {trial_number + 1} is out of range.")
    return None


def handle_selection():
    """Main function to handle user selections and plot data."""
    # Retrieve current user selections
    selected_channels = get_channel_selection()
    selected_subunits = get_subunit_selection()
    bin_size = int(bin_size_entry.get())
    sigma = float(sigma_entry.get())
    apply_z_score = z_score_var.get()
    trial_based_zoom = trial_zoom_var.get()  # Check if trial-based zoom is enabled

    if trial_based_zoom:
        try:
            trial_number = int(trial_number_entry.get()) - 1  # Get the trial number from user input
        except ValueError:
            print("Invalid trial number. Please enter an integer.")
            return

        # Verify trial_number is within range
        if trial_number < 0 or trial_number >= len(loaded_data['kinematics']['x']['start']):
            print(f"Invalid trial number. Please enter a number between 1 and {len(loaded_data['kinematics']['x']['start'])}.")
            return

        # Extract kinematics data and time array for the specified trial
        kinematics_data = loaded_data['kinematics']
        extracted_kinematics, time_array = get_trial_data_for_plot(kinematics_data, trial_number)

        # Generate heatmap data and subunit names
        smoothed_spikes_matrix, time_bins, subunit_names = generate_heatmap_data_and_subunit_names(
            selected_channels, selected_subunits, bin_size, sigma, apply_z_score
        )

        # Filter heatmap data for the specified zoom window
        zoom_start, zoom_end = time_array[0], time_array[-1]
        filtered_matrix, filtered_time_bins = filter_heatmap_data_for_zoom(smoothed_spikes_matrix, time_bins, zoom_start, zoom_end)

        # Plot filtered heatmap
        plot_heatmap(filtered_matrix, subunit_names, filtered_time_bins)

        # Plot filtered kinematics data
        axs[1].clear()  # Clear the axis before plotting
        plot_combined_kinematics(extracted_kinematics, ax=axs[1], time_array=time_array)
        axs[1].set_title(f'Kinematics Data (Trial {trial_number + 1})')
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        # Generate full data without zoom
        smoothed_spikes_matrix, time_bins, subunit_names = generate_heatmap_data_and_subunit_names(
            selected_channels, selected_subunits, bin_size, sigma, apply_z_score
        )

        # Plot full heatmap
        plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins)

        # Plot full kinematics data
        kinematics_data = loaded_data['kinematics']
        axs[1].clear()  # Clear the axis before plotting
        plot_combined_kinematics(kinematics_data, ax=axs[1])
        axs[1].set_title('Full Kinematics Data')
        fig.canvas.draw()
        fig.canvas.flush_events()

# Create the main tkinter window
root = tk.Tk()
root.title("Neuro-Kinematic Data Visualization")

# Create a frame for the user controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

# Create the matplotlib figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # Two subplots vertically stacked
fig.subplots_adjust(hspace=0.5)
canvas = FigureCanvasTkAgg(fig, master=root)  # Create canvas to embed the plot in Tkinter
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Create entry for channel selection
channel_label = tk.Label(control_frame, text="Channels (e.g., 1, 2, 3-5):")
channel_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
channel_entry = tk.Entry(control_frame)
channel_entry.grid(row=0, column=1, padx=5, pady=5)

# Create checkbox for subunit selection
subunit_label = tk.Label(control_frame, text="Select Subunits:")
subunit_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
subunit1_var = tk.BooleanVar(value=True)  # Default is selected
subunit2_var = tk.BooleanVar(value=True)  # Default is selected
subunit1_checkbox = tk.Checkbutton(control_frame, text="Subunit 1", variable=subunit1_var)
subunit1_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
subunit2_checkbox = tk.Checkbutton(control_frame, text="Subunit 2", variable=subunit2_var)
subunit2_checkbox.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

# Create entry for bin size
bin_size_label = tk.Label(control_frame, text="Bin Size (ms):")
bin_size_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
bin_size_entry = tk.Entry(control_frame)
bin_size_entry.grid(row=2, column=1, padx=5, pady=5)
bin_size_entry.insert(0, "50")  # Default bin size is 50 ms

# Create entry for sigma
sigma_label = tk.Label(control_frame, text="Sigma:")
sigma_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
sigma_entry = tk.Entry(control_frame)
sigma_entry.grid(row=3, column=1, padx=5, pady=5)
sigma_entry.insert(0, "1.0")  # Default sigma is 1.0

# Create checkbox for z-score normalization
z_score_var = tk.BooleanVar(value=False)
z_score_checkbox = tk.Checkbutton(control_frame, text="Apply Z-Score Normalization", variable=z_score_var)
z_score_checkbox.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

# Create entry for trial number selection
trial_number_label = tk.Label(control_frame, text="Trial Number (for zoom):")
trial_number_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
trial_number_entry = tk.Entry(control_frame)
trial_number_entry.grid(row=5, column=1, padx=5, pady=5)

# Create checkbox for trial-based zoom
trial_zoom_var = tk.BooleanVar(value=False)
trial_zoom_checkbox = tk.Checkbutton(control_frame, text="Enable Trial-based Zoom", variable=trial_zoom_var)
trial_zoom_checkbox.grid(row=5, column=2, padx=5, pady=5, sticky=tk.W)

# Add a button to update the plot
update_button = tk.Button(control_frame, text="Update Plot", command=handle_selection)
update_button.grid(row=6, column=0, columnspan=3, pady=10)

# Properly handle the close event to avoid endless loops
def on_closing():
    plt.close('all')  # Close all Matplotlib plots
    root.quit()       # Stop the main Tkinter loop
    root.destroy()    # Destroy the Tkinter window

# Set the close event handler
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter main loop
root.mainloop()
