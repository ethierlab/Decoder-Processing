import tkinter as tk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter1d
import os

global colorbar  # Declare colorbar as a global variable
colorbar = None  # Initialize it to None

# Constants for kinematics data
kinematics_sampling_rate = 200  # 200 Hz
time_step_kinematics = 1 / kinematics_sampling_rate  # Time step in seconds

# Dictionary to store loaded data
loaded_data = {}

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

# Directly load all required files and store them in the `loaded_data` dictionary
loaded_data['kinematics'] = load_pickle_file('kinematics.pkl')
loaded_data['tdt_signals'] = load_pickle_file('tdt_signals.pkl')
loaded_data['experiment_data'] = load_pickle_file('experiment_data.pkl')  # Include the heatmap data

# Verify that all required data has been loaded
if not all(loaded_data.values()):
    print("Some required data files could not be loaded. Please check the file paths.")
else:
    print("All required data files have been successfully loaded.")

# Function to bin and smooth spike times with debug info
def bin_and_smooth_spike_times(spike_times, bin_size=0.005, sigma=None):
    # Calculate the maximum time (should be in seconds)
    max_time = np.ceil(max(spike_times))
    
    # Print bin size and check time unit interpretation
    print(f"Bin Size: {bin_size} seconds")
    print(f"Spike Times Unit (Max Time): {max_time} seconds")

    # Generate bin edges based on bin size
    bin_edges = np.arange(0, max_time + bin_size, bin_size)
    print(f"Generated Bin Edges (in seconds): {bin_edges[:5]}...")  # Print first few for inspection

    # Compute histogram for spike counts
    binned_counts, _ = np.histogram(spike_times, bins=bin_edges)
    print(f"Binned Counts (before smoothing): {binned_counts[:10]}...")  # Print first few for inspection

    # Apply Gaussian smoothing if sigma is provided
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
        print(f"Binned Counts (after smoothing): {binned_counts[:10]}...")  # Print first few for inspection

    # Clip negative values
    binned_counts = np.clip(binned_counts, a_min=0, a_max=None)

    return binned_counts, bin_edges[:-1]  # Return binned values and bin edges
# Function to remove all extra axes that might have been added to the figure
def reset_figure():
    """Remove all axes from the figure and reset it."""
    global axs, fig, colorbar  # Reference the global variables

    # Clear all axes from the figure
    for ax in fig.get_axes():
        fig.delaxes(ax)
    
    # Reinitialize primary axes
    axs = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]
    
    # Clear colorbar reference
    colorbar = None
    
    # Reset the layout
    fig.clear()
    fig.set_size_inches(12, 10)
    print("Figure reset complete. Axes re-initialized.")


# Function to generate a heatmap with optional zooming and a colorbar
def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins, zoom_start=None, zoom_end=None):
    global colorbar  # Reference the global colorbar variable

    # Reset the figure and axes completely
    reset_figure()

    # Convert time to seconds for labeling
    # No conversion needed since time_bins should be in seconds

    # Determine the range to display based on zoom
    if zoom_start is not None and zoom_end is not None:
        zoom_indices = (time_bins >= zoom_start) & (time_bins <= zoom_end)
        time_bins = time_bins[zoom_indices]
        smoothed_spikes_matrix = smoothed_spikes_matrix[:, zoom_indices]

    # Clip the matrix to ensure no negative values
    smoothed_spikes_matrix = np.clip(smoothed_spikes_matrix, a_min=0, a_max=None)

    # Calculate vmin and vmax for better contrast
    vmin = np.min(smoothed_spikes_matrix)
    vmax = np.max(smoothed_spikes_matrix)
    if vmax == vmin:
        vmax = vmin + 1  # Avoid division by zero in visualization

    # Display the heatmap with adjusted vmin and vmax for better visualization
    cax = axs[0].imshow(smoothed_spikes_matrix, aspect='auto', cmap='hot', 
                        extent=[time_bins[0], time_bins[-1], len(subunit_names), 0], 
                        vmin=vmin, vmax=vmax)
    axs[0].set_yticks(ticks=np.arange(len(subunit_names)))
    axs[0].set_yticklabels(labels=subunit_names)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Subunits')
    axs[0].set_title('Heatmap of Smoothed Spike Times Across Subunits')

    # Add a new colorbar next to the heatmap to indicate color values
    colorbar = fig.colorbar(cax, ax=axs[0], orientation='vertical')
    colorbar.set_label('Spike Count / Binned Value')  # Label the colorbar

    # Plot any additional elements on axs[1] if needed
    axs[1].set_title('Additional Plot')
    
    # Redraw the figure with the new layout
    fig.tight_layout()
    fig.canvas.draw()



# Function to z-score normalize the data
def z_score_normalize(data):
    """Apply z-score normalization to a 1D array."""
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:  # Avoid division by zero
        return data - mean
    return (data - mean) / std_dev

# Function to get selected subunits
def get_subunit_selection():
    subunit_1_selected = subunit_1_var.get()
    subunit_2_selected = subunit_2_var.get()
    selected_subunits = []
    if subunit_1_selected:
        selected_subunits.append("Subunit 1")
    if subunit_2_selected:
        selected_subunits.append("Subunit 2")
    return selected_subunits

# Function to get the selected channels
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

# Function to plot kinematics data
def plot_combined_kinematics(data, ax, zoom_start=None, zoom_end=None):
    position_colors = {
        'start': 'blue',
        'middle': 'green',
        'tip': 'red',
        'angle_left': 'purple',
        'angle_right': 'orange'
    }

    t_0_times = loaded_data['tdt_signals']['Event Time']  # Get the T_0 times from loaded data
    selected_types = get_selected_types()

    if x_var.get() == 1:
        plot_kinematics_axis(data, 'x', ax, t_0_times, selected_types, position_colors, zoom_start, zoom_end)
    if y_var.get() == 1:
        plot_kinematics_axis(data, 'y', ax, t_0_times, selected_types, position_colors, zoom_start, zoom_end)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')  # General position label for both x and y
    ax.legend(loc="upper right")
    ax.grid(True)

# Function to plot a single kinematics axis (x or y) with zooming
def plot_kinematics_axis(data, axis, ax, t_0_times, selected_types, position_colors, zoom_start=None, zoom_end=None):
    for key in selected_types:  # Only plot the selected types
        color = position_colors[key]
        for trial_index, trial_data in enumerate(data[axis][key]):
            if trial_index >= len(t_0_times):
                print(f"Warning: More trials in kinematics data than in T_0 times. Skipping trial {trial_index}.")
                continue
            
            # Aligning the kinematics data to T_0 time
            t_0_time = t_0_times[trial_index]  # Get the T_0 time for this trial
            start_time = t_0_time - 1  # T_0 - 1 second
            time = np.arange(start_time, start_time + len(trial_data) * time_step_kinematics, time_step_kinematics)

            # Convert trial_data to numpy array for proper indexing
            trial_data = np.array(trial_data)

            # Apply zoom if specified
            if zoom_start is not None and zoom_end is not None:
                zoom_indices = (time >= zoom_start) & (time <= zoom_end)
                time = time[zoom_indices]
                trial_data = trial_data[zoom_indices]

            ax.plot(time, trial_data, label=f"{axis}-{key}" if trial_index == 0 else "", color=color, alpha=0.6)



# Function to get the selected kinematic types
def get_selected_types():
    selected_types = []
    if start_var.get() == 1:
        selected_types.append('start')
    if middle_var.get() == 1:
        selected_types.append('middle')
    if tip_var.get() == 1:
        selected_types.append('tip')
    if angle_left_var.get() == 1:
        selected_types.append('angle_left')
    if angle_right_var.get() == 1:
        selected_types.append('angle_right')
    return selected_types

# Function to handle the final selection and data processing
def handle_selection():
    selected_subunits = get_subunit_selection()
    selected_channels = get_channel_selection()
    print(f"Selected Subunits: {selected_subunits}")
    print(f"Selected Channels: {selected_channels}")

    # Get sigma, bin_size, z-score, zoom_start, and zoom_end values from GUI
    try:
        bin_size = float(bin_size_entry.get())
        sigma = float(sigma_entry.get())
        zoom_start = float(zoom_start_entry.get()) if zoom_start_entry.get() else None
        zoom_end = float(zoom_end_entry.get()) if zoom_end_entry.get() else None
        # Get z-score value from GUI
        apply_z_score = z_score_var.get()

    except ValueError:
        print("Invalid input for bin size, sigma, or zoom values. Please enter valid numbers.")
        return

    # Initialize the list to store the smoothed data for each subunit
    smoothed_spikes_matrix = []
    subunit_names = []

    # Determine the maximum time length for aligning the heatmap
    max_time_length = 0

    # Flag to check if Subunit 2 is missing
    subunit_2_missing = False

    # Process each channel and ID based on user selection
    for channel, ids in loaded_data['experiment_data']['data'].items():
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

    # Plot kinematics data if either x or y is selected
    axs[1].clear()  # Clear the axis before plotting
    if x_var.get() == 1 or y_var.get() == 1:
        plot_combined_kinematics(loaded_data['kinematics'], ax=axs[1], zoom_start=zoom_start, zoom_end=zoom_end)
    axs[1].set_title('Kinematics Data')
    fig.canvas.draw()
    fig.canvas.flush_events()

    # If Subunit 2 was selected but not available in any channel
    if subunit_2_missing:
        print("Note: Subunit 2 is not available in one or more selected channels.")

# Initialize the main window
root = tk.Tk()
root.title("Kinematics and Heatmap Visualization Tool")
root.geometry("1200x800")  # Set the window size to maximize the space

# Create a frame for controls on the left
control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, fill=tk.Y)

# Create the figure with two subplots for the graphs
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Embed the Matplotlib figure in the Tkinter window on the right side
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Controls in the left frame
# Checkboxes for x and y selection
x_var = tk.IntVar(value=1)  # 1 means selected by default
y_var = tk.IntVar(value=1)

x_checkbox = tk.Checkbutton(control_frame, text="Show X", variable=x_var)
y_checkbox = tk.Checkbutton(control_frame, text="Show Y", variable=y_var)

# Display checkboxes for x and y
x_checkbox.pack()
y_checkbox.pack()

# Checkboxes for each type of kinematic data
start_var = tk.IntVar(value=1)  # 1 means selected by default
middle_var = tk.IntVar(value=1)
tip_var = tk.IntVar(value=1)
angle_left_var = tk.IntVar(value=1)
angle_right_var = tk.IntVar(value=1)

start_checkbox = tk.Checkbutton(control_frame, text="Show Start", variable=start_var)
middle_checkbox = tk.Checkbutton(control_frame, text="Show Middle", variable=middle_var)
tip_checkbox = tk.Checkbutton(control_frame, text="Show Tip", variable=tip_var)
angle_left_checkbox = tk.Checkbutton(control_frame, text="Show Angle Left", variable=angle_left_var)
angle_right_checkbox = tk.Checkbutton(control_frame, text="Show Angle Right", variable=angle_right_var)

# Display checkboxes for each type
start_checkbox.pack()
middle_checkbox.pack()
tip_checkbox.pack()
angle_left_checkbox.pack()
angle_right_checkbox.pack()

# Section for subunit selection
subunit_label = tk.Label(control_frame, text="Select Subunits:")
subunit_label.pack()

# Variables to store checkbox values
subunit_1_var = tk.IntVar(value=1)  # 1 means selected by default
subunit_2_var = tk.IntVar(value=1)

# Checkboxes for subunit selection
subunit_1_checkbox = tk.Checkbutton(control_frame, text="Subunit 1", variable=subunit_1_var)
subunit_2_checkbox = tk.Checkbutton(control_frame, text="Subunit 2", variable=subunit_2_var)

# Display checkboxes for subunit selection
subunit_1_checkbox.pack()
subunit_2_checkbox.pack()

# Section for channel selection
channel_label = tk.Label(control_frame, text="Enter Channels (e.g., 1-3, 5, 7):")
channel_label.pack()

channel_entry = tk.Entry(control_frame)
channel_entry.insert(0, "1-3, 5, 7")  # Default channel value
channel_entry.pack()

# Section for bin size input
bin_size_label = tk.Label(control_frame, text="Enter Bin Size (s):")
bin_size_label.pack()
bin_size_entry = tk.Entry(control_frame)
bin_size_entry.insert(0, "0.005")  # Default value
bin_size_entry.pack()

# Section for sigma input
sigma_label = tk.Label(control_frame, text="Enter Sigma (for Gaussian filter):")
sigma_label.pack()
sigma_entry = tk.Entry(control_frame)
sigma_entry.insert(0, "1.0")  # Default value
sigma_entry.pack()

# Section for z-score normalization option
z_score_var = tk.IntVar(value=0)  # 0 means not selected by default
z_score_checkbox = tk.Checkbutton(control_frame, text="Apply Z-Score Normalization", variable=z_score_var)
z_score_checkbox.pack()

# Section for zoom start and end input
zoom_start_label = tk.Label(control_frame, text="Zoom Start (ms):")
zoom_start_label.pack()
zoom_start_entry = tk.Entry(control_frame)
zoom_start_entry.pack()

zoom_end_label = tk.Label(control_frame, text="Zoom End (ms):")
zoom_end_label.pack()
zoom_end_entry = tk.Entry(control_frame)
zoom_end_entry.pack()

# "Apply" button to update the heatmap and kinematics plots
apply_button = tk.Button(control_frame, text="Apply", command=handle_selection)
apply_button.pack()

# Properly handle the close event to avoid endless loops
def on_closing():
    plt.close('all')  # Close all Matplotlib plots
    root.quit()       # Stop the main Tkinter loop
    root.destroy()    # Destroy the Tkinter window

# Set the close event handler
root.protocol("WM_DELETE_WINDOW", on_closing)

# Show the Tkinter window
root.mainloop()
