import tkinter as tk
import tkinter.ttk as ttk
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

# Global variables
smoothed_spikes_matrix = None
raw_binned_counts_matrix = None
subunit_names = None
time_bins_aligned = None
figures = []
t_0_times = None

# Function to load pickle data
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to bin and smooth spike times
def bin_and_smooth_spike_times(spike_times, bin_size=5, sigma=None):
    spike_times = np.array(spike_times)
    spike_times_ms = spike_times * 1000
    max_time = int(np.ceil(max(spike_times_ms)))
    bin_edges = np.arange(0, max_time + bin_size, bin_size)
    binned_counts, _ = np.histogram(spike_times_ms, bins=bin_edges)
    binned_counts = binned_counts.astype(float)
    if sigma is not None:
        binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
    return binned_counts, bin_edges[:-1]

# Function to get selected subunits
def get_subunit_selection():
    selected_subunits = []
    if subunit_1_var.get():
        selected_subunits.append("Subunit 1")
    if subunit_2_var.get():
        selected_subunits.append("Subunit 2")
    return selected_subunits

# Function to parse channels
def parse_channels(input_text):
    channels = set()
    input_text = input_text.replace(' ', '')
    if input_text:
        for part in input_text.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                channels.update(range(start, end + 1))
            else:
                channels.add(int(part))
    return sorted(channels)

# Function to get selected channels
def get_channel_selection():
    input_text = channel_entry.get()
    channels = parse_channels(input_text)
    return channels

def toggle_trial_selection():
    if plot_all_trials_var.get():
        # Disable the trial dropdown menu
        trial_menu.config(state="disabled")
    else:
        # Enable the trial dropdown menu
        trial_menu.config(state="normal")

# Function to handle data processing
def handle_selection():
    global smoothed_spikes_matrix
    global raw_binned_counts_matrix
    global subunit_names
    global time_bins_aligned
    global t_0_times
    
    # Clear previous figures and listbox entries
    figures.clear()
    figure_listbox.delete(0, tk.END)
    save_button.config(state="disabled")
    peth_button.config(state="disabled")
    average_activity_button.config(state="disabled")
    peth_subunit_menu['menu'].delete(0, 'end')
    
    selected_subunits = get_subunit_selection()
    selected_channels = get_channel_selection()
    apply_z_score = z_score_var.get()
    
    try:
        bin_size = int(bin_size_entry.get())
        smoothing_length = float(smoothing_length_entry.get())
        sigma = smoothing_length / bin_size
        zoom_start = float(zoom_start_entry.get()) if zoom_start_entry.get() else None
        zoom_end = float(zoom_end_entry.get()) if zoom_end_entry.get() else None
    except ValueError:
        print("Invalid input for bin size, smoothing length, or zoom values.")
        return
    
    # Load experiment data
    experiment_data = load_pickle('experiment_data.pkl')
    
    smoothed_spikes_matrix = []
    raw_binned_counts_matrix = []
    subunit_names = []
    max_time_length = 0
    
    for channel, ids in experiment_data['data'].items():
        channel_number = int(channel.split('Channel')[1])
        if channel_number not in selected_channels:
            continue
        for id_, values in ids.items():
            subunit_number = id_.split('#')[1]
            if f"Subunit {subunit_number}" not in selected_subunits:
                continue
            spike_times = values['spike_times']
            binned_counts, time_bins = bin_and_smooth_spike_times(spike_times, bin_size=bin_size, sigma=None)
            binned_counts = binned_counts.astype(float)
            raw_binned_counts_matrix.append(binned_counts)
            smoothed_spikes = gaussian_filter1d(binned_counts, sigma=sigma)
            if apply_z_score:
                smoothed_spikes = z_score_normalize(smoothed_spikes)
            if len(smoothed_spikes) > max_time_length:
                max_time_length = len(smoothed_spikes)
            smoothed_spikes_matrix.append(smoothed_spikes)
            subunit_name = f"{channel_number}#{subunit_number}"
            subunit_names.append(subunit_name)
    
    # Align matrices
    for i in range(len(smoothed_spikes_matrix)):
        if len(smoothed_spikes_matrix[i]) < max_time_length:
            smoothed_spikes_matrix[i] = np.pad(smoothed_spikes_matrix[i],
                                               (0, max_time_length - len(smoothed_spikes_matrix[i])),
                                               'constant')
        if len(raw_binned_counts_matrix[i]) < max_time_length:
            raw_binned_counts_matrix[i] = np.pad(raw_binned_counts_matrix[i],
                                                 (0, max_time_length - len(raw_binned_counts_matrix[i])),
                                                 'constant')
    smoothed_spikes_matrix = np.array(smoothed_spikes_matrix)
    raw_binned_counts_matrix = np.array(raw_binned_counts_matrix)
    time_bins_aligned = np.arange(max_time_length) * bin_size
    
    # Plot heatmap
    plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins_aligned, zoom_start, zoom_end)
    
    # Update PETH subunit dropdown menu
    peth_subunit_var.set(subunit_names[0])
    peth_subunit_menu['menu'].delete(0, 'end')
    for subunit in subunit_names:
        peth_subunit_menu['menu'].add_command(label=subunit, command=tk._setit(peth_subunit_var, subunit))
    peth_subunit_menu.config(state="normal")  # Enable the dropdown menu
    peth_button.config(state="normal")
        
    # Load t_0_times
    tdt_file = tdt_file_entry.get()
    try:
        tdt_signals = load_pickle(tdt_file)
        t_0_times = tdt_signals['Event Time'] * 1000  # Convert to ms if necessary
    except Exception as e:
        print(f"Error loading TDT signals: {e}")
        return
    
    # Populate trial dropdown menu
    trial_numbers = [f"Trial {idx + 1}" for idx in range(len(t_0_times))]
    trial_var.set(trial_numbers[0])  # Set default value
    trial_menu['menu'].delete(0, 'end')
    for trial_label_text in trial_numbers:
        trial_menu['menu'].add_command(label=trial_label_text, command=tk._setit(trial_var, trial_label_text))
    trial_menu.config(state="normal")  # Enable the dropdown menu

    # Reset the "Plot All Trials" checkbox
    plot_all_trials_var.set(0)
    toggle_trial_selection()

    # Enable the average activity button
    average_activity_button.config(state="normal")


    # Enable save button if any figures have been generated
    if figures:
        save_button.config(state="normal")

# Function to plot heatmap
def plot_heatmap(smoothed_spikes_matrix, subunit_names, time_bins, zoom_start=None, zoom_end=None):
    global figures
    fig = plt.figure(figsize=(12, 8))
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
    figures.append(('Heatmap', fig))
    figure_listbox.insert(tk.END, 'Heatmap')
    save_button.config(state="normal")

# Function to compute and plot PETH for selected subunit
def compute_and_plot_peths():
    global smoothed_spikes_matrix
    global time_bins_aligned
    global subunit_names
    global t_0_times
    global figures

    try:
        pre_event_time = float(pre_event_entry.get())
        post_event_time = float(post_event_entry.get())
        bin_size = int(bin_size_entry.get())
    except ValueError:
        print("Invalid input for pre-event or post-event time or bin size.")
        return

    # Check if 'Plot All Subunits' is checked
    if plot_all_subunits_var.get():
        # Plot PETHs for all subunits
        selected_indices = list(range(len(subunit_names)))
        selected_subunits = subunit_names
    else:
        # Get the selected subunit index
        selected_subunit = peth_subunit_var.get()
        try:
            selected_indices = [subunit_names.index(selected_subunit)]
            selected_subunits = [selected_subunit]
        except ValueError:
            print(f"Selected subunit {selected_subunit} not found.")
            return

    # Extract the smoothed spikes for the selected subunits
    selected_smoothed_spikes = smoothed_spikes_matrix[selected_indices]

    # Compute PETHs
    peths_mean, peths_std, time_bins_peth = compute_peths(
        selected_smoothed_spikes,
        time_bins_aligned,
        t_0_times,
        bin_size,
        pre_event_time,
        post_event_time
    )

    # Plot PETHs for the selected subunits
    plot_peths(peths_mean, peths_std, time_bins_peth, selected_subunits, pre_event_time, post_event_time)

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
            if idx_start < 0 or idx_end > len(subunit_spikes):
                continue
            trial_data = subunit_spikes[idx_start:idx_end]
            expected_length = len(time_bins_peth)
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

def plot_peths(peths_mean, peths_std, time_bins_peth, subunit_names, pre_event_time, post_event_time):
    global figures
    global save_button

    num_subunits = len(subunit_names)
    for i in range(num_subunits):
        fig = plt.figure()
        mean = peths_mean[i]
        std = peths_std[i]
        plt.plot(time_bins_peth, mean, label='Mean')
        plt.fill_between(time_bins_peth, mean - std, mean + std, alpha=0.3, label='Std Dev')
        plt.title(f'PETH for Ch{subunit_names[i]}')
        plt.xlabel('Time relative to event (ms)')
        plt.ylabel('Spike Density')
        plt.axvline(0, color='red', linestyle='--')
        plt.legend()
        plt.show()
        fig_label = f"PETH_{subunit_names[i].replace('#', '_')}_pre{int(pre_event_time)}_post{int(post_event_time)}"
        figures.append((fig_label, fig))
        figure_listbox.insert(tk.END, fig_label)
        save_button.config(state="normal")

# Function to compute and plot average activity over subunits for selected trial(s)
def compute_and_plot_average_activity():
    global smoothed_spikes_matrix
    global time_bins_aligned
    global t_0_times
    global figures

    try:
        pre_event_time = float(pre_event_entry.get())
        post_event_time = float(post_event_entry.get())
        bin_size = int(bin_size_entry.get())
    except ValueError:
        print("Invalid input for pre-event or post-event time or bin size.")
        return

    if plot_all_trials_var.get():
        # "Plot All Trials" is checked
        # Loop over all trials
        for trial_index, t0 in enumerate(t_0_times):
            selected_t0_times = [t0]  # Single trial in a list

            # Compute average activity
            average_activity, std_activity, time_bins_peth = compute_average_activity(
                smoothed_spikes_matrix,
                time_bins_aligned,
                selected_t0_times,
                bin_size,
                pre_event_time,
                post_event_time
            )

            # Label for the plot
            trial_number = trial_index + 1
            plot_label = f"Average_Activity_Trial_{trial_number}"

            # Plot the average activity
            plot_average_activity(average_activity, std_activity, time_bins_peth, plot_label)
    else:
        # "Plot All Trials" is unchecked
        # Get the selected trial
        selected_trial_label = trial_var.get()
        if not selected_trial_label:
            print("No trial selected.")
            return

        try:
            # Extract the trial index from the label (e.g., "Trial 1" -> index 0)
            trial_index = int(selected_trial_label.split(' ')[1]) - 1
            selected_t0_times = [t_0_times[trial_index]]  # Single trial in a list
        except Exception as e:
            print(f"Error parsing selected trial: {e}")
            return

        # Compute average activity
        average_activity, std_activity, time_bins_peth = compute_average_activity(
            smoothed_spikes_matrix,
            time_bins_aligned,
            selected_t0_times,
            bin_size,
            pre_event_time,
            post_event_time
        )

        # Label for the plot
        trial_number = trial_index + 1
        plot_label = f"Average_Activity_Trial_{trial_number}"

        # Plot the average activity
        plot_average_activity(average_activity, std_activity, time_bins_peth, plot_label)



def compute_average_activity(smoothed_spikes_matrix, time_bins_aligned, t0_times, bin_size, pre_event_time, post_event_time):
    num_subunits = smoothed_spikes_matrix.shape[0]
    time_bins_peth = np.arange(-pre_event_time, post_event_time + bin_size, bin_size)
    all_subunits_activity = []

    for i in range(num_subunits):
        subunit_spikes = smoothed_spikes_matrix[i]
        trials = []
        for t0 in t0_times:
            window_start = t0 - pre_event_time
            window_end = t0 + post_event_time
            idx_start = np.searchsorted(time_bins_aligned, window_start)
            idx_end = np.searchsorted(time_bins_aligned, window_end)
            if idx_start < 0 or idx_end > len(subunit_spikes):
                continue
            trial_data = subunit_spikes[idx_start:idx_end]
            expected_length = len(time_bins_peth)
            if len(trial_data) < expected_length:
                trial_data = np.pad(trial_data, (0, expected_length - len(trial_data)), 'constant')
            elif len(trial_data) > expected_length:
                trial_data = trial_data[:expected_length]
            trials.append(trial_data)
        if trials:
            trials = np.array(trials)
            # Average across trials for this subunit
            subunit_avg = np.mean(trials, axis=0)
            all_subunits_activity.append(subunit_avg)
        else:
            all_subunits_activity.append(np.zeros(len(time_bins_peth)))
    all_subunits_activity = np.array(all_subunits_activity)
    # Compute mean and std across subunits
    average_activity = np.mean(all_subunits_activity, axis=0)
    std_activity = np.std(all_subunits_activity, axis=0)
    return average_activity, std_activity, time_bins_peth

def plot_average_activity(average_activity, std_activity, time_bins_peth, plot_label):
    global figures
    global save_button
    global figure_listbox

    fig = plt.figure()
    mean = average_activity
    std = std_activity
    plt.plot(time_bins_peth, mean, label='Mean')
    plt.fill_between(time_bins_peth, mean - std, mean + std, alpha=0.3, label='Std Dev')
    plt.xlabel('Time relative to event (ms)')
    plt.ylabel('Spike Density')
    plt.title(f'Average Activity Across Subunits\n(Trials {plot_label.split("_")[-1]})')
    plt.axvline(0, color='red', linestyle='--')
    plt.legend()
    plt.show()
    # Add figure to the list for saving
    figures.append((plot_label, fig))
    figure_listbox.insert(tk.END, plot_label)
    save_button.config(state="normal")

# Function to save selected plots
def save_plots():
    global figures
    selected_indices = figure_listbox.curselection()
    if not selected_indices:
        print("No figures selected to save.")
        return
    directory = os.getcwd()
    print(f"Saving selected plots to current directory: {directory}")
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

# Function to z-score normalize data
def z_score_normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:
        return data
    return (data - mean) / std_dev

# GUI Setup
root = tk.Tk()
root.title("Interactive Selection Tool")

# Variables for checkboxes
plot_all_subunits_var = tk.IntVar(value=0)  # For 'Plot All Subunits' checkbox
select_all_trials_var = tk.IntVar(value=0)  # For 'Select All Trials' checkbox


# Variable for 'Select All Trials' checkbox
select_all_trials_var = tk.IntVar(value=0)  # 0 means unchecked by default

# Frames for layout
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Subunit and Channel Selection
selection_frame = tk.LabelFrame(left_frame, text="Selection")
selection_frame.pack(fill=tk.X, padx=5, pady=5)

# TDT file path
tdt_file_label = tk.Label(selection_frame, text="Enter TDT File Path:")
tdt_file_label.grid(row=4, column=0, sticky=tk.W)
tdt_file_entry = tk.Entry(selection_frame)
tdt_file_entry.insert(0, "tdt_signals.pkl")
tdt_file_entry.grid(row=4, column=1, sticky=tk.W)

# Signal file path
signal_file_label = tk.Label(selection_frame, text="Enter Signal File Path:")
signal_file_label.grid(row=5, column=0, sticky=tk.W)
signal_file_entry = tk.Entry(selection_frame)
signal_file_entry.insert(0, "experiment_data.pkl")
signal_file_entry.grid(row=5, column=1, sticky=tk.W)


# Variables
subunit_1_var = tk.IntVar(value=1)
subunit_2_var = tk.IntVar(value=1)
z_score_var = tk.IntVar(value=0)
peth_subunit_var = tk.StringVar()

# Subunit selection
subunit_label = tk.Label(selection_frame, text="Select Subunits:")
subunit_label.grid(row=0, column=0, sticky=tk.W)
subunit_1_checkbox = tk.Checkbutton(selection_frame, text="Subunit 1", variable=subunit_1_var)
subunit_2_checkbox = tk.Checkbutton(selection_frame, text="Subunit 2", variable=subunit_2_var)
subunit_1_checkbox.grid(row=1, column=0, sticky=tk.W)
subunit_2_checkbox.grid(row=1, column=1, sticky=tk.W)

# Channel selection
channel_label = tk.Label(selection_frame, text="Enter Channels (e.g., 1-3,5,7):")
channel_label.grid(row=2, column=0, sticky=tk.W)
channel_entry = tk.Entry(selection_frame)
channel_entry.insert(0, '1-32')
channel_entry.grid(row=2, column=1, sticky=tk.W)

# Z-score normalization
z_score_checkbox = tk.Checkbutton(selection_frame, text="Apply Z-Score Normalization", variable=z_score_var)
z_score_checkbox.grid(row=3, column=0, columnspan=2, sticky=tk.W)

# Parameters Frame
parameters_frame = tk.LabelFrame(left_frame, text="Parameters")
parameters_frame.pack(fill=tk.X, padx=5, pady=5)

# Bin size
bin_size_label = tk.Label(parameters_frame, text="Enter Bin Size (ms):")
bin_size_label.grid(row=0, column=0, sticky=tk.W)
bin_size_entry = tk.Entry(parameters_frame)
bin_size_entry.insert(0, "5")
bin_size_entry.grid(row=0, column=1, sticky=tk.W)

# Smoothing length
smoothing_length_label = tk.Label(parameters_frame, text="Smoothing Length (ms):")
smoothing_length_label.grid(row=1, column=0, sticky=tk.W)
smoothing_length_entry = tk.Entry(parameters_frame)
smoothing_length_entry.insert(0, "50")
smoothing_length_entry.grid(row=1, column=1, sticky=tk.W)

# Zoom start
zoom_start_label = tk.Label(parameters_frame, text="Zoom Start Time (ms):")
zoom_start_label.grid(row=2, column=0, sticky=tk.W)
zoom_start_entry = tk.Entry(parameters_frame)
zoom_start_entry.grid(row=2, column=1, sticky=tk.W)

# Zoom end
zoom_end_label = tk.Label(parameters_frame, text="Zoom End Time (ms):")
zoom_end_label.grid(row=3, column=0, sticky=tk.W)
zoom_end_entry = tk.Entry(parameters_frame)
zoom_end_entry.grid(row=3, column=1, sticky=tk.W)

# Process Data button
selection_button = tk.Button(parameters_frame, text="Process Data", command=handle_selection)
selection_button.grid(row=4, column=0, columnspan=2, pady=5)

# PETH Frame
peth_frame = tk.LabelFrame(left_frame, text="PETH Plotting")
peth_frame.pack(fill=tk.X, padx=5, pady=5)

# Variable to store the state of the 'Plot All Subunits' checkbox
plot_all_subunits_var = tk.IntVar(value=0)  # 0 means unchecked by default
# Variable for "Plot All Trials" checkbox
plot_all_trials_var = tk.IntVar(value=0)

# Add the checkbox to the peth_frame
plot_all_subunits_checkbox = tk.Checkbutton(peth_frame, text="Plot All Subunits", variable=plot_all_subunits_var)
plot_all_subunits_checkbox.grid(row=4, column=0, columnspan=2, sticky=tk.W)

# PETH subunit selection
peth_subunit_label = tk.Label(peth_frame, text="Select Subunit for PETH Plot:")
peth_subunit_label.grid(row=0, column=0, sticky=tk.W)
peth_subunit_menu = tk.OptionMenu(peth_frame, peth_subunit_var, [])
peth_subunit_menu.config(state="disabled")
peth_subunit_menu.grid(row=0, column=1, sticky=tk.W)

# Pre-event time
pre_event_label = tk.Label(peth_frame, text="Pre-Event Time (ms):")
pre_event_label.grid(row=1, column=0, sticky=tk.W)
pre_event_entry = tk.Entry(peth_frame)
pre_event_entry.insert(0, "1000")
pre_event_entry.grid(row=1, column=1, sticky=tk.W)

# Post-event time
post_event_label = tk.Label(peth_frame, text="Post-Event Time (ms):")
post_event_label.grid(row=2, column=0, sticky=tk.W)
post_event_entry = tk.Entry(peth_frame)
post_event_entry.insert(0, "2000")
post_event_entry.grid(row=2, column=1, sticky=tk.W)


# Compute and Plot PETHs button
peth_button = tk.Button(peth_frame, text="Compute and Plot PETHs", command=compute_and_plot_peths)
peth_button.config(state="disabled")
peth_button.grid(row=5, column=0, columnspan=2, pady=5)

# Average Activity Frame
average_activity_frame = tk.LabelFrame(left_frame, text="Average Activity Over Subunits")
average_activity_frame.pack(fill=tk.X, padx=5, pady=5)

# Variable for trial selection
trial_var = tk.StringVar()

# Trial selection
trial_label = tk.Label(average_activity_frame, text="Select Trial:")
trial_label.grid(row=0, column=0, sticky=tk.W)
trial_menu = tk.OptionMenu(average_activity_frame, trial_var, [])
trial_menu.config(state="disabled")  # Initially disabled
trial_menu.grid(row=0, column=1, sticky=tk.W)

# Plot All Trials checkbox
plot_all_trials_checkbox = tk.Checkbutton(average_activity_frame, text="Plot All Trials", variable=plot_all_trials_var, command=toggle_trial_selection)
plot_all_trials_checkbox.grid(row=1, column=0, columnspan=2, sticky=tk.W)

# Compute and Plot Average Activity button
average_activity_button = tk.Button(average_activity_frame, text="Compute and Plot Average Activity", command=compute_and_plot_average_activity)
average_activity_button.config(state="disabled")
average_activity_button.grid(row=2, column=0, columnspan=2, pady=5)


# Save Plots button
save_button = tk.Button(left_frame, text="Save Selected Plots", command=save_plots)
save_button.config(state="disabled")
save_button.pack(pady=5)

# Generated Figures Listbox
figure_listbox_label = tk.Label(right_frame, text="Generated Figures:")
figure_listbox_label.pack()
figure_listbox = tk.Listbox(right_frame, selectmode=tk.MULTIPLE, width=50, height=20)
figure_listbox.pack()

# Start the main loop
root.mainloop()
