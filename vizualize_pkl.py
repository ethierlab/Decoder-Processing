import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, TextBox

def load_and_visualize_data(kinematics_file, tdt_signals_file, experiment_data_file):
    # Load kinematics and TDT signals data
    with open(kinematics_file, 'rb') as f:
        kinematics_data = pickle.load(f)
    
    with open(tdt_signals_file, 'rb') as f:
        tdt_signals = pickle.load(f)

    with open(experiment_data_file, 'rb') as f:
        experiment_data = pickle.load(f)
    
    # Extract T_0 times and force signal from TDT signals
    t_0_times = tdt_signals['Event Time']
    force_signal = tdt_signals['Levier']
    force_sampling_rate = 1017.3  # 1017.3 Hz for the force signal
    kinematics_sampling_rate = 200  # 200 Hz for the kinematic data
    time_step_kinematics = 1 / kinematics_sampling_rate

    # Define colors for each position type
    position_colors = {
        'start': 'blue',
        'middle': 'green',
        'tip': 'red',
        'angle_left': 'purple',
        'angle_right': 'orange'
    }

    # Create the figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    
    # Initial channel for the raster plot
    initial_channel = 'Channel01'
    update_raster_plot(axs[0], experiment_data['data'], initial_channel)

    # Dictionary to hold the lines for easy toggling
    plotted_lines = {'x': {}, 'y': {}}

    # Combined X and Y Position Plot
    for key in ['start', 'middle', 'tip', 'angle_left', 'angle_right']:
        color = position_colors[key]
        plotted_lines['x'][key] = []
        plotted_lines['y'][key] = []
        for trial_index, trial_data_x in enumerate(kinematics_data['x'][key]):
            trial_data_y = kinematics_data['y'][key][trial_index]  # Corresponding y data
            if trial_index >= len(t_0_times):
                print(f"Warning: More trials in kinematics data than in T_0 times. Skipping trial {trial_index}.")
                continue
            start_time = t_0_times[trial_index] - 1  # T_0 - 1 second
            time = np.arange(start_time, start_time + len(trial_data_x) * time_step_kinematics, time_step_kinematics)
            # Plot X and Y data
            line_x, = axs[1].plot(time, trial_data_x, color=color, alpha=0.6, linestyle='--')
            line_y, = axs[1].plot(time, trial_data_y, color=color, alpha=0.6)
            plotted_lines['x'][key].append(line_x)
            plotted_lines['y'][key].append(line_y)

    axs[1].set_title('X and Y Positions Aligned with T_0 Times')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position')
    axs[1].grid(True)

    # Force Plot (unchanged)
    total_time_force = len(force_signal) * (1 / force_sampling_rate)
    time_force = np.arange(0, total_time_force, 1 / force_sampling_rate)
    axs[2].plot(time_force, force_signal, label='Force Signal')
    axs[2].set_title('Force Signal')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Force')
    axs[2].grid(True)

    # Create CheckButtons for selecting which position types and X/Y to display
    position_labels = ['start', 'middle', 'tip', 'angle_left', 'angle_right']
    visibility_pos = [True] * len(position_labels)
    ax_check_pos = plt.axes([0.005, 0.44, 0.1, 0.2])  # Adjust position for position type checkboxes
    check_pos = CheckButtons(ax_check_pos, position_labels, visibility_pos)

    # Create CheckButtons for selecting which labels to display (X, Y)
    display_labels = ['X', 'Y']
    visibility_xy = [True, True]
    ax_check_xy = plt.axes([0.005, 0.64, 0.1, 0.1])  # Adjust position for X/Y checkboxes
    check_xy = CheckButtons(ax_check_xy, display_labels, visibility_xy)

    # Flags to track visibility
    xy_visibility = {'X': True, 'Y': True}
    position_visibility = {key: True for key in position_labels}

    def update_plot():
        """Update the visibility of the plots based on the current checkbox selections."""
        for key in position_labels:
            x_visible = xy_visibility['X'] and position_visibility[key]
            y_visible = xy_visibility['Y'] and position_visibility[key]

            for line_x in plotted_lines['x'][key]:
                line_x.set_visible(x_visible)
            for line_y in plotted_lines['y'][key]:
                line_y.set_visible(y_visible)

        plt.draw()

    def toggle_visibility_position(label):
        """Toggle visibility for specific position types."""
        position_visibility[label] = not position_visibility[label]
        update_plot()

    check_pos.on_clicked(toggle_visibility_position)

    def toggle_visibility_xy(label):
        """Toggle visibility for X or Y."""
        xy_visibility[label] = not xy_visibility[label]
        update_plot()

    check_xy.on_clicked(toggle_visibility_xy)

    plt.tight_layout()
    plt.show()

def update_raster_plot(ax, spike_data, channel):
    ax.clear()
    raster_data = []
    subunit_labels = []

    for subunit_key, subunit_data in spike_data[channel].items():
        spike_times = np.array(subunit_data['spike_times'])
        raster_data.append(spike_times)
        subunit_labels.append(subunit_key)

    ax.eventplot(raster_data, linelengths=0.8)
    ax.set_title(f'Raster Plot for {channel}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Subunits')
    ax.grid(True)
    plt.draw()

# Example usage
kinematics_file = 'kinematics.pkl'  # Path to the kinematics file
tdt_signals_file = 'tdt_signals.pkl'  # Path to the TDT signals file
experiment_data_file = 'experiment_data.pkl'  # Path to the experiment data file

load_and_visualize_data(kinematics_file, tdt_signals_file, experiment_data_file)
