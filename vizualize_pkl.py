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
    raster_sampling_rate = 24414.1  # 24414.1 Hz for the raster plot data
    time_step_kinematics = 1 / kinematics_sampling_rate
    time_step_force = 1 / force_sampling_rate
    time_step_raster = 1 / raster_sampling_rate

    # Define colors for each position type
    position_colors = {
        'start': 'blue',
        'middle': 'green',
        'tip': 'red',
        'angle_left': 'purple',
        'angle_right': 'orange'
    }

    # Create the figure with four subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    # Initial channel for the raster plot
    initial_channel = 'Channel01'
    update_raster_plot(axs[0], experiment_data['data'], initial_channel, time_step_raster)

    # Dictionary to hold the lines for easy toggling
    plotted_lines = {'x': {}, 'y': {}}

    # 2. X Position Plot
    for key in ['start', 'middle', 'tip', 'angle_left', 'angle_right']:
        color = position_colors[key]
        plotted_lines['x'][key] = []
        for trial_index, trial_data in enumerate(kinematics_data['x'][key]):
            if trial_index >= len(t_0_times):
                print(f"Warning: More trials in kinematics data than in T_0 times. Skipping trial {trial_index}.")
                continue
            start_time = t_0_times[trial_index] - 1  # T_0 - 1 second
            time = np.arange(start_time, start_time + len(trial_data) * time_step_kinematics, time_step_kinematics)
            line, = axs[1].plot(time, trial_data, label=key if trial_index == 0 else "", color=color, alpha=0.6)
            plotted_lines['x'][key].append(line)

    axs[1].set_title('X Positions Aligned with T_0 Times')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('X Position')
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # 3. Y Position Plot
    for key in ['start', 'middle', 'tip', 'angle_left', 'angle_right']:
        color = position_colors[key]
        plotted_lines['y'][key] = []
        for trial_index, trial_data in enumerate(kinematics_data['y'][key]):
            if trial_index >= len(t_0_times):
                print(f"Warning: More trials in kinematics data than in T_0 times. Skipping trial {trial_index}.")
                continue
            start_time = t_0_times[trial_index] - 1  # T_0 - 1 second
            time = np.arange(start_time, start_time + len(trial_data) * time_step_kinematics, time_step_kinematics)
            line, = axs[2].plot(time, trial_data, label=key if trial_index == 0 else "", color=color, alpha=0.6)
            plotted_lines['y'][key].append(line)

    axs[2].set_title('Y Positions Aligned with T_0 Times')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Y Position')
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    # 4. Lever Force Plot
    total_time_force = len(force_signal) * time_step_force
    time_force = np.arange(0, total_time_force, time_step_force)
    axs[3].plot(time_force, force_signal, label='Force Signal')
    axs[3].set_title('Force Signal')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Force')
    axs[3].grid(True)

    # Create TextBox for entering channel number
    axbox = plt.axes([0.05, 0.75, 0.02, 0.02])# Position of the textbox
    text_box = TextBox(axbox, 'Channel', initial="1")

    def submit(text):
        try:
            channel_number = int(text)
            if 1 <= channel_number <= 32:
                channel_name = f'Channel{channel_number:02d}'
                if channel_name in experiment_data['data']:
                    update_raster_plot(axs[0], experiment_data['data'], channel_name, time_step_raster)
                    plt.draw()
                else:
                    print(f"Channel {channel_name} not found in data.")
            else:
                print("Please enter a number between 1 and 32.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 32.")

    text_box.on_submit(submit)

    # Create CheckButtons for selecting which labels to display
    labels = ['start', 'middle', 'tip', 'angle_left', 'angle_right']
    visibility = [True] * len(labels)
    ax_check = plt.axes([0.005, 0.44, 0.06, 0.1])  # Smaller position for the check button
    check = CheckButtons(ax_check, labels, visibility)

    def toggle_visibility(label):
        visible = not plotted_lines['x'][label][0].get_visible()
        for line in plotted_lines['x'][label]:
            line.set_visible(visible)
        for line in plotted_lines['y'][label]:
            line.set_visible(visible)
        plt.draw()

    check.on_clicked(toggle_visibility)

    plt.tight_layout()
    plt.show()

def update_raster_plot(ax, spike_data, channel, time_step_raster):
    ax.clear()
    raster_data = []
    subunit_labels = []

    for subunit_key, subunit_data in spike_data[channel].items():
        spike_times = np.array(subunit_data['spike_times'])
        time = np.arange(0, len(spike_times) * time_step_raster, time_step_raster)
        raster_data.append(spike_times)
        subunit_labels.append(subunit_key)

    ax.eventplot(raster_data, linelengths=0.8)
    ax.set_title(f'Raster Plot for {channel}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Subunits')
    ax.set_yticks(range(len(subunit_labels)))
    ax.set_yticklabels(subunit_labels)
    ax.grid(True)
    plt.draw()

# Example usage
kinematics_file = 'kinematics.pkl'  # Path to the kinematics file
tdt_signals_file = 'tdt_signals.pkl'  # Path to the TDT signals file
experiment_data_file = 'experiment_data.pkl'  # Path to the experiment data file

load_and_visualize_data(kinematics_file, tdt_signals_file, experiment_data_file)
