import pickle
import os
from neo.io import PlexonIO
import numpy as np

def read_plexon_file(file_path):
    """
    Reads a Plexon file, extracts spike times and waveforms for each unit,
    calculates the mean waveform and standard deviation, and organizes the data logically.

    Parameters:
    - file_path (str): The path to the Plexon file (.plx or .pl2).

    Returns:
    - dict: A dictionary containing spike times, waveforms, mean waveforms, and standard deviations for each subunit.
    """
    # Initialize the Plexon reader
    reader = PlexonIO(filename=file_path)
    
    # Read the data block
    block = reader.read_block(lazy=False, load_waveforms=True)

    # Dictionary to store spike data
    spike_data = {}

    # Variables to track the minimum and maximum spike times
    min_time = float('inf')
    max_time = float('-inf')

    # Iterate over each segment (trial or time period)
    for segment_idx, segment in enumerate(block.segments):
        # Iterate over each spiketrain in the segment
        for spiketrain in segment.spiketrains:
            unit_name = spiketrain.name
            unit_id = spiketrain.annotations.get('id', 'Unknown')
            print(f"Processing unit: {unit_name}, ID: {unit_id}")

            # Store spike times (timestamps)
            spike_times = spiketrain.times.magnitude  # Convert to a plain NumPy array

            # Update min and max time to calculate duration
            if len(spike_times) > 0:
                min_time = min(min_time, spike_times.min())
                max_time = max(max_time, spike_times.max())

            # Store spike waveforms if available
            waveforms = spiketrain.waveforms
            if waveforms is not None:
                # Convert waveforms to a plain NumPy array and store
                waveforms = waveforms.magnitude * 1000  # convert mV to µV

                # Ensure waveforms are correctly shaped (num_spikes, num_samples_per_waveform)
                if len(waveforms.shape) == 3:
                    num_spikes, num_channels, num_samples = waveforms.shape
                    waveforms = waveforms.reshape(num_spikes * num_channels, num_samples)
                elif len(waveforms.shape) == 2:
                    num_spikes, num_samples = waveforms.shape
                else:
                    raise ValueError(f"Unexpected waveform shape: {waveforms.shape}")

                # Calculate the mean and standard deviation of the waveforms
                mean_waveform = np.mean(waveforms, axis=0)
                std_waveform = np.std(waveforms, axis=0)

                # Print the shape of the waveforms array to verify
                print(f"Final waveforms shape: {waveforms.shape}")

                # Ensure the spike data dictionary has a sub-unit entry
                if unit_name not in spike_data:
                    spike_data[unit_name] = {}

                # Split into sub-units (e.g., by channel or ID)
                subunit_key = f"ID_{unit_id}"
                spike_data[unit_name][subunit_key] = {
                    'spike_times': spike_times.tolist(),  # convert to list for pickle
                    'waveforms': waveforms.tolist(),  # convert to list for pickle
                    'mean_waveform': mean_waveform.tolist(),  # save the mean waveform
                    'std_waveform': std_waveform.tolist()  # save the standard deviation of the waveform
                }
            else:
                print(f"No waveforms available for Unit: {unit_name}")

    # Calculate duration based on the earliest and latest spike times
    if min_time < float('inf') and max_time > float('-inf'):
        experiment_duration = max_time - min_time
    else:
        experiment_duration = 0.0

    return spike_data, experiment_duration

def save_data_to_pickle(rat_ID, date, duration, behavior, spike_data, output_file):
    """
    Saves the experiment data, including metadata and spike data, to a pickle file.

    Parameters:
    - rat_ID (str): The rat identifier.
    - date (str): The date of the experiment.
    - duration (float): The duration of the experiment in seconds.
    - behavior (str): The behavior being studied.
    - spike_data (dict): The spike data read from the Plexon file.
    - output_file (str): The file path to save the pickle file.
    """
    data_dict = {
        'metadata': {
            'rat_ID': rat_ID,
            'date': date,
            'duration': duration,
            'behavior': behavior,
        },
        'data': spike_data
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data saved to {output_file}")

# Example usage
file_path = 'your_file.pl2'  # Replace with your Plexon file path
rat_ID = 'Rat123'
date = '2024-09-11'
behavior = 'Exploration'

# Read the Plexon file, get spike data and experiment duration
spike_data, experiment_duration = read_plexon_file(file_path)

# Save everything to a pickle file
output_pickle_file = 'experiment_data.pkl'
save_data_to_pickle(rat_ID, date, experiment_duration, behavior, spike_data, output_pickle_file)
