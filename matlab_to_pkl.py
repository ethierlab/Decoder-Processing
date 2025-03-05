import numpy as np
import pickle
import scipy.io
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

#############################################
# Part 1: Load and convert the .mat file
#############################################

#.mat file name
mat_file = 'Jango_2013-12-09_WF_001_bin.mat'

# Load the MATLAB file
data = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
binnedData = data['binnedData']

# Extract spike rate data, trialtable, and force data from the binnedData structure
spikeratedata_mat = binnedData.spikeratedata  # Expected shape: (N_samples x N_channels)
trialtable = binnedData.trialtable            # First column assumed to be "Event time"
forcedatabin = binnedData.forcedatabin          # Expected shape: (N_samples x 2) for force x and y

# Build a spike data dictionary.
# Format:
# {
#    "Channel01": {"ID_ch1#2": array_of_values}, 
#    "Channel02": {"ID_ch2#2": array_of_values},
#    ...,
#    "Event time": array_of_event_times
# }
spikeratedata = {}
num_channels = spikeratedata_mat.shape[1]
for i in range(num_channels):
    channel_name = f"Channel{i+1:02d}"  # e.g., Channel01, Channel02, ...
    id_name = f"ID_ch{i+1}#2"            # As in the original code
    spike_values = spikeratedata_mat[:, i]
    spikeratedata[channel_name] = {id_name: spike_values}

# Add event times (assumed to be in seconds)
event_time = trialtable[:, 0]
spikeratedata["Event time"] = event_time

# Build a force dictionary
force = {
    "Force": {
        "x": forcedatabin[:, 0],
        "y": forcedatabin[:, 1]
    }
}

print("Loaded and converted .mat file.")
print("Spike data channels:", [k for k in spikeratedata.keys() if k != "Event time"])
print("Number of event times:", len(event_time))
print("Force data length (x):", len(force["Force"]["x"]))

#############################################
# Save the intermediate neural and force data
#############################################
with open('spikerate_data.pkl', 'wb') as f:
    pickle.dump(spikeratedata, f)
print("Neural (spike rate) data saved to spikerate_data.pkl")

with open('force_data.pkl', 'wb') as f:
    pickle.dump(force, f)
print("Force data saved to force_data.pkl")

#############################################
# Part 2: Process the spike data (smoothing, dimensionality reduction, trial extraction)
#############################################

# Parameters
bin_size = 0.05          # Time resolution in seconds
smoothing_length = 0.05
sigma = (smoothing_length / bin_size) / 2  # Gaussian smoothing sigma
window_start = -1.0      # Start of window relative to event (in seconds)
window_end = 4.0         # End of window relative to event (in seconds)
n_component = 30         # Number of components for dimensionality reduction

# Smooth spike rate data
# We skip the "Event time" key and smooth the data for each neuron in each channel.
smoothed_spikerate = {}
for channel, neurons in spikeratedata.items():
    if channel == "Event time":
        continue
    smoothed_spikerate[channel] = {}
    for neuron, data_array in neurons.items():
        smoothed_data = gaussian_filter1d(data_array, sigma).astype(float)
        smoothed_spikerate[channel][neuron] = smoothed_data

# Concatenate smoothed spike rate data for projection.
# For each channel we assume each neuron contributes one column. We then horizontally stack
# the data from all channels. Note that each channel’s data is transposed so that rows are time.
concatenated_data = np.hstack([
    np.array([smoothed_spikerate[channel][neuron] for neuron in smoothed_spikerate[channel]]).T
    for channel in smoothed_spikerate
])
print("Shape of concatenated_data:", concatenated_data.shape)

# Function to perform dimensionality reduction using the specified method.
def perform_reduction(data, method):
    if method == "PCA":
        reducer = PCA(n_components=n_component)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=n_component)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_component)
    else:
        raise ValueError("Invalid reduction method")
    return reducer.fit_transform(data)

# Perform all three reductions
pca_projected = perform_reduction(concatenated_data, "PCA")
umap_projected = perform_reduction(concatenated_data, "UMAP")
tsne_projected = perform_reduction(concatenated_data, "t-SNE")

print("Shape of PCA projected data:", pca_projected.shape)
print("Shape of UMAP projected data:", umap_projected.shape)
print("Shape of t-SNE projected data:", tsne_projected.shape)

#############################################
# Extract trial data based on event times
#############################################

def extract_projected_data_per_trial(data, event_times, bin_size, window_start, window_end):
    """
    For each event time, extract a segment of the projected data corresponding to the 
    window [window_start, window_end] relative to the event. Interpolate the data onto a 
    common time base.
    """
    common_times = np.arange(window_start, window_end, bin_size)
    trial_data_dict = {}
    total_timepoints = data.shape[0]
    
    for idx, t0 in enumerate(event_times):
        # Compute relative times for the full data sequence
        relative_times = np.arange(total_timepoints) * bin_size - t0
        # Find indices corresponding to the desired window
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

        if len(indices) == 0:
            continue

        segment = data[indices, :]        # Segment of the data in this trial window
        times_segment = relative_times[indices]

        # Interpolate each dimension onto the common time grid
        interpolated_data = np.zeros((len(common_times), segment.shape[1]))
        for i in range(segment.shape[1]):
            f = interp1d(times_segment, segment[:, i], kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            interpolated_data[:, i] = f(common_times)

        # Store the trial data as (dimensions x number_of_common_times)
        trial_data_dict[idx] = interpolated_data.T

    return trial_data_dict

# Use event times from the spikeratedata dictionary
event_times = spikeratedata["Event time"]

# Extract trials for each dimensionality reduction result
pca_trials = extract_projected_data_per_trial(pca_projected, event_times, bin_size, window_start, window_end)
umap_trials = extract_projected_data_per_trial(umap_projected, event_times, bin_size, window_start, window_end)
tsne_trials = extract_projected_data_per_trial(tsne_projected, event_times, bin_size, window_start, window_end)

print("Number of PCA trials extracted:", len(pca_trials))
for i in range(min(3, len(pca_trials))):
    print(f"PCA trial {i} shape:", pca_trials[i].shape)

# Process force data per trial.
# We assume the force time series is sampled with the same bin_size.
force_trials = {
    "x": {
        idx: np.array(force["Force"]["x"])[
            int(t0 / bin_size) + int(window_start / bin_size) :
            int(t0 / bin_size) + int(window_end / bin_size)
        ].tolist()
        for idx, t0 in enumerate(event_times)
    },
    "y": {
        idx: np.array(force["Force"]["y"])[
            int(t0 / bin_size) + int(window_start / bin_size) :
            int(t0 / bin_size) + int(window_end / bin_size)
        ].tolist()
        for idx, t0 in enumerate(event_times)
    }
}

# Check force trial lengths
force_x_lengths = [len(x_data) for x_data in force_trials["x"].values()]
force_y_lengths = [len(y_data) for y_data in force_trials["y"].values()]
print("Force X lengths for first few trials:", force_x_lengths[:5])
print("Force Y lengths for first few trials:", force_y_lengths[:5])
print("Total Force X trials:", len(force_x_lengths))
print("Total Force Y trials:", len(force_y_lengths))

#############################################
# Part 3: Save the processed data
#############################################

processed_data = {
    "PCA": pca_trials,
    "UMAP": umap_trials,
    "t-SNE": tsne_trials,
    "Force": force_trials
}

output_path = 'Jango_dataset.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"Processed data saved to {output_path}")
