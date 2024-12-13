import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Parameters

bin_size = 0.005  # Time resolution in seconds
smoothing_length = 0.05
sigma = (smoothing_length / bin_size) / 2 # Gaussian smoothing sigma
window_start = -1.0  # Start of window relative to event
window_end = 4.0  # End of window relative to event
n_component = 30
# Load data
with open('spikeratedata.pkl', 'rb') as f:
    spikeratedata = pickle.load(f)

with open('force.pkl', 'rb') as f:
    force = pickle.load(f)

# Smooth spikerate data
smoothed_spikerate = {}
for channel, neurons in spikeratedata.items():
    if channel == "Event time":
        continue
    smoothed_spikerate[channel] = {}
    for neuron, data in neurons.items():
        smoothed_spikerate[channel][neuron] = gaussian_filter1d(data, sigma).astype(float)

# Perform dimensionality reduction
def perform_reduction(data, method):
    if method == "PCA":
        reducer = PCA(n_components=n_component)
    # elif method == "UMAP":
    #     reducer = umap.UMAP(n_components=3)
    # elif method == "t-SNE":
    #     reducer = TSNE(n_components=3)
    # else:
    #     raise ValueError("Invalid reduction method")
    return reducer.fit_transform(data)

# Concatenate smoothed spike rate data for projection
concatenated_data = np.hstack([
    np.array([smoothed_spikerate[channel][neuron] for neuron in neurons]).T
    for channel, neurons in smoothed_spikerate.items()
])

pca_projected = perform_reduction(concatenated_data, "PCA")
# umap_projected = perform_reduction(concatenated_data, "UMAP")
# tsne_projected = perform_reduction(concatenated_data, "t-SNE")

# Trial alignment
def extract_projected_data_per_trial(data, event_times, bin_size, window_start, window_end):
    common_times = np.arange(window_start, window_end + bin_size, bin_size)
    trial_data_dict = {}
    for idx, t0 in enumerate(event_times):
        relative_times = np.arange(0, len(data)) * bin_size - t0
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

        if len(indices) == 0:
            continue

        segment = data[indices, :]
        times_segment = relative_times[indices]

        interpolated_data = np.zeros((len(common_times), segment.shape[1]))
        for i in range(segment.shape[1]):
            f = interp1d(times_segment, segment[:, i], kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            interpolated_data[:, i] = f(common_times)

        trial_data_dict[idx] = interpolated_data.T

    return trial_data_dict

# Extract trials
event_times = spikeratedata["Event time"]
pca_trials = extract_projected_data_per_trial(pca_projected, event_times, bin_size, window_start, window_end)
# umap_trials = extract_projected_data_per_trial(umap_projected, event_times, bin_size, window_start, window_end)
# tsne_trials = extract_projected_data_per_trial(tsne_projected, event_times, bin_size, window_start, window_end)

force_trials = {
    "x": {
        idx: np.array(force["Force"]["x"])[
            int(t0 / bin_size) + int(window_start / bin_size): 
            int(t0 / bin_size) + int(window_end / bin_size)
        ].tolist()
        for idx, t0 in enumerate(event_times)
    },
    "y": {
        idx: np.array(force["Force"]["y"])[
            int(t0 / bin_size) + int(window_start / bin_size): 
            int(t0 / bin_size) + int(window_end / bin_size)
        ].tolist()
        for idx, t0 in enumerate(event_times)
    }
}

# Save processed data
processed_data = {
    "PCA": pca_trials,
    # "UMAP": umap_trials,
    # "t-SNE": tsne_trials,
    "Force": force_trials
}

output_path = 'Jango_dataset.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"Processed data saved to {output_path}")
