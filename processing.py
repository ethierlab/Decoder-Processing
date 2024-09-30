import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Function to load the pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to extract spike times based on unit selection
def extract_spike_times(data_dict, unit_selection):
    spike_times_list = []
    for channel_key in data_dict:
        channel_data = data_dict[channel_key]
        units_found = False  # Flag to check if any units are found in the channel

        # Extract the channel number from the channel key
        channel_number = channel_key.replace('Channel', '').lstrip('0')  # Remove 'Channel' and leading zeros

        # Depending on unit selection, select unit keys
        unit_keys = []
        if unit_selection == 'unit1' or unit_selection == 'both':
            unit_key1 = f'ID_ch{channel_number}#1'
            unit_keys.append(unit_key1)
        if unit_selection == 'unit2' or unit_selection == 'both':
            unit_key2 = f'ID_ch{channel_number}#2'
            unit_keys.append(unit_key2)

        for unit_key in unit_keys:
            if unit_key in channel_data:
                spike_times = channel_data[unit_key]['spike_times']
                spike_times_list.append(spike_times)
                units_found = True
            else:
                # Optional: print that the unit was not found
                # print(f"Unit {unit_key} not found in {channel_key}.")
                pass

        # Optional: Log if no units are found in a channel
        if not units_found:
            print(f"No selected units found in {channel_key}.")

    return spike_times_list

# Function to bin the spike times
def bin_spike_times(spike_times_list, bin_size, duration):
    n_neurons = len(spike_times_list)
    n_bins = int(np.ceil(duration / bin_size))
    print(f"Number of bins for bin_size={bin_size}s: {n_bins}")  # Added print for the number of bins
    spike_counts = np.zeros((n_neurons, n_bins))

    bin_edges = np.arange(0, duration + bin_size, bin_size)

    for i, neuron_spike_times in enumerate(spike_times_list):
        if len(neuron_spike_times) > 0:
            counts, _ = np.histogram(neuron_spike_times, bins=bin_edges)
            spike_counts[i, :] = counts
        else:
            # Handle neurons with no spikes
            spike_counts[i, :] = 0
    return spike_counts

# Function to smooth the data with a Gaussian filter
def smooth_data(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=1)
    return smoothed_data

# Function to apply PCA and return explained variance ratios
def apply_pca(data):
    pca = PCA(n_components=None)  # Compute all principal components
    pca_result = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# Function to apply UMAP
def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=3):
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    umap_result = umap_model.fit_transform(data)
    return umap_result

# Function to apply t-SNE
def apply_tsne(data, n_components=3, perplexity=30, n_iter=1000):
    n_samples = data.shape[0]
    # Adjust perplexity if necessary
    perplexity = min(perplexity, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5  # Set a minimum perplexity
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

# Function to plot the 4x4 grid of results
def plot_grid_results(results, bin_sizes, smoothing_lengths, title_prefix):
    n_rows = len(bin_sizes)
    n_cols = len(smoothing_lengths)
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plot_num = 1
    for i, bin_size in enumerate(bin_sizes):
        for j, smoothing_length in enumerate(smoothing_lengths):
            result = results.get((bin_size, smoothing_length))
            ax = fig.add_subplot(n_rows, n_cols, plot_num, projection='3d')
            if result is not None:
                # Use the first 3 principal components for plotting
                if result.shape[1] >= 3:
                    ax.scatter(result[:, 0], result[:, 1], result[:, 2], s=5)
                else:
                    ax.text(0.5, 0.5, 'Not enough components', horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            # Annotate bin size and smoothing length within the subplot
            ax.text2D(0.05, 0.95, f"Bin: {bin_size}s\nSmooth: {smoothing_length}s", transform=ax.transAxes)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plot_num += 1
    # Add a global title
    fig.suptitle(f'{title_prefix} 3D Projections', fontsize=16)
    plt.show()

# Function to plot the variance explained graphs in a 4x4 grid
def plot_variance_explained(explained_variance_dict, bin_sizes, smoothing_lengths):
    n_rows = len(bin_sizes)
    n_cols = len(smoothing_lengths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, bin_size in enumerate(bin_sizes):
        for j, smoothing_length in enumerate(smoothing_lengths):
            explained_variance = explained_variance_dict.get((bin_size, smoothing_length))
            ax = axes[i, j]
            if explained_variance is not None:
                components = np.arange(1, len(explained_variance) + 1)
                cumulative_variance = np.cumsum(explained_variance) * 100
                ax.bar(components, explained_variance * 100)
                ax.plot(components, cumulative_variance, marker='o', color='red')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Variance Explained (%)')
                ax.set_ylim(0, 100)
                ax.set_title(f"Bin: {bin_size}s, Smooth: {smoothing_length}s")
            else:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
    # Add a global title
    fig.suptitle('PCA Variance Explained', fontsize=16)
    plt.show()

# Path to the pkl file
pkl_file = 'experiment_data.pkl'

# Load the data
data = load_data(pkl_file)
data_dict = data['data']  # The dictionary containing channel data

# Define unit selection: 'unit1', 'unit2', or 'both'
unit_selection = 'both'  # Change this to 'unit1' or 'unit2' as needed

# Extract spike times for the selected units
spike_times_list = extract_spike_times(data_dict, unit_selection)

# Do NOT convert spike times to milliseconds; they are already in seconds
# spike_times_list = [spike_times * 1000 for spike_times in spike_times_list]

# Check if we have any spike times
if not spike_times_list:
    raise ValueError("No spike times were extracted. Please check your unit selection and data.")

# Calculate the duration in seconds
duration_list = [np.max(spike_times) for spike_times in spike_times_list if len(spike_times) > 0]
if duration_list:
    duration = max(duration_list)
else:
    raise ValueError("No spike times found in the data.")

# Define bin sizes and desired smoothing lengths in seconds
bin_sizes = [0.005, 0.01, 0.02, 0.05]  # in seconds
smoothing_lengths = [0.01, 0.02, 0.05, 0.1]  # in seconds

# Initialize dictionaries to store results
pca_results = {}
pca_variance = {}
umap_results = {}
tsne_results = {}

# Loop over bin sizes and desired smoothing lengths
for bin_size in bin_sizes:
    for smoothing_length in smoothing_lengths:
        # Compute sigma to achieve desired smoothing length
        sigma = smoothing_length / bin_size
        print(f"Processing bin_size: {bin_size}s, smoothing_length: {smoothing_length}s, computed sigma: {sigma}")
        # Bin the spike times
        binned_data = bin_spike_times(spike_times_list, bin_size, duration)
        # Check if binned data is valid
        if binned_data.size == 0:
            print(f"No data to process for bin_size {bin_size}s and smoothing_length {smoothing_length}s.")
            continue
        # Smooth the data
        smoothed_data = smooth_data(binned_data, sigma=sigma)
        # Transpose data to have samples as rows (time bins)
        smoothed_data_T = smoothed_data.T

        # Apply PCA
        try:
            pca_result, explained_variance = apply_pca(smoothed_data_T)
            # Use only the first 3 principal components for projections
            if pca_result.shape[1] >= 3:
                pca_results[(bin_size, smoothing_length)] = pca_result[:, :3]
            else:
                pca_results[(bin_size, smoothing_length)] = None
            pca_variance[(bin_size, smoothing_length)] = explained_variance
        except Exception as e:
            print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
            pca_results[(bin_size, smoothing_length)] = None
            pca_variance[(bin_size, smoothing_length)] = None
        # Apply UMAP
        try:
            n_neighbors = min(15, smoothed_data_T.shape[0] - 1)
            if n_neighbors < 2:
                raise ValueError("Not enough samples for UMAP.")
            umap_result = apply_umap(smoothed_data_T, n_neighbors=n_neighbors)
            umap_results[(bin_size, smoothing_length)] = umap_result
        except Exception as e:
            print(f"UMAP failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
            umap_results[(bin_size, smoothing_length)] = None
        # Apply t-SNE
        try:
            tsne_result = apply_tsne(smoothed_data_T)
            tsne_results[(bin_size, smoothing_length)] = tsne_result
        except Exception as e:
            print(f"t-SNE failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
            tsne_results[(bin_size, smoothing_length)] = None

# Visualize the results
plot_grid_results(pca_results, bin_sizes, smoothing_lengths, title_prefix='PCA')
plot_variance_explained(pca_variance, bin_sizes, smoothing_lengths)
plot_grid_results(umap_results, bin_sizes, smoothing_lengths, title_prefix='UMAP')
plot_grid_results(tsne_results, bin_sizes, smoothing_lengths, title_prefix='t-SNE')
