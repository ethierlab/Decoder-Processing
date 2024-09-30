import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Function to load the pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to extract spike times based on unit selection
def extract_spike_times(spike_times_dict, unit_selection):
    """
    spike_times_dict: nested dictionary containing spike times
    unit_selection: 'unit1', 'unit2', or 'both'
    Returns:
        spike_times_list: list of arrays, each array contains spike times for one neuron
    """
    spike_times_list = []
    for channel_key in spike_times_dict:
        channel_data = spike_times_dict[channel_key]
        channel_number = channel_key.replace('Channel', '')
        units_found = False  # Flag to check if any units are found in the channel

        if unit_selection == 'unit1' or unit_selection == 'both':
            unit_key = f'ID_ch{channel_number}#1'
            if unit_key in channel_data:
                spike_times = channel_data[unit_key]['spike_times']
                spike_times_list.append(spike_times)
                units_found = True

        if unit_selection == 'unit2' or unit_selection == 'both':
            unit_key = f'ID_ch{channel_number}#2'
            if unit_key in channel_data:
                spike_times = channel_data[unit_key]['spike_times']
                spike_times_list.append(spike_times)
                units_found = True

        # Optional: Log if no units are found in a channel
        if not units_found:
            print(f"No units found in {channel_key} for the selected unit(s).")

    return spike_times_list

# Function to bin the spike times
def bin_spike_times(spike_times_list, bin_size, duration):
    """
    spike_times_list: list of arrays, each array contains spike times for one neuron
    bin_size: bin size in ms
    duration: total duration of the recording in ms
    Returns:
        spike_counts: numpy array of shape (n_neurons, n_bins)
    """
    n_neurons = len(spike_times_list)
    n_bins = int(np.ceil(duration / bin_size))
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

# Function to smooth the data with a gaussian filter
def smooth_data(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=1)
    return smoothed_data

# Function to apply PCA
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca_result

# Function to apply UMAP
def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=2):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    umap_result = umap_model.fit_transform(data)
    return umap_result

# Function to apply t-SNE
def apply_tsne(data, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

# Function to plot the grid of results
def plot_grid_results(results, bin_sizes, sigma_values, title_prefix):
    n_rows = len(bin_sizes)
    n_cols = len(sigma_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    for i, bin_size in enumerate(bin_sizes):
        for j, sigma in enumerate(sigma_values):
            result = results.get((bin_size, sigma))
            ax = axes[i, j]
            if result is not None:
                ax.scatter(result[:, 0], result[:, 1], s=5)
                ax.set_title(f'{title_prefix}\nBinSize: {bin_size}ms, Sigma: {sigma}')
            else:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{title_prefix}\nBinSize: {bin_size}ms, Sigma: {sigma}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
    plt.tight_layout()
    plt.show()

# Path to the pkl file
pkl_file = 'experiment_data.pkl'

# Load the data
data = load_data(pkl_file)
spike_times_dict = data['data']['spike_times']

# Define unit selection: 'unit1', 'unit2', or 'both'
unit_selection = 'both'  # Change this to 'unit1' or 'unit2' as needed

# Extract spike times for the selected units
spike_times_list = extract_spike_times(spike_times_dict, unit_selection)

# Check if we have any spike times
if not spike_times_list:
    raise ValueError("No spike times were extracted. Please check your unit selection and data.")

# Calculate the duration
duration_list = [np.max(spike_times) for spike_times in spike_times_list if len(spike_times) > 0]
if duration_list:
    duration = max(duration_list)
else:
    raise ValueError("No spike times found in the data.")

# Define bin sizes and sigma values
bin_sizes = [10, 20, 50, 100]  # in milliseconds
sigma_values = [1, 2, 5, 10]

# Initialize dictionaries to store results
pca_results = {}
umap_results = {}
tsne_results = {}

# Loop over bin sizes and sigma values
for bin_size in bin_sizes:
    for sigma in sigma_values:
        # Bin the spike times
        binned_data = bin_spike_times(spike_times_list, bin_size, duration)
        # Check if binned data is valid
        if binned_data.size == 0:
            print(f"No data to process for bin size {bin_size}ms and sigma {sigma}.")
            continue
        # Smooth the data
        smoothed_data = smooth_data(binned_data, sigma=sigma)
        # Transpose data to have samples as rows
        smoothed_data_T = smoothed_data.T
        # Apply PCA
        try:
            pca_result = apply_pca(smoothed_data_T)
            pca_results[(bin_size, sigma)] = pca_result
        except Exception as e:
            print(f"PCA failed for bin size {bin_size}ms and sigma {sigma}: {e}")
            pca_results[(bin_size, sigma)] = None
        # Apply UMAP
        try:
            umap_result = apply_umap(smoothed_data_T)
            umap_results[(bin_size, sigma)] = umap_result
        except Exception as e:
            print(f"UMAP failed for bin size {bin_size}ms and sigma {sigma}: {e}")
            umap_results[(bin_size, sigma)] = None
        # Apply t-SNE
        try:
            tsne_result = apply_tsne(smoothed_data_T)
            tsne_results[(bin_size, sigma)] = tsne_result
        except Exception as e:
            print(f"t-SNE failed for bin size {bin_size}ms and sigma {sigma}: {e}")
            tsne_results[(bin_size, sigma)] = None

# Visualize the results
plot_grid_results(pca_results, bin_sizes, sigma_values, title_prefix='PCA')
plot_grid_results(umap_results, bin_sizes, sigma_values, title_prefix='UMAP')
plot_grid_results(tsne_results, bin_sizes, sigma_values, title_prefix='t-SNE')
