import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count

# Function to load a pickle file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to extract spike times for each subunit
def extract_spike_times(data_dict, unit_selection):
    spike_times_dict = {}
    for channel_key in data_dict:
        channel_data = data_dict[channel_key]
        channel_number = channel_key.replace('Channel', '').lstrip('0')
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
                spike_times_dict[unit_key] = spike_times
            else:
                print(f"Unit {unit_key} not found in {channel_key}.")
    return spike_times_dict

# Function to bin spike times
def bin_spike_times(spike_times_list, bin_size, duration):
    n_neurons = len(spike_times_list)
    n_bins = int(np.ceil(duration / bin_size))
    spike_counts = np.zeros((n_neurons, n_bins))
    bin_edges = np.arange(0, duration + bin_size, bin_size)
    bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centers of bins
    for i, neuron_spike_times in enumerate(spike_times_list):
        if len(neuron_spike_times) > 0:
            counts, _ = np.histogram(neuron_spike_times, bins=bin_edges)
            spike_counts[i, :] = counts
        else:
            spike_counts[i, :] = 0
    return spike_counts, bin_times

# Function to smooth data with a Gaussian filter
def smooth_data(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=1)
    return smoothed_data

# Function to apply PCA using PyTorch
def apply_pca_torch(data, n_components=None, return_components=False):
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    data_mean = torch.mean(data_tensor, dim=0)
    data_centered = data_tensor - data_mean
    cov_matrix = torch.mm(data_centered.t(), data_centered) / (data_centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
    pca_result = torch.mm(data_centered, eigenvectors)
    explained_variance = eigenvalues / torch.sum(eigenvalues)
    pca_result = pca_result.cpu().numpy()
    explained_variance = explained_variance.cpu().numpy()
    eigenvectors = eigenvectors.cpu().numpy()
    if return_components:
        return pca_result, explained_variance, eigenvectors
    else:
        return pca_result, explained_variance

# Function to visualize variance explained by PCA
def plot_variance_explained_single(explained_variance):
    components = np.arange(1, len(explained_variance) + 1)
    cumulative_variance = np.cumsum(explained_variance) * 100
    plt.figure(figsize=(8, 6))
    plt.bar(components, explained_variance * 100, alpha=0.7, label='Variance explained by component')
    plt.plot(components, cumulative_variance, marker='o', color='red', label='Cumulative variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.title('Variance Explained by Principal Components')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to apply UMAP
def apply_umap(data, n_components=3):
    umap = UMAP(n_components=n_components)
    umap_result = umap.fit_transform(data)
    return umap_result

# Function to apply t-SNE
def apply_tsne(data, n_components=3):
    tsne = TSNE(n_components=n_components)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

# Function to project and visualize PCA, UMAP, and t-SNE around events
def project_and_visualize(data, method_name, event_times, bin_size, window_start=-1.0, window_end=2.0, n_components=3, trial_selection='all'):
    # Define a common time grid
    common_times = np.arange(window_start, window_end, bin_size)  # Assuming bin_size is 0.01

    # Initialize a list to store the extracted data for each T_0
    extracted_all_events = []

    # Determine which trials to visualize
    if trial_selection == 'all':
        selected_trials = range(len(event_times))
    elif isinstance(trial_selection, int):
        selected_trials = [trial_selection]
    elif isinstance(trial_selection, (list, tuple)):
        selected_trials = trial_selection
    else:
        raise ValueError("Invalid trial selection. Use 'all', an integer, or a list/tuple of integers.")

    for idx in selected_trials:
        if idx >= len(event_times):
            print(f"Trial index {idx} is out of range. Skipping.")
            continue

        t0 = event_times[idx]

        # Shift times relative to T_0
        relative_times = np.arange(0, len(data)) * 0.01 - t0  # Calculating relative times from data length and bin size

        # Find indices corresponding to the time window [-1s, +2s]
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

        if len(indices) == 0:
            continue

        # Extract data segments for this time window
        segment = data[indices, :n_components]
        times_segment = relative_times[indices]

        # Interpolate onto the common time grid to align results
        interpolated_data = np.zeros((len(common_times), segment.shape[1]))
        for i in range(segment.shape[1]):
            f = interp1d(times_segment, segment[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_data[:, i] = f(common_times)

        extracted_all_events.append(interpolated_data)

        # Visualize the projection of the first 3 components for this event
        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')
        ax.plot(interpolated_data[:, 0], interpolated_data[:, 1], interpolated_data[:, 2])

        # Add markers for specific times (-1s, 0s, +2s)
        time_markers = {
            -1.0: 'red',
            0.0: 'green',
            2.0: 'black'
        }
        for t_mark, color in time_markers.items():
            idx_t = np.where(np.isclose(common_times, t_mark, atol=1e-6))[0]
            if idx_t.size > 0:
                idx_t = idx_t[0]
                ax.scatter(interpolated_data[idx_t, 0], interpolated_data[idx_t, 1], interpolated_data[idx_t, 2], color=color, s=50, marker='o')

        ax.set_xlabel(f'{method_name}1')
        ax.set_ylabel(f'{method_name}2')
        ax.set_zlabel(f'{method_name}3')
        ax.set_title(f'Projection 3 Components for Trial at {t0:.2f}s ({method_name})')
        plt.show()

    return extracted_all_events

# Function to average across all trials
def average_across_trials(extracted_data):
    extracted_data_array = np.array(extracted_data)  # Shape: (n_trials, n_times, n_components)
    average_data = np.mean(extracted_data_array, axis=0)  # Shape: (n_times, n_components)
    return average_data

# Wrapper function for multiprocessing
def process_unit(unit_key, spike_times, bin_size, duration, sigma):
    binned_data, bin_times = bin_spike_times([spike_times], bin_size, duration)
    smoothed_data = smooth_data(binned_data, sigma=sigma)
    return unit_key, smoothed_data

# Main code
if __name__ == "__main__":
    pkl_file = 'experiment_data.pkl'
    tdt_file = 'tdt_signals.pkl'

    data = load_data(pkl_file)
    data_dict = data['data']
    tdt_signals = load_data(tdt_file)
    t_0_times = tdt_signals['Event Time']

    unit_selection = 'unit2'
    spike_times_dict = extract_spike_times(data_dict, unit_selection)

    if not spike_times_dict:
        raise ValueError("No spike times were extracted. Please check your unit selection and data.")

    duration_list = [np.max(spike_times) for spike_times in spike_times_dict.values() if len(spike_times) > 0]
    if duration_list:
        duration = max(duration_list)
    else:
        raise ValueError("No spike times found in the data.")

    bin_size = 0.005
    smoothing_length = 0.05
    sigma = (smoothing_length / bin_size) /2

    # Use multiprocessing to process each unit in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_unit, [(unit_key, spike_times, bin_size, duration, sigma) for unit_key, spike_times in spike_times_dict.items()])

    smoothed_data_dict = {unit_key: smoothed_data for unit_key, smoothed_data in results}

    all_smoothed_data = np.vstack([data for data in smoothed_data_dict.values()])
    all_smoothed_data_T = all_smoothed_data.T

    # Trial selection variable
    trial_selection = 'all'  # Can be 'all', an integer, or a list of integers

    # Apply PCA
    try:
        pca_result, explained_variance, pca_components = apply_pca_torch(all_smoothed_data_T, return_components=True)
    except Exception as e:
        print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
        exit()

    # Visualize variance explained by PCA
    # plot_variance_explained_single(explained_variance)

    # # Apply UMAP
    # umap_result = apply_umap(all_smoothed_data_T)

    # # Apply t-SNE
    # tsne_result = apply_tsne(all_smoothed_data_T)

    # Project and visualize PCA, UMAP, and t-SNE
    pca_extracted = project_and_visualize(pca_result, 'PCA', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection)
    # umap_extracted = project_and_visualize(umap_result, 'UMAP', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection)
    # tsne_extracted = project_and_visualize(tsne_result, 't-SNE', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection)

    # Average across all trials for PCA, UMAP, and t-SNE
    pca_average = average_across_trials(pca_extracted)
    # umap_average = average_across_trials(umap_extracted)
    # tsne_average = average_across_trials(tsne_extracted)

    # Visualize the average for PCA, UMAP, and t-SNE
    for method_name, average_data in zip(['PCA', 'UMAP', 't-SNE'], [pca_average, umap_average, tsne_average]):
        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')
        ax.plot(average_data[:, 0], average_data[:, 1], average_data[:, 2])
        ax.set_xlabel(f'{method_name}1')
        ax.set_ylabel(f'{method_name}2')
        ax.set_zlabel(f'{method_name}3')
        ax.set_title(f'Average Projection 3 Components ({method_name})')
        plt.show()
