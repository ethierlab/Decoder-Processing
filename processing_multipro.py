import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from itertools import product
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
from umap import UMAP
import multiprocessing

print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))
else:
    print("Aucun GPU CUDA n'a été détecté.")

# Function to load the pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to extract spike times based on unit selection
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

# Function to bin the spike times and return bin_times
def bin_spike_times(spike_times_list, bin_edges):
    n_neurons = len(spike_times_list)
    n_bins = len(bin_edges) - 1
    spike_counts = np.zeros((n_neurons, n_bins))

    for i, neuron_spike_times in enumerate(spike_times_list):
        if len(neuron_spike_times) > 0:
            counts, _ = np.histogram(neuron_spike_times, bins=bin_edges)
            spike_counts[i, :] = counts
        else:
            spike_counts[i, :] = 0

    return spike_counts

# Function to smooth the data with a Gaussian filter
def smooth_data(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=1)
    return smoothed_data

# Function to apply PCA using PyTorch
def apply_pca(data, n_components=None):
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
    pca_result = torch.mm(data_centered, eigenvectors)
    explained_variance = eigenvalues / torch.sum(eigenvalues)
    return pca_result.cpu().numpy(), explained_variance.cpu().numpy()

# Function to apply UMAP
def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=3):
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    return umap_model.fit_transform(data)

# Function to apply t-SNE
def apply_tsne(data, n_components=3, perplexity=30, n_iter=1000):
    perplexity = min(perplexity, (data.shape[0] - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(data)

# Function to process a combination of bin_size and smoothing_length
def process_combination(bin_size, smoothing_length, spike_times_list, bin_edges, pca_results, explained_variance_results, umap_results, tsne_results):
    binned_data = bin_spike_times(spike_times_list, bin_edges)
    sigma = smoothing_length / bin_size
    if binned_data.size == 0:
        print(f"No data to process for bin_size {bin_size}s and smoothing_length {smoothing_length}s.")
        return
    smoothed_data = smooth_data(binned_data, sigma=sigma)
    smoothed_data_T = smoothed_data.T
    try:
        pca_result, explained_variance = apply_pca(smoothed_data_T)
        pca_results[(bin_size, smoothing_length)] = pca_result[:, :3] if pca_result.shape[1] >= 3 else None
        explained_variance_results[(bin_size, smoothing_length)] = explained_variance
    except Exception as e:
        print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
    try:
        n_neighbors = min(15, smoothed_data_T.shape[0] - 1)
        if n_neighbors >= 2:
            umap_results[(bin_size, smoothing_length)] = apply_umap(smoothed_data_T, n_neighbors=n_neighbors)
    except Exception as e:
        print(f"UMAP failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
    try:
        tsne_results[(bin_size, smoothing_length)] = apply_tsne(smoothed_data_T)
    except Exception as e:
        print(f"t-SNE failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")

# Function to run multiprocessing for different combinations
def run_in_parallel(spike_times_list, duration, bin_sizes, smoothing_lengths):
    manager = multiprocessing.Manager()
    pca_results = manager.dict()
    explained_variance_results = manager.dict()
    umap_results = manager.dict()
    tsne_results = manager.dict()
    bin_edges_dict = {}

    for bin_size in bin_sizes:
        bin_edges = np.arange(0, duration + bin_size, bin_size)
        bin_edges_dict[bin_size] = bin_edges

    processes = []
    for bin_size, smoothing_length in product(bin_sizes, smoothing_lengths):
        process = multiprocessing.Process(
            target=process_combination,
            args=(bin_size, smoothing_length, spike_times_list, bin_edges_dict[bin_size], pca_results, explained_variance_results, umap_results, tsne_results)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    return dict(pca_results), dict(explained_variance_results), dict(umap_results), dict(tsne_results), bin_edges_dict

# Function to plot the results
def plot_grid_results(results, bin_sizes, smoothing_lengths, title_prefix, event_times, bin_edges_dict):
    combinations = list(product(bin_sizes, smoothing_lengths))
    n_rows = len(bin_sizes)
    n_cols = len(smoothing_lengths)

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plot_num = 1
    for bin_size, smoothing_length in combinations:
        result = results.get((bin_size, smoothing_length))
        bin_edges = bin_edges_dict.get(bin_size)
        bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax = fig.add_subplot(n_rows, n_cols, plot_num, projection='3d')

        if result is not None and bin_edges is not None:
            if result.shape[1] >= 3:
                ax.scatter(result[:, 0], result[:, 1], result[:, 2], s=5, alpha=0.3, label='Overall Projection')
                event_indices = [np.argmin(np.abs(bin_times - t)) for t in event_times if t >= bin_times[0] and t <= bin_times[-1]]
                event_data = result[event_indices, :3]
                ax.scatter(event_data[:, 0], event_data[:, 1], event_data[:, 2], s=20, color='r', alpha=0.8, label='Event Times')
            else:
                ax.text(0.5, 0.5, 'Not enough components', horizontalalignment='center', verticalalignment='center')
        else:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')

        ax.text2D(0.05, 0.95, f"Bin Size: {bin_size}s\nSmooth Length: {smoothing_length}s", transform=ax.transAxes)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plot_num += 1

    fig.suptitle(f'{title_prefix} 3D Projections', fontsize=16)
    plt.show()

# Function to plot explained variance
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

    fig.suptitle('PCA Variance Explained', fontsize=16)
    plt.show()

# Main script
if __name__ == '__main__':
    pkl_file = 'experiment_data.pkl'
    tdt_file = 'tdt_signals.pkl'

    data = load_data(pkl_file)
    data_dict = data['data']
    tdt_signals = load_data(tdt_file)
    t_0_times = tdt_signals['Event Time']

    unit_selection = 'both'
    spike_times_dict = extract_spike_times(data_dict, unit_selection)
    spike_times_list = list(spike_times_dict.values())

    if not spike_times_list:
        raise ValueError("No spike times were extracted. Please check your unit selection and data.")

    duration_list = [np.max(np.array(spike_times, dtype=float)) for spike_times in spike_times_list if len(spike_times) > 0]
    if duration_list:
        duration = max(duration_list)
    else:
        raise ValueError("No valid numeric spike times found in the data.")

    bin_sizes = [0.2, 0.5]
    smoothing_lengths = [1, 1.5, 2, 2.5]

    pca_results, explained_variance_results, umap_results, tsne_results, bin_edges_dict = run_in_parallel(
        spike_times_list, duration, bin_sizes, smoothing_lengths
    )

    plot_grid_results(pca_results, bin_sizes, smoothing_lengths, title_prefix='PCA', event_times=t_0_times, bin_edges_dict=bin_edges_dict)
    plot_grid_results(umap_results, bin_sizes, smoothing_lengths, title_prefix='UMAP', event_times=t_0_times, bin_edges_dict=bin_edges_dict)
    plot_grid_results(tsne_results, bin_sizes, smoothing_lengths, title_prefix='t-SNE', event_times=t_0_times, bin_edges_dict=bin_edges_dict)

    plot_variance_explained(explained_variance_results, bin_sizes, smoothing_lengths)
