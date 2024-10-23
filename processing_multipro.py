import numpy as np
import pickle
import torch
import multiprocessing
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
from umap import UMAP


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
        
    # Ensure n_neighbors is valid
    if n_neighbors >= data.shape[0]:
        raise ValueError(f"n_neighbors ({n_neighbors}) is larger than the number of samples ({data.shape[0]})")
    
    # Run UMAP
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    umap_result = umap_model.fit_transform(data)
    
    if umap_result is None or umap_result.size == 0:
        raise ValueError("UMAP returned an empty result.")
    
    return umap_result


# Function to apply t-SNE
def apply_tsne(data, n_components=3, perplexity=30, n_iter=1000, learning_rate=200, early_exaggeration=12, metric='euclidean'):
    perplexity = min(perplexity, (data.shape[0] - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        n_iter=n_iter, 
        learning_rate=learning_rate, 
        early_exaggeration=early_exaggeration, 
        metric=metric
    )
    return tsne.fit_transform(data)

def process_combination(bin_size, smoothing_length, spike_times_list, bin_edges, results_dict, method_configs):
    print(f"[Process PID {multiprocessing.current_process().pid}] Started processing for bin_size={bin_size}, smoothing_length={smoothing_length}")

    binned_data = bin_spike_times(spike_times_list, bin_edges)
    sigma = (smoothing_length / bin_size) / 2
    if binned_data.size == 0:
        print(f"No data to process for bin_size {bin_size}s and smoothing_length {smoothing_length}s.")
        return

    smoothed_data = smooth_data(binned_data, sigma=sigma)
    smoothed_data_T = smoothed_data.T

    # Check for NaNs or infinite values in the data
    if np.any(np.isnan(smoothed_data_T)) or np.any(np.isinf(smoothed_data_T)):
        print(f"Invalid data (NaNs or infinite values) detected for bin_size {bin_size}s and smoothing_length {smoothing_length}s")
        return

    # Run the selected methods and configurations
    for method, configs in method_configs.items():
        for config_index, config in enumerate(configs):
            result_key = (method, config_index, bin_size, smoothing_length)
            print(f"[Process PID {multiprocessing.current_process().pid}] Processing {method} config {config_index} for bin_size={bin_size}, smoothing_length={smoothing_length}")

            if method == 'PCA':
                try:
                    pca_result, explained_variance = apply_pca(smoothed_data_T, n_components=config['n_components'])
                    results_dict[result_key] = (pca_result[:, :3], explained_variance)
                    print(f"[Process PID {multiprocessing.current_process().pid}] Completed PCA config {config_index}")
                except Exception as e:
                    print(f"PCA failed for config {config_index}, bin_size {bin_size}s: {e}")
            elif method == 'UMAP':
                try:
                    n_neighbors = min(config['n_neighbors'], smoothed_data_T.shape[0] - 1)
                    if n_neighbors < 2:
                        print(f"n_neighbors too small for UMAP config {config_index}, bin_size {bin_size}s")
                        continue

                    umap_result = apply_umap(smoothed_data_T, n_neighbors=config['n_neighbors'], min_dist=config['min_dist'], n_components=config['n_components'])

                    results_dict[result_key] = umap_result
                    print(f"[Process PID {multiprocessing.current_process().pid}] Completed UMAP config {config_index}")


                except Exception as e:
                    print(f"UMAP failed for config {config_index}, bin_size {bin_size}s: {e}")

            elif method == 't-SNE':
                try:
                    tsne_result = apply_tsne(
                        smoothed_data_T,
                        n_components=config['n_components'],
                        perplexity=config['perplexity'],
                        n_iter=config['n_iter'],
                        learning_rate=config['learning_rate'],
                        early_exaggeration=config['early_exaggeration'],
                        metric=config['metric']
                    )
                    results_dict[result_key] = tsne_result
                    print(f"[Process PID {multiprocessing.current_process().pid}] Completed t-SNE config {config_index}")
                except Exception as e:
                    print(f"t-SNE failed for config {config_index}, bin_size {bin_size}s: {e}")

    print(f"[Process PID {multiprocessing.current_process().pid}] Finished processing for bin_size={bin_size}, smoothing_length={smoothing_length}")


# Function to run multiprocessing for different combinations
def run_in_parallel(spike_times_list, duration, bin_sizes, smoothing_lengths, method_configs):
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    bin_edges_dict = {}
    total_combinations = len(bin_sizes) * len(smoothing_lengths)
    
    for bin_size in bin_sizes:
        bin_edges = np.arange(0, duration + bin_size, bin_size)
        bin_edges_dict[bin_size] = bin_edges

    processes = []
    for bin_size, smoothing_length in product(bin_sizes, smoothing_lengths):
        process = multiprocessing.Process(
            target=process_combination,
            args=(bin_size, smoothing_length, spike_times_list, bin_edges_dict[bin_size], results_dict, method_configs)
        )
        processes.append(process)
        process.start()

    # Use tqdm to display progress as processes complete
    for process in tqdm(processes, desc='Processing combinations', unit='process'):
        process.join()

    # Convert manager dict to a regular dict
    return dict(results_dict), bin_edges_dict

# Function to plot the results
def plot_grid_results(results, bin_sizes, smoothing_lengths, title_prefix, event_times, bin_edges_dict, display='all', graph='group', max_plots_per_figure=9, event_mean='yes'):
    combinations = list(product(bin_sizes, smoothing_lengths))
    num_plots = len(combinations)

    # Déterminer le nombre de figures nécessaires
    num_figures = math.ceil(num_plots / max_plots_per_figure)

    # Diviser les combinaisons en morceaux basés sur max_plots_per_figure
    chunks = [combinations[i:i + max_plots_per_figure] for i in range(0, num_plots, max_plots_per_figure)]

    for fig_num, chunk in enumerate(chunks):
        num_subplots = len(chunk)
        if graph == 'group':
            num_cols = math.ceil(math.sqrt(num_subplots))
            num_rows = math.ceil(num_subplots / num_cols)
            fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        elif graph == 'single':
            pass  # Nous gérerons les graphiques uniques individuellement

        for idx, (bin_size, smoothing_length) in enumerate(chunk):
            result = results.get((bin_size, smoothing_length))
            bin_edges = bin_edges_dict.get(bin_size)
            bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2

            if graph == 'group':
                ax = fig.add_subplot(num_rows, num_cols, idx + 1, projection='3d')
            elif graph == 'single':
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

            if result is not None and bin_edges is not None:
                if result.shape[1] >= 3:
                    # Code existant pour tracer les points de dispersion
                    if display in ['all', 'projection']:
                        ax.scatter(result[:, 0], result[:, 1], result[:, 2], s=5, alpha=0.1, label='Overall Projection', zorder=1)

                    # Code existant pour tracer les événements
                    if display in ['all', 'events']:
                        pre_stim = 1  # 1 seconde avant t_0
                        post_stim = 2  # 2 secondes après t_0

                        # Tracer les événements
                        pre_mask = np.any([(bin_times >= t_0 - pre_stim) & (bin_times < t_0) for t_0 in event_times], axis=0)
                        during_mask = np.any([(bin_times >= t_0) & (bin_times < t_0 + 1) for t_0 in event_times], axis=0)
                        post_mask = np.any([(bin_times >= t_0 + 1) & (bin_times < t_0 + post_stim) for t_0 in event_times], axis=0)

                        if np.any(pre_mask):
                            ax.scatter(
                                result[pre_mask, 0], result[pre_mask, 1], result[pre_mask, 2],
                                s=10, color='red', alpha=0.5, label='1s Before', zorder=2
                            )
                        if np.any(during_mask):
                            ax.scatter(
                                result[during_mask, 0], result[during_mask, 1], result[during_mask, 2],
                                s=10, color='purple', alpha=0.5, label='1s During', zorder=2
                            )
                        if np.any(post_mask):
                            ax.scatter(
                                result[post_mask, 0], result[post_mask, 1], result[post_mask, 2],
                                s=10, color='green', alpha=0.5, label='1s After', zorder=2
                            )

                        # Calculer et tracer la trajectoire moyenne si event_mean est 'yes'
                        if event_mean == 'yes':
                            mean_trajectory = []
                            mean_times = []
                            total_counts = []

                            # Initialiser les tableaux pour le calcul de la trajectoire moyenne
                            rel_time_bins = np.arange(-pre_stim, post_stim + bin_size, bin_size)
                            mean_trajectory = np.zeros((len(rel_time_bins) - 1, result.shape[1]))
                            count_trajectory = np.zeros(len(rel_time_bins) - 1)

                            for t_0 in event_times:
                                rel_times = bin_times - t_0
                                mask = (rel_times >= -pre_stim) & (rel_times <= post_stim)
                                rel_times_window = rel_times[mask]
                                result_window = result[mask]

                                # Discrétiser les temps relatifs en bins
                                bin_indices = np.digitize(rel_times_window, rel_time_bins) - 1  # Les indices commencent à 0
                                # Accumuler les sommes et les comptes pour la moyenne
                                for i in range(len(rel_time_bins) - 1):
                                    bin_mask = bin_indices == i
                                    if np.any(bin_mask):
                                        mean_trajectory[i] += np.sum(result_window[bin_mask], axis=0)
                                        count_trajectory[i] += np.sum(bin_mask)

                            # Calculer la trajectoire moyenne
                            valid_bins = count_trajectory > 0
                            mean_trajectory[valid_bins] /= count_trajectory[valid_bins, np.newaxis]
                            mean_rel_times = (rel_time_bins[:-1] + rel_time_bins[1:]) / 2

                            # Tracer la trajectoire moyenne
                            if np.any(valid_bins):
                                ax.plot(
                                    mean_trajectory[valid_bins, 0],
                                    mean_trajectory[valid_bins, 1],
                                    mean_trajectory[valid_bins, 2],
                                    color='yellow',
                                    linewidth=2,
                                    # marker='o',
                                    markersize=4,
                                    label='Mean Trajectory',
                                    zorder=5
                                )
                        if graph == 'single':
                            ax.legend()
                else:
                    ax.text(0.5, 0.5, 0.5, 'Not enough components', horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(0.5, 0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')

            # Assurez-vous que les titres et les étiquettes d'axes sont définis pour tous les sous-graphiques
            ax.set_title(f"Bin: {bin_size}s\nSmooth: {smoothing_length}s")
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

            if graph == 'single':
                fig.suptitle(f'{title_prefix} 3D Projection', fontsize=16)
                plt.show()

        if graph == 'group':
            fig.suptitle(f'{title_prefix} 3D Projections - Figure {fig_num + 1} of {num_figures}', fontsize=16)
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
    print("CUDA disponible :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Nom du GPU :", torch.cuda.get_device_name(0))
    else:
        print("Aucun GPU CUDA n'a été détecté.")
    
    pkl_file = 'experiment_data.pkl'
    tdt_file = 'tdt_signals.pkl'

    data = load_data(pkl_file)
    data_dict = data['data']
    tdt_signals = load_data(tdt_file)
    t_0_times = tdt_signals['Event Time']

    display = 'events'  # Choisir entre 'all', 'events', ou 'projection'
    graph = 'group'  # Choose between 'single' or 'group'
    max_plots_per_figure = 9  # Set the maximum number of plots per figure
    event_mean = 'yes'   # Choose between 'yes' or 'no'
    unit_selection = 'unit2' # Choisir entre 'both', 'unit1', ou 'unit2'
    methods_to_run = ['UMAP']  # You can modify this to select one, two, or all methods ('PCA', 'UMAP', 't-SNE').
    
    # Define multiple t-SNE, PCA, and UMAP configurations
    tsne_configs = [
        {'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000, 'early_exaggeration': 12, 'metric': 'euclidean'},
        {'perplexity': 10, 'learning_rate': 500, 'n_iter': 1500, 'early_exaggeration': 15, 'metric': 'cosine'}
    ]

    umap_configs = [
        {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 3},
        {'n_neighbors': 100, 'min_dist': 0.05, 'n_components': 3}
    ]

    pca_configs = [
        {'n_components': 3}
    ]
    
    # Dynamically select configurations based on the methods the user wants to run
    selected_methods = {}
    if 'PCA' in methods_to_run:
        selected_methods['PCA'] = pca_configs
    if 'UMAP' in methods_to_run:
        selected_methods['UMAP'] = umap_configs
    if 't-SNE' in methods_to_run:
        selected_methods['t-SNE'] = tsne_configs


    spike_times_dict = extract_spike_times(data_dict, unit_selection)
    spike_times_list = list(spike_times_dict.values())
    
    if not spike_times_list:
        raise ValueError("No spike times were extracted. Please check your unit selection and data.")

    duration_list = [np.max(np.array(spike_times, dtype=float)) for spike_times in spike_times_list if len(spike_times) > 0]
    if duration_list:
        duration = max(duration_list)
    else:
        raise ValueError("No valid numeric spike times found in the data.")

    bin_sizes = [0.01, 0.015, 0.02,0.025,0.03,0.035,0.04,0.045, 0.05]
    smoothing_lengths = [0.03, 0.05, 0.075 , 0.1]
    # bin_sizes = [0.5]
    # smoothing_lengths = [2]

    # Run selected methods with their configurations
    all_results, bin_edges_dict = run_in_parallel(spike_times_list, duration, bin_sizes, smoothing_lengths, selected_methods)

    # Process and plot the results
    for method in selected_methods:
        for config_index in range(len(selected_methods[method])):
            # Collect all results for this method and configuration
            result_dict = {}
            variance_dict = {}  # For PCA variance explained
            for key, value in all_results.items():
                key_method, key_config_index, bin_size, smoothing_length = key
                if key_method == method and key_config_index == config_index:
                    if method == 'PCA':
                        # value is a tuple: (pca_result, explained_variance)
                        pca_result, explained_variance = value
                        result_dict[(bin_size, smoothing_length)] = pca_result
                        variance_dict[(bin_size, smoothing_length)] = explained_variance
                    else:
                        # value is the result (e.g., UMAP or t-SNE result)
                        result_dict[(bin_size, smoothing_length)] = value

            if result_dict:
                # Plot all the results for this configuration
                plot_grid_results(
                    result_dict,
                    bin_sizes,
                    smoothing_lengths,
                    title_prefix=f'{method} Config {config_index + 1}',
                    event_times=t_0_times,
                    bin_edges_dict=bin_edges_dict,
                    display=display,
                    graph=graph,  # Pass the graph parameter
                    max_plots_per_figure=max_plots_per_figure,  # Pass the max plots per figure
                    event_mean=event_mean  # Control mean trajectory plotting
                )

                # For PCA, also plot the variance explained
                if method == 'PCA':
                    plot_variance_explained(
                        variance_dict,
                        bin_sizes,
                        smoothing_lengths
                    )
            else:
                print(f"No results found for {method} Config {config_index}")