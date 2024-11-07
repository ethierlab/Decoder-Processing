import numpy as np
import pickle
import torch
import multiprocessing
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy import stats

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
            spike_counts[i, :] = counts.astype(float)
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

def process_combination(bin_size, smoothing_length, spike_times_list, bin_edges, results_dict, method_configs, dimension):
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
                    pca_result, explained_variance = apply_pca(smoothed_data_T, n_components=dimension)
                    results_dict[result_key] = (pca_result[:, :dimension], explained_variance)
                    print(f"[Process PID {multiprocessing.current_process().pid}] Completed PCA config {config_index}")
                except Exception as e:
                    print(f"PCA failed for config {config_index}, bin_size {bin_size}s: {e}")
            elif method == 'UMAP':
                try:
                    n_neighbors = min(config['n_neighbors'], smoothed_data_T.shape[0] - 1)
                    if n_neighbors < 2:
                        print(f"n_neighbors too small for UMAP config {config_index}, bin_size {bin_size}s")
                        continue

                    umap_result = apply_umap(smoothed_data_T, n_neighbors=config['n_neighbors'], min_dist=config['min_dist'], n_components=dimension)

                    results_dict[result_key] = umap_result
                    print(f"[Process PID {multiprocessing.current_process().pid}] Completed UMAP config {config_index}")


                except Exception as e:
                    print(f"UMAP failed for config {config_index}, bin_size {bin_size}s: {e}")

            elif method == 't-SNE':
                try:
                    tsne_result = apply_tsne(
                        smoothed_data_T,
                        n_components=dimension,
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
def run_in_parallel(spike_times_list, duration, bin_sizes, smoothing_lengths, method_configs, dimension):
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    bin_edges_dict = {}
    total_combinations = len(bin_sizes) * len(smoothing_lengths)
    
    for bin_size in bin_sizes:
        bin_edges = np.arange(0, duration + bin_size, bin_size)
        bin_edges_dict[bin_size] = bin_edges

    processes = []
    for bin_size, smoothing_length in product(bin_sizes, smoothing_lengths):
        args = (bin_size, smoothing_length, spike_times_list, bin_edges_dict[bin_size], results_dict, method_configs, dimension)
        process = multiprocessing.Process(
            target=process_combination,
            args=args
        )
        processes.append(process)
        process.start()

    # Use tqdm to display progress as processes complete
    for process in processes:
        process.join()

    # Convert manager dict to a regular dict
    return dict(results_dict), bin_edges_dict


def plot_density(result, x_pc, y_pc, bin_times, event_times, display, cmap='Blues', label='Overall Projection', alpha=0.8, event_mean='yes'):

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    color = cmap_obj(0.6)

    # Validate data before plotting
    if len(result) == 0 or np.all(np.isnan(result[:, x_pc])) or np.all(np.isnan(result[:, y_pc])):
        print("Warning: No valid data to plot.")
        return

    # Set up the JointGrid
    g = sns.JointGrid(x=[], y=[], height=8, ratio=5, marginal_ticks=True)

    # # Number of levels
    # num_levels = 8

    # # Create a colormap with the same color repeated
    # same_color_cmap = ListedColormap([color] * num_levels)
    
    # z_scores = np.abs(stats.zscore(result[:, [x_pc, y_pc]]))
    # outlier_mask = (z_scores > 3).any(axis=1)
    # if np.any(outlier_mask):
    #     print(f"Removing {np.sum(outlier_mask)} outliers")
    #     result = result[~outlier_mask]

    # Plot densities based on the selected display
    if display == 'all' or display == 'projection':
        # Overall density
        sns.kdeplot(
            x=result[:, x_pc],
            y=result[:, y_pc],
            cmap=cmap,
            alpha=alpha,
            linewidths=1.5,
            # levels=num_levels,
            bw_adjust=1.5,
            ax=g.ax_joint
        )
        sns.kdeplot(x=result[:, x_pc], ax=g.ax_marg_x, color=color, alpha=0.6, linewidth=1.5)
        sns.kdeplot(y=result[:, y_pc], ax=g.ax_marg_y, color=color, alpha=0.6, linewidth=1.5)

    if display == 'all' or display == 'events':
        # Define colors for each period
        period_colors = {'pre': 'green', 'during': 'red', 'post': 'orange'}

        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0

        # Create masks for event periods
        pre_mask = np.any([(bin_times >= t_0 - pre_stim) & (bin_times < t_0) for t_0 in event_times], axis=0)
        during_mask = np.any([(bin_times >= t_0) & (bin_times < t_0 + 1) for t_0 in event_times], axis=0)
        post_mask = np.any([(bin_times >= t_0 + 1) & (bin_times < t_0 + post_stim) for t_0 in event_times], axis=0)

        # Plot densities for each event period with uniform lines
        for period, mask, color_period in zip(
            ['1s Before', '1s During', '1s After'],
            [pre_mask, during_mask, post_mask],
            [period_colors['pre'], period_colors['during'], period_colors['post']]
        ):
            if np.any(mask):
                # Create a colormap with the same color
                # same_color_cmap = ListedColormap([color_period] * num_levels)
                sns.kdeplot(
                    x=result[mask, x_pc],
                    y=result[mask, y_pc],
                    cmap=cmap,
                    alpha=alpha,
                    linewidths=1.5,
                    # levels=num_levels,
                    bw_adjust=1.5,
                    ax=g.ax_joint,
                    label=period
                )
                sns.kdeplot(x=result[mask, x_pc], ax=g.ax_marg_x, color=color_period, alpha=0.5, linewidth=1.5)
                sns.kdeplot(y=result[mask, y_pc], ax=g.ax_marg_y, color=color_period, alpha=0.5, linewidth=1.5)

        # Add legend
        handles, labels = g.ax_joint.get_legend_handles_labels()
        if handles:
            g.ax_joint.legend(handles=handles, labels=labels)

    # Add marginal KDEs based on the selected display
    if display == 'all' or display == 'projection':
        sns.kdeplot(x=result[:, x_pc], ax=g.ax_marg_x, color=color, alpha=0.6, linewidth=1.5)
        sns.kdeplot(y=result[:, y_pc], ax=g.ax_marg_y, color=color, alpha=0.6, linewidth=1.5)
    elif display == 'events':
        # Marginal KDEs for specific event segments
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0

        pre_mask = np.any([(bin_times >= t_0 - pre_stim) & (bin_times < t_0) for t_0 in event_times], axis=0)
        during_mask = np.any([(bin_times >= t_0) & (bin_times < t_0 + 1) for t_0 in event_times], axis=0)
        post_mask = np.any([(bin_times >= t_0 + 1) & (bin_times < t_0 + post_stim) for t_0 in event_times], axis=0)

        if np.any(pre_mask):
            sns.kdeplot(x=result[pre_mask, x_pc], ax=g.ax_marg_x, color='green', alpha=0.5, linewidth=1.5)
            sns.kdeplot(y=result[pre_mask, y_pc], ax=g.ax_marg_y, color='green', alpha=0.5, linewidth=1.5)

        if np.any(during_mask):
            sns.kdeplot(x=result[during_mask, x_pc], ax=g.ax_marg_x, color='red', alpha=0.5, linewidth=1.5)
            sns.kdeplot(y=result[during_mask, y_pc], ax=g.ax_marg_y, color='red', alpha=0.5, linewidth=1.5)

        if np.any(post_mask):
            sns.kdeplot(x=result[post_mask, x_pc], ax=g.ax_marg_x, color='orange', alpha=0.5, linewidth=1.5)
            sns.kdeplot(y=result[post_mask, y_pc], ax=g.ax_marg_y, color='orange', alpha=0.5, linewidth=1.5)

    # Plot mean trajectory if event_mean == 'yes'
    if event_mean == 'yes':
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0
        rel_time_bins = np.arange(-pre_stim, post_stim + bin_times[1] - bin_times[0], bin_times[1] - bin_times[0])
        mean_trajectory = np.zeros((len(rel_time_bins) - 1, 2))
        count_trajectory = np.zeros(len(rel_time_bins) - 1)

        # Loop through each event time to compute the mean trajectory
        for t_0 in event_times:
            rel_times = bin_times - t_0
            mask = (rel_times >= -pre_stim) & (rel_times <= post_stim)
            rel_times_window = rel_times[mask]
            result_window = result[mask]
            bin_indices = np.digitize(rel_times_window, rel_time_bins) - 1
            for i in range(len(rel_time_bins) - 1):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    mean_trajectory[i] += np.sum(result_window[bin_mask][:, [x_pc, y_pc]], axis=0)
                    count_trajectory[i] += np.sum(bin_mask)

        # Calculate the average for valid bins
        valid_bins = count_trajectory > 0
        mean_trajectory[valid_bins] /= count_trajectory[valid_bins, np.newaxis]

        # Plot the mean trajectory with 'hot' color bar to represent time evolution
        if np.any(valid_bins):
            norm = plt.Normalize(-pre_stim, post_stim)
            colors = matplotlib.colormaps.get_cmap('hot')(norm(rel_time_bins[:-1][valid_bins]))

            for i in range(len(mean_trajectory) - 1):
                if valid_bins[i] and valid_bins[i + 1]:
                    g.ax_joint.plot(
                        mean_trajectory[i:i + 2, 0],  # x coordinates
                        mean_trajectory[i:i + 2, 1],  # y coordinates
                        color=colors[i], linewidth=2, zorder=5
                    )

    # Set axis labels for main density plot
    g.ax_joint.set_xlabel(f'PC{x_pc + 1}', fontsize=10)
    g.ax_joint.set_ylabel(f'PC{y_pc + 1}', fontsize=10)

    # Set titles for the plots
    g.ax_joint.set_title(f"Density Plot with Marginals\nPC{x_pc + 1} vs PC{y_pc + 1}", fontsize=12)




# Function to handle scatter plotting in 2D
def plot_2d_scatter(ax, result, x_pc, y_pc, bin_times, event_times, display, event_mean='yes'):
    # Plot the individual dots if specified by the display parameter
    if display in ['all', 'projection']:
        ax.scatter(result[:, x_pc], result[:, y_pc], s=5, alpha=0.3, label='Overall Projection', zorder=1)

    # Plot event-specific colors if requested
    if display in ['all', 'events']:
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0

        pre_mask = np.any([(bin_times >= t_0 - pre_stim) & (bin_times < t_0) for t_0 in event_times], axis=0)
        if np.any(pre_mask):
            ax.scatter(result[pre_mask, x_pc], result[pre_mask, y_pc], s=10, color='green', alpha=0.5, label='1s Before', zorder=2)

        during_mask = np.any([(bin_times >= t_0) & (bin_times < t_0 + 1) for t_0 in event_times], axis=0)
        if np.any(during_mask):
            ax.scatter(result[during_mask, x_pc], result[during_mask, y_pc], s=10, color='red', alpha=0.5, label='1s During', zorder=2)

        post_mask = np.any([(bin_times >= t_0 + 1) & (bin_times < t_0 + post_stim) for t_0 in event_times], axis=0)
        if np.any(post_mask):
            ax.scatter(result[post_mask, x_pc], result[post_mask, y_pc], s=10, color='orange', alpha=0.5, label='1s After', zorder=2)

    # Plot mean trajectory if event_mean == 'yes'
    if event_mean == 'yes':
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0
        rel_time_bins = np.arange(-pre_stim, post_stim + bin_times[1] - bin_times[0], bin_times[1] - bin_times[0])
        mean_trajectory = np.zeros((len(rel_time_bins) - 1, 2))
        count_trajectory = np.zeros(len(rel_time_bins) - 1)

        # Loop through each event time to compute the mean trajectory
        for t_0 in event_times:
            rel_times = bin_times - t_0
            mask = (rel_times >= -pre_stim) & (rel_times <= post_stim)
            rel_times_window = rel_times[mask]
            result_window = result[mask]
            bin_indices = np.digitize(rel_times_window, rel_time_bins) - 1
            for i in range(len(rel_time_bins) - 1):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    mean_trajectory[i] += np.sum(result_window[bin_mask][:, [x_pc, y_pc]], axis=0)
                    count_trajectory[i] += np.sum(bin_mask)

        # Calculate the average for valid bins
        valid_bins = count_trajectory > 0
        mean_trajectory[valid_bins] /= count_trajectory[valid_bins, np.newaxis]

        # Plot the mean trajectory
        if np.any(valid_bins):
            norm = plt.Normalize(-pre_stim, post_stim)
            colors = matplotlib.colormaps.get_cmap('hot')(norm(rel_time_bins[:-1][valid_bins]))

            for i in range(len(mean_trajectory) - 1):
                if valid_bins[i] and valid_bins[i + 1]:
                    ax.plot(
                        mean_trajectory[i:i + 2, 0],  # x coordinates
                        mean_trajectory[i:i + 2, 1],  # y coordinates
                        color=colors[i], linewidth=2, zorder=5
                    )

# Function to handle scatter plotting in 3D
def plot_3d_scatter(ax, result, bin_times, event_times, display, event_mean='yes'):
    # Plot the individual dots if specified by the display parameter
    if display in ['all', 'projection']:
        ax.scatter(result[:, 0], result[:, 1], result[:, 2], s=5, alpha=0.3, label='Overall Projection', zorder=1)

    # Plot event-specific colors if requested
    if display in ['all', 'events']:
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0

        pre_mask = np.any([(bin_times >= t_0 - pre_stim) & (bin_times < t_0) for t_0 in event_times], axis=0)
        if np.any(pre_mask):
            ax.scatter(result[pre_mask, 0], result[pre_mask, 1], result[pre_mask, 2], s=10, color='green', alpha=0.5, label='1s Before', zorder=2)

        during_mask = np.any([(bin_times >= t_0) & (bin_times < t_0 + 1) for t_0 in event_times], axis=0)
        if np.any(during_mask):
            ax.scatter(result[during_mask, 0], result[during_mask, 1], result[during_mask, 2], s=10, color='red', alpha=0.5, label='1s During', zorder=2)

        post_mask = np.any([(bin_times >= t_0 + 1) & (bin_times < t_0 + post_stim) for t_0 in event_times], axis=0)
        if np.any(post_mask):
            ax.scatter(result[post_mask, 0], result[post_mask, 1], result[post_mask, 2], s=10, color='orange', alpha=0.5, label='1s After', zorder=2)

    # Plot mean trajectory if event_mean == 'yes'
    if event_mean == 'yes':
        pre_stim = 1  # 1 second before t_0
        post_stim = 2  # 2 seconds after t_0
        rel_time_bins = np.arange(-pre_stim, post_stim + bin_times[1] - bin_times[0], bin_times[1] - bin_times[0])
        mean_trajectory = np.zeros((len(rel_time_bins) - 1, 3))
        count_trajectory = np.zeros(len(rel_time_bins) - 1)

        # Loop through each event time to compute the mean trajectory
        for t_0 in event_times:
            rel_times = bin_times - t_0
            mask = (rel_times >= -pre_stim) & (rel_times <= post_stim)
            rel_times_window = rel_times[mask]
            result_window = result[mask]
            bin_indices = np.digitize(rel_times_window, rel_time_bins) - 1
            for i in range(len(rel_time_bins) - 1):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    mean_trajectory[i] += np.sum(result_window[bin_mask], axis=0)
                    count_trajectory[i] += np.sum(bin_mask)

        # Calculate the average for valid bins
        valid_bins = count_trajectory > 0
        mean_trajectory[valid_bins] /= count_trajectory[valid_bins, np.newaxis]

        # Plot the mean trajectory
        if np.any(valid_bins):
            norm = plt.Normalize(-pre_stim, post_stim)
            colors = matplotlib.colormaps.get_cmap('hot')(norm(rel_time_bins[:-1][valid_bins]))

            for i in range(len(mean_trajectory) - 1):
                if valid_bins[i] and valid_bins[i + 1]:
                    ax.plot(
                        mean_trajectory[i:i + 2, 0],  # x coordinates
                        mean_trajectory[i:i + 2, 1],  # y coordinates
                        zs=mean_trajectory[i:i + 2, 2],  # z coordinates
                        color=colors[i], linewidth=2, zorder=5
                    )


# Main plotting function
def plot_grid_results(results, bin_sizes, smoothing_lengths, title_prefix, event_times, bin_edges_dict, dimension=3, display='all', graph='group', max_plots_per_figure=9, event_mean='yes', plot_type='dots', plot_combinations=False):
    combinations_list = list(product(bin_sizes, smoothing_lengths))
    num_plots = len(combinations_list)

    # Always plot all combinations in density mode, calculate PCA in 3D
    if plot_type == 'density':
        pc_combinations = [(0, 1), (1, 2), (0, 2)]  # Always use 2D for density plots but calculate PCA in 3D
    else:
        # Use standard dimension logic for scatter plots
        if dimension == 2:
            pc_combinations = [(0, 1)]
        elif dimension == 3:
            pc_combinations = [(0, 1), (1, 2), (0, 2)] if plot_combinations else [(0, 1)]

    # Handle density plots independently
    if plot_type == 'density':
        # Loop over each combination of bin_size and smoothing_length
        for idx, (bin_size, smoothing_length) in enumerate(combinations_list):
            result = results.get((bin_size, smoothing_length))
            bin_edges = bin_edges_dict.get(bin_size)
            bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2

            if result is None or bin_edges is None:
                continue

            # Loop over each PC combination
            for x_pc, y_pc in pc_combinations:
                # Create a new figure for each combination in density mode with marginals
                fig, ax_main = plt.subplots(figsize=(10, 8))

                # Call the function to plot the density with marginal KDEs
                plot_density(result, x_pc, y_pc, bin_times, event_times, display, event_mean=event_mean)

                # Display each figure
                plt.tight_layout()
                plt.savefig(f' Density{x_pc+1}vs{y_pc+1}', dpi=700)
                plt.show()

        return  # Exit function after handling density plots since they are handled separately

    # Original logic for dots scatter plots (group/single)
    num_figures = math.ceil((num_plots * len(pc_combinations)) / max_plots_per_figure)
    chunks = [combinations_list[i:i + max_plots_per_figure // len(pc_combinations)] for i in range(0, num_plots, max_plots_per_figure // len(pc_combinations))]

    # Loop over chunks, each chunk corresponds to one figure to display
    for fig_num, chunk in enumerate(chunks):
        num_subplots = len(chunk) * len(pc_combinations)

        # Create a new figure for each chunk of plots
        fig = plt.figure(figsize=(10, 8))

        if graph == 'group':
            num_cols = math.ceil(math.sqrt(num_subplots))
            num_rows = math.ceil(num_subplots / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
            axes = np.array(axes).flatten()
            plt.subplots_adjust(hspace=0.6, wspace=0.5)
        elif graph == 'single':
            # Only create a 3D axis if dimension == 3 and scatter plot
            if dimension == 3 and plot_type == 'dots':
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

        plot_idx = 0
        for idx, (bin_size, smoothing_length) in enumerate(chunk):
            result = results.get((bin_size, smoothing_length))
            bin_edges = bin_edges_dict.get(bin_size)
            bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            if result is None or bin_edges is None:
                continue

            # Loop over each PC combination and assign a separate subplot to each
            for x_pc, y_pc in pc_combinations:
                if graph == 'group':
                    if plot_idx >= len(axes):
                        print(f"Warning: Plot index {plot_idx} exceeds available axes. Skipping remaining plots.")
                        break
                    ax = axes[plot_idx]
                elif graph == 'single':
                    if plot_idx > 0:
                        # Create a new figure for every new combination in single mode
                        fig = plt.figure(figsize=(8, 6))
                        if dimension == 3 and plot_type == 'dots':
                            ax = fig.add_subplot(111, projection='3d')
                        else:
                            ax = fig.add_subplot(111)
            for x_pc, y_pc in pc_combinations:
                # Handle scatter plots
                if plot_type == 'dots':
                    # Plot based on selected dimension
                    if dimension == 2:
                        plot_2d_scatter(ax, result, x_pc, y_pc, bin_times, event_times, display, event_mean=event_mean)
                    elif dimension == 3:
                        # Ensure ax is 3D
                        if not hasattr(ax, 'get_proj'):
                            ax = fig.add_subplot(ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start, projection='3d')

                        plot_3d_scatter(ax, result, bin_times, event_times, display, event_mean=event_mean)

                plot_idx += 1

        # Hide empty subplots in 'group' mode
        if graph == 'group':
            for ax in axes[num_subplots:]:
                ax.set_visible(False)
            # Show each group of plots immediately after generating them
            fig.suptitle(f'{title_prefix} - {dimension}D Projections - Figure {fig_num + 1} of {num_figures}', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            print('test')
            plt.show()

        elif graph == 'single':
            # Show single plot immediately after generating
            fig.suptitle(f'{title_prefix} - {dimension}D Projection', fontsize=16)
            plt.tight_layout()
            print('test 2')
            plt.show()




# Function to plot explained variance
def plot_variance_explained(explained_variance_dict, bin_sizes, smoothing_lengths):
    n_rows = len(bin_sizes)
    n_cols = len(smoothing_lengths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Ensure 'axes' is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([axes]).T

    for i, bin_size in enumerate(bin_sizes):
        for j, smoothing_length in enumerate(smoothing_lengths):
            explained_variance = explained_variance_dict.get((bin_size, smoothing_length))
            ax = axes[i, j]
            if explained_variance is not None:
                components = np.arange(1, len(explained_variance) + 1)
                cumulative_variance = np.cumsum(explained_variance) * 100
                ax.bar(components, explained_variance * 100, alpha=0.7, label='Individual Variance')
                ax.plot(components, cumulative_variance, marker='o', color='red', label='Cumulative Variance')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Variance Explained (%)')
                ax.set_ylim(0, 100)
                ax.set_title(f"Bin: {bin_size}s, Smooth: {smoothing_length}s")
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')

    fig.suptitle('PCA Variance Explained', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig ('PCA Variance Explained', dpi=700)
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

    display = 'projection'  # Choose between 'all', 'events', or 'projection'
    graph = 'single'  # Choose between 'single' or 'group'
    max_plots_per_figure = 9  # Set the maximum number of plots per figure
    event_mean = 'yes'   # Choose between 'yes' or 'no'
    dimension = 3  # Choose between 2 or 3
    unit_selection = 'unit2' # Choose between  'both', 'unit1', or 'unit2'
    # methods_to_run = ['PCA', 't-SNE', 'UMAP'] # You can modify this to select one, two, or all methods ('PCA', 'UMAP', 't-SNE').
    methods_to_run = ['PCA']
    plot_type = 'density'  # Options: 'dots' or 'density'
    # Define multiple t-SNE, PCA, and UMAP configurations
    tsne_configs = [
        {'n_components': 3, 'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000, 'early_exaggeration': 12, 'metric': 'euclidean'},
        {'n_components': 3, 'perplexity': 10, 'learning_rate': 500, 'n_iter': 1500, 'early_exaggeration': 15, 'metric': 'cosine'}
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

    # bin_sizes = [0.01, 0.015, 0.02,0.025,0.03,0.035,0.04,0.045, 0.05]
    # smoothing_lengths = [0.03, 0.05, 0.075 , 0.1]
    bin_sizes = [0.01]
    smoothing_lengths = [0.05]

    # Run selected methods with their configurations
    all_results, bin_edges_dict = run_in_parallel(spike_times_list, duration, bin_sizes, smoothing_lengths, selected_methods,dimension=dimension)

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
                    dimension=dimension,
                    display=display,
                    graph=graph,
                    max_plots_per_figure=max_plots_per_figure,
                    event_mean=event_mean,
                    plot_type=plot_type
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