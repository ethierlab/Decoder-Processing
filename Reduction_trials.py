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
from matplotlib import cm
from matplotlib.colors import Normalize

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
            spike_counts[i, :] = counts.astype(float)
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
def project_and_visualize(data, method_name, event_times, bin_size,
                          window_start=-1.0, window_end=2.0,
                          n_components=3, trial_selection='all',
                          average_over_trials='all', projection_dim=3):

    # Define a common time grid
    common_times = np.arange(window_start, window_end + bin_size, bin_size)

    # Initialize a list to store the extracted data for each event for averaging
    extracted_all_events = []

    # --- Process trials for averaging ---
    if average_over_trials is None:
        print("No trials selected for averaging.")
    else:
        # Determine trials to process for averaging
        if average_over_trials == 'all':
            trials_for_average = range(len(event_times))
        elif isinstance(average_over_trials, int):
            trials_for_average = [average_over_trials]
        elif isinstance(average_over_trials, (list, tuple, np.ndarray)):
            trials_for_average = average_over_trials
        else:
            raise ValueError("Invalid average_over_trials. Use 'all', an integer, or a list/tuple of integers.")

        for idx in trials_for_average:
            if idx >= len(event_times):
                print(f"Trial index {idx} is out of range for averaging. Skipping.")
                continue

            t0 = event_times[idx]

            # Shift times relative to t_0
            relative_times = np.arange(0, len(data)) * bin_size - t0

            # Find indices corresponding to the time window
            indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

            if len(indices) == 0:
                continue

            # Extract data segments for this time window
            segment = data[indices, :n_components]
            times_segment = relative_times[indices]

            # Interpolate onto the common time grid to align results
            interpolated_data = np.zeros((len(common_times), segment.shape[1]))
            for i in range(segment.shape[1]):
                f = interp1d(times_segment, segment[:, i], kind='linear',
                             bounds_error=False, fill_value="extrapolate")
                interpolated_data[:, i] = f(common_times)

            extracted_all_events.append(interpolated_data)

    # --- Process trials for plotting ---
    if trial_selection is None:
        print("No trials selected for plotting. Skipping individual trial plots.")
    else:
        # Determine which trials to visualize
        if trial_selection == 'all':
            selected_trials = range(len(event_times))
        elif isinstance(trial_selection, int):
            selected_trials = [trial_selection]
        elif isinstance(trial_selection, (list, tuple, np.ndarray)):
            selected_trials = trial_selection
        else:
            raise ValueError("Invalid trial_selection. Use 'all', an integer, a list/tuple of integers, or None.")

        for idx in selected_trials:
            if idx >= len(event_times):
                print(f"Trial index {idx} is out of range for plotting. Skipping.")
                continue

            t0 = event_times[idx]

            # Shift times relative to t_0
            relative_times = np.arange(0, len(data)) * bin_size - t0

            # Find indices corresponding to the time window
            indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

            if len(indices) == 0:
                continue

            # Extract data segments for this time window
            segment = data[indices, :n_components]
            times_segment = relative_times[indices]

            # Interpolate onto the common time grid to align results
            interpolated_data = np.zeros((len(common_times), segment.shape[1]))
            for i in range(segment.shape[1]):
                f = interp1d(times_segment, segment[:, i], kind='linear',
                             bounds_error=False, fill_value="extrapolate")
                interpolated_data[:, i] = f(common_times)

            # Visualize the projection with color progression
            fig = plt.figure(figsize=(10, 7))

            # Generate color map based on common_times ranging from window_start to window_end
            norm = Normalize(vmin=window_start, vmax=window_end)
            colors = cm.hot(norm(common_times))

            if projection_dim == 1:
                ax = fig.add_subplot(111)
                for i in range(len(common_times) - 1):
                    ax.plot(common_times[i:i+2], interpolated_data[i:i+2, 0], color=colors[i])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'{method_name}1')
                ax.set_title(f'1D Projection for Trial at {t0:.2f}s ({method_name})')
            elif projection_dim == 2:
                ax = fig.add_subplot(111)
                for i in range(len(common_times) - 1):
                    ax.plot(interpolated_data[i:i+2, 0],
                            interpolated_data[i:i+2, 1], color=colors[i])
                ax.set_xlabel(f'{method_name}1')
                ax.set_ylabel(f'{method_name}2')
                ax.set_title(f'2D Projection for Trial at {t0:.2f}s ({method_name})')
            elif projection_dim == 3:
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(common_times) - 1):
                    ax.plot(interpolated_data[i:i+2, 0],
                            interpolated_data[i:i+2, 1],
                            interpolated_data[i:i+2, 2], color=colors[i])
                ax.set_xlabel(f'{method_name}1')
                ax.set_ylabel(f'{method_name}2')
                ax.set_zlabel(f'{method_name}3')
                ax.set_title(f'3D Projection for Trial at {t0:.2f}s ({method_name})')
            else:
                raise ValueError("Invalid projection_dim. Choose 1, 2, or 3.")

            # Time markers to indicate specific times on the plot
            time_markers = {
                window_start: 'red',
                0.0: 'green',
                window_end: 'black'
            }

            # Plot markers with a small tolerance to match closest time in common_times
            tolerance = 1e-6
            for t_mark, color in time_markers.items():
                idx_t = np.where(np.abs(common_times - t_mark) < tolerance)[0]
                if idx_t.size > 0:
                    idx_t = idx_t[0]
                    if projection_dim == 1:
                        ax.scatter(common_times[idx_t], interpolated_data[idx_t, 0], color=color, s=50, marker='o')
                    elif projection_dim == 2:
                        ax.scatter(interpolated_data[idx_t, 0], interpolated_data[idx_t, 1], color=color, s=50, marker='o')
                    elif projection_dim == 3:
                        ax.scatter(interpolated_data[idx_t, 0], interpolated_data[idx_t, 1],
                                   interpolated_data[idx_t, 2], color=color, s=50, marker='o')

            # Add color bar with range window_start to window_end
            sm = cm.ScalarMappable(cmap="hot", norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Time (s)")

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

    # Trial selection variable for visualization
    trial_selection = 1  # Can be 'all', an integer, or a list of integers or None
    average_over_trials = 'all' # Can be 'all', an integer, or a list of integers or None
    selected_methods = ['PCA']  # Use 'PCA', 'UMAP', 't-SNE', or 'all' to select methods
    
    show_average = True  # Set to False if you do not want to see the average projections
    projection_dim = 2 # Set to 1, 2, or 3 for 1D, 2D, or 3D projections

    # Define window_start and window_end
    window_start = -1.0
    window_end = 2.0
    # Define common_times using the same window and bin_size
    common_times = np.arange(window_start, window_end + bin_size, bin_size)
    
    # Data containers for results and averages
    results = {}
    averages = {}
    
    
    # Apply PCA if selected
    if 'PCA' in selected_methods or selected_methods == 'all':
        try:
            pca_result, explained_variance, pca_components = apply_pca_torch(all_smoothed_data_T, return_components=True)
            results['PCA'] = pca_result
            # Visualize variance explained by PCA if needed
            # plot_variance_explained_single(explained_variance)
            pca_extracted = project_and_visualize(pca_result, 'PCA', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection, average_over_trials=average_over_trials, projection_dim=projection_dim)
            averages['PCA'] = average_across_trials(pca_extracted)
        except Exception as e:
            print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")

    # Apply UMAP if selected
    if 'UMAP' in selected_methods or selected_methods == 'all':
        try:
            umap_result = apply_umap(all_smoothed_data_T)
            results['UMAP'] = umap_result
            umap_extracted = project_and_visualize(umap_result, 'UMAP', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection, average_over_trials=average_over_trials, projection_dim=projection_dim)
            averages['UMAP'] = average_across_trials(umap_extracted)
        except Exception as e:
            print(f"UMAP failed: {e}")

    # Apply t-SNE if selected
    if 't-SNE' in selected_methods or selected_methods == 'all':
        try:
            tsne_result = apply_tsne(all_smoothed_data_T)
            results['t-SNE'] = tsne_result
            tsne_extracted = project_and_visualize(tsne_result, 't-SNE', t_0_times, bin_size, window_start=-1.0, window_end=2.0, trial_selection=trial_selection, average_over_trials=average_over_trials, projection_dim=projection_dim)
            averages['t-SNE'] = average_across_trials(tsne_extracted)
        except Exception as e:
            print(f"t-SNE failed: {e}")

    # Visualize the average for each selected method if show_average is True
    if show_average and average_over_trials is not None:
    # Create a normalization object for the color map from -1 to 2
        norm = plt.Normalize(vmin=window_start, vmax=window_end)
        colors = plt.cm.hot(norm(common_times))

        for method_name in selected_methods:
            if method_name in averages and averages[method_name] is not None:
                average_data = averages[method_name]

                # Ensure that common_times and average_data have matching lengths
                if len(common_times) != average_data.shape[0]:
                    print(f"Length mismatch: common_times has length {len(common_times)}, "
                        f"but average_data has length {average_data.shape[0]}")
                    continue  # Skip plotting if lengths don't match

                fig = plt.figure(figsize=(10, 7))

                # Plot based on projection_dim
                if projection_dim == 1:
                    ax = fig.add_subplot(111)
                    for i in range(len(common_times) - 1):
                        ax.plot(common_times[i:i+2], average_data[i:i+2, 0], color=colors[i])
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{method_name}1')
                    ax.set_title(f'Average 1D Projection ({method_name})')

                elif projection_dim == 2:
                    ax = fig.add_subplot(111)
                    for i in range(len(common_times) - 1):
                        ax.plot(average_data[i:i+2, 0], average_data[i:i+2, 1], color=colors[i])
                    ax.set_xlabel(f'{method_name}1')
                    ax.set_ylabel(f'{method_name}2')
                    ax.set_title(f'Average 2D Projection ({method_name})')

                elif projection_dim == 3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(len(common_times) - 1):
                        ax.plot(
                            average_data[i:i+2, 0],
                            average_data[i:i+2, 1],
                            average_data[i:i+2, 2],
                            color=colors[i]
                        )
                    ax.set_xlabel(f'{method_name}1')
                    ax.set_ylabel(f'{method_name}2')
                    ax.set_zlabel(f'{method_name}3')
                    ax.set_title(f'Average 3D Projection ({method_name})')

                else:
                    raise ValueError("Invalid projection_dim. Choose 1, 2, or 3.")
                time_markers = {
                    window_start: 'red',
                    0.0: 'green',
                    window_end: 'black'
                }

                # Plot markers with a small tolerance to match closest time in common_times
                tolerance = 1e-6
                for t_mark, color in time_markers.items():
                    idx_t = np.where(np.abs(common_times - t_mark) < tolerance)[0]
                    if idx_t.size > 0:
                        idx_t = idx_t[0]
                        if projection_dim == 1:
                            ax.scatter(common_times[idx_t], average_data[idx_t, 0], color=color, s=50, marker='o')
                        elif projection_dim == 2:
                            ax.scatter(
                                average_data[idx_t, 0],
                                average_data[idx_t, 1],
                                color=color,
                                s=50,
                                marker='o'
                            )
                        elif projection_dim == 3:
                            ax.scatter(
                                average_data[idx_t, 0],
                                average_data[idx_t, 1],
                                average_data[idx_t, 2],
                                color=color,
                                s=50,
                                marker='o'
                            )

                # Add color bar with range -1 to 2
                sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='Time (s)')

                plt.show()
