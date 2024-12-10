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

def extract_continuous_data_per_trial(force_times, force_data, event_times, window_start, window_end, common_times):
    
    trial_data_dict = {}
    for idx, t0 in enumerate(event_times):
        # Calculate window start and end times for this trial
        t_start = t0 + window_start
        t_end = t0 + window_end

        # Find indices in force_times within this window
        indices = np.where((force_times >= t_start) & (force_times <= t_end))[0]

        if len(indices) == 0:
            continue

        # Extract data segment for this trial
        times_segment = force_times[indices] - t0  # Shift times relative to t0
        data_segment = force_data[indices]

        # Interpolate onto common_times
        f = interp1d(times_segment, data_segment, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_data = f(common_times)  # Shape: (len(common_times),)

        # Store the interpolated data
        trial_data_dict[idx] = interpolated_data[np.newaxis, :]  # Shape: (1, len(common_times))

    return trial_data_dict


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

# Function to extract projected data per trial with interpolation onto common_times
def extract_projected_data_per_trial(data, event_times, bin_size, window_start, window_end, common_times):
    """
    Extracts the projected data per trial, interpolated onto common_times.

    Parameters:
    - data: numpy array of shape (T, n_components)
    - event_times: list or array of event times
    - bin_size: time resolution
    - window_start: start of the window relative to event (e.g., -1.0)
    - window_end: end of the window relative to event (e.g., 2.0)
    - common_times: array of time points to interpolate onto

    Returns:
    - trial_data_dict: dictionary where keys are trial indices, and values are matrices of n_components x len(common_times)
    """
    trial_data_dict = {}
    for idx, t0 in enumerate(event_times):
        # Shift times relative to t_0
        relative_times = np.arange(0, len(data)) * bin_size - t0

        # Find indices corresponding to the time window
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

        if len(indices) == 0:
            continue

        # Extract data segments for this time window
        segment = data[indices, :]  # shape: (len(indices), n_components)
        times_segment = relative_times[indices]

        # Interpolate onto the common time grid to align results
        interpolated_data = np.zeros((len(common_times), segment.shape[1]))
        for i in range(segment.shape[1]):
            f = interp1d(times_segment, segment[:, i], kind='linear',
                         bounds_error=False, fill_value="extrapolate")
            interpolated_data[:, i] = f(common_times)

        # Store the interpolated data
        trial_data_dict[idx] = interpolated_data.T  # shape: n_components x len(common_times)

    return trial_data_dict

# Function to average across all trials
def average_across_trials(extracted_data):
    extracted_data_array = np.array(list(extracted_data.values()))  # Shape: (n_trials, n_components, n_times)
    average_data = np.mean(extracted_data_array, axis=0)  # Shape: (n_components, n_times)
    return average_data

# Wrapper function for multiprocessing
def process_unit(unit_key, spike_times, bin_size, duration, sigma):
    binned_data, bin_times = bin_spike_times([spike_times], bin_size, duration)
    smoothed_data = smooth_data(binned_data, sigma=sigma)
    return unit_key, smoothed_data

# Align projected data to event times
def align_projected_data(projection_data, event_times, bin_size, window_start, window_end, common_times):
    n_components = projection_data.shape[1]
    aligned_data = {i: [] for i in range(n_components)}
    for t0 in event_times:
        relative_times = np.arange(0, len(projection_data) * bin_size, bin_size) - t0
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]
        if len(indices) == 0:
            continue
        for i in range(n_components):
            segment = projection_data[indices, i]
            interp_data = np.interp(common_times, relative_times[indices], segment)
            aligned_data[i].append(interp_data)
    return aligned_data

def compute_mean_std(data_array, axis, indices_to_average):
    """
    Computes the mean and standard deviation over specified axis.

    Parameters:
    - data_array: numpy array of shape (n_trials, n_components, len(common_times))
    - axis: integer, axis over which to compute mean and std (0 for trials, 1 for components)
    - indices_to_average: list of indices to average over (either trial indices or component indices)

    Returns:
    - mean_data: numpy array of shape determined by the operation
    - std_data: numpy array of shape determined by the operation
    """
    # Select the data along the axis
    if axis == 0:
        # Averaging over trials
        selected_data = data_array[indices_to_average, :, :]  # Shape: (n_selected_trials, n_components, len(common_times))
    elif axis == 1:
        # Averaging over components
        selected_data = data_array[:, indices_to_average, :]  # Shape: (n_trials, n_selected_components, len(common_times))
    else:
        raise ValueError("Axis must be 0 (trials) or 1 (components).")

    # Compute mean and std over the specified axis
    mean_data = np.mean(selected_data, axis=axis)
    std_data = np.std(selected_data, axis=axis)

    return mean_data, std_data

def plot_peth_with_error(mean_data, std_data, common_times, method_name, averaged_over='trials', indices=None):
    """
    Plots the mean ± std over time.

    Parameters:
    - mean_data: numpy array of mean values
    - std_data: numpy array of std values
    - common_times: array of time points
    - method_name: string, name of the dimensionality reduction method
    - averaged_over: string, 'trials' or 'components' to indicate averaging
    - indices: list of indices that were averaged over
    """
    plt.figure(figsize=(10, 7))
    if averaged_over == 'trials':
        n_components = mean_data.shape[0]
        for comp in range(n_components):
            plt.plot(common_times, mean_data[comp, :], label=f'{method_name} Component {comp + 1}')
            plt.fill_between(common_times,
                             mean_data[comp, :] - std_data[comp, :],
                             mean_data[comp, :] + std_data[comp, :],
                             alpha=0.3)
        plt.title(f'PETH Averaged Over Trials ({method_name})')
        plt.ylabel("Component Value")
    elif averaged_over == 'components':
        n_trials = mean_data.shape[0]
        for idx in range(n_trials):
            plt.plot(common_times, mean_data[idx, :], label=f'Trial {indices[idx] + 1}')
            plt.fill_between(common_times,
                             mean_data[idx, :] - std_data[idx, :],
                             mean_data[idx, :] + std_data[idx, :],
                             alpha=0.3)
        plt.title(f'PETH Averaged Over Components ({method_name})')
        plt.ylabel("Mean Component Value")
    else:
        raise ValueError("averaged_over must be 'trials' or 'components'.")
    
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Time (s) relative to event")
    plt.legend()
    plt.grid(True)

    plt.show()

def compute_component_mean_std(trial_data, pcs_to_average):
    """
    Computes the mean and standard deviation over selected components for a single trial.

    Parameters:
    - trial_data: numpy array of shape (n_components, len(common_times))
    - pcs_to_average: list of component indices (zero-based) to average over

    Returns:
    - mean_over_components: array of shape (len(common_times),)
    - std_over_components: array of shape (len(common_times),)
    """
    # Select the components
    selected_data = trial_data[pcs_to_average, :]  # shape: (n_selected_components, len(common_times))
    # Compute mean and std over components (axis=0)
    mean_over_components = np.mean(selected_data, axis=0)  # shape: (len(common_times),)
    std_over_components = np.std(selected_data, axis=0)    # shape: (len(common_times),)
    return mean_over_components, std_over_components


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
    sigma = (smoothing_length / bin_size) / 2

    # Use multiprocessing to process each unit in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_unit, [(unit_key, spike_times, bin_size, duration, sigma) for unit_key, spike_times in spike_times_dict.items()])

    smoothed_data_dict = {unit_key: smoothed_data for unit_key, smoothed_data in results}

    all_smoothed_data = np.vstack([data for data in smoothed_data_dict.values()])
    all_smoothed_data_T = all_smoothed_data.T  # Shape: (T, n_neurons)

    # Variables for PETH visualization
    window_start = -1.0
    window_end = 2.0
    bin_size = 0.005

    # Define common_times using the same window and bin_size
    common_times = np.arange(window_start, window_end + bin_size, bin_size)

    # Variables to handle saving
    save_data = False
    save_filename = 'projected_data_test.pkl'

    # Variable for PETH visualization
    PETH = True
    # Specify the maximum number of PCs to plot
    max_pc = 3

    # Data containers for results and averages
    results = {}
    averages = {}
    saved_data = {}


    # Parameters to specify
    average_over = 'trials'  # 'trials' or 'components'
    # For averaging over trials
    pcs_to_use = [0, 1, 2]  # Zero-based indices of PCs to use
    trials_to_average = 'all'  # 'all' or list of trial indices

    # For averaging over components
    trials_to_use = [0]  # Zero-based indices of trials to use
    pcs_to_average = 'all'  # 'all' or list of component indices ([0, 1, 2])
    
    # Trial selection variable for visualization
    trial_selection = 'all'  # Can be 'all', an integer, or a list of integers or None
    selected_methods = ['PCA']  # Use 'PCA', 'UMAP', 't-SNE', or 'all' to select methods

    show_average = False  # Set to False if you do not want to see the average projections
    projection_dim = 1  # Set to 1, 2, or 3 for 1D, 2D, or 3D projections
    
    # Apply PCA if selected
    if 'PCA' in selected_methods or selected_methods == 'all':
        try:
            pca_result, explained_variance, pca_components = apply_pca_torch(all_smoothed_data_T, return_components=True)
            results['PCA'] = pca_result

            # Extract projected data per trial with interpolation
            pca_trial_data = extract_projected_data_per_trial(
                pca_result, t_0_times, bin_size, window_start, window_end, common_times
            )

            # Store the extracted data in the specified format
            saved_data['PCA'] = pca_trial_data

            # Average over trials for visualization
            averages['PCA'] = average_across_trials(pca_trial_data)

        except Exception as e:
            print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")

    # Apply UMAP if selected
    if 'UMAP' in selected_methods or selected_methods == 'all':
        try:
            umap_result = apply_umap(all_smoothed_data_T)
            results['UMAP'] = umap_result

            # Extract projected data per trial with interpolation
            umap_trial_data = extract_projected_data_per_trial(
                umap_result, t_0_times, bin_size, window_start, window_end, common_times
            )

            # Store the extracted data
            saved_data['UMAP'] = umap_trial_data

            # Average over trials for visualization
            averages['UMAP'] = average_across_trials(umap_trial_data)

        except Exception as e:
            print(f"UMAP failed: {e}")

    # Apply t-SNE if selected
    if 't-SNE' in selected_methods or selected_methods == 'all':
        try:
            tsne_result = apply_tsne(all_smoothed_data_T)
            results['t-SNE'] = tsne_result

            # Extract projected data per trial with interpolation
            tsne_trial_data = extract_projected_data_per_trial(
                tsne_result, t_0_times, bin_size, window_start, window_end, common_times
            )

            # Store the extracted data
            saved_data['t-SNE'] = tsne_trial_data

            # Average over trials for visualization
            averages['t-SNE'] = average_across_trials(tsne_trial_data)

        except Exception as e:
            print(f"t-SNE failed: {e}")
    try:
        # Extract 'Force' data
        force_fs = 1017.3
        force_data = tdt_signals['Levier']  # Shape: (N,)

        # Create time axis for 'Force' data
        force_times = np.arange(len(force_data)) / force_fs

        # Extract per-trial 'Force' data
        force_trial_data = extract_continuous_data_per_trial(
            force_times, force_data, t_0_times, window_start, window_end, common_times
        )

        # Save 'Force' data into 'saved_data'
        saved_data['Force'] = force_trial_data

        # Optionally, compute average over trials for visualization
        averages['Force'] = average_across_trials(force_trial_data)

    except Exception as e:
        print(f"Processing Force data failed: {e}")
    # Save the projected data if required
    if save_data:
        with open(save_filename, 'wb') as f:
            pickle.dump(saved_data, f)

    if PETH:
        

        for method in selected_methods:
            if method not in saved_data:
                print(f"Skipping PETH for {method} as the method did not run successfully.")
                continue

            trial_data_dict = saved_data[method]
            # Convert to array: Shape (n_trials, n_components, len(common_times))
            data_array = np.array([trial_data_dict[idx] for idx in sorted(trial_data_dict.keys())])
            n_trials, n_components, n_times = data_array.shape

            if average_over == 'trials':
                # Average over trials for selected PCs
                if pcs_to_use == 'all':
                    pcs_to_use_indices = list(range(n_components))
                else:
                    pcs_to_use_indices = pcs_to_use

                if trials_to_average == 'all':
                    trials_to_average_indices = list(range(n_trials))
                else:
                    trials_to_average_indices = trials_to_average

                # Select the data
                data_selected = data_array[trials_to_average_indices, :, :]  # Shape: (n_selected_trials, n_components, n_times)
                # Keep only selected PCs
                data_selected = data_selected[:, pcs_to_use_indices, :]  # Shape: (n_selected_trials, n_selected_PCs, n_times)

                # Compute mean and std over trials (axis=0)
                mean_data = np.mean(data_selected, axis=0)  # Shape: (n_selected_PCs, n_times)
                std_data = np.std(data_selected, axis=0)

                # Plot
                plot_peth_with_error(mean_data, std_data, common_times, method, averaged_over='trials')

            elif average_over == 'components':
                # Average over components for selected trials
                if pcs_to_average == 'all':
                    pcs_to_average_indices = list(range(n_components))
                else:
                    pcs_to_average_indices = pcs_to_average

                if trials_to_use == 'all':
                    trials_to_use_indices = list(range(n_trials))
                else:
                    trials_to_use_indices = trials_to_use

                # Select the data
                data_selected = data_array[trials_to_use_indices, :, :]  # Shape: (n_selected_trials, n_components, n_times)
                # Keep only selected PCs
                data_selected = data_selected[:, pcs_to_average_indices, :]  # Shape: (n_selected_trials, n_selected_PCs, n_times)

                # Compute mean and std over components (axis=1)
                mean_data = np.mean(data_selected, axis=1)  # Shape: (n_selected_trials, n_times)
                std_data = np.std(data_selected, axis=1)

                # Plot
                plot_peth_with_error(mean_data, std_data, common_times, method, averaged_over='components', indices=trials_to_use_indices)

            else:
                print("Invalid option for average_over. Choose 'trials' or 'components'.")

    
    # Visualize the average for each selected method if show_average is True
    if show_average and trial_selection is not None:
        # Create a normalization object for the color map from window_start to window_end
        norm = plt.Normalize(vmin=window_start, vmax=window_end)
        colors = plt.cm.hot(norm(common_times))

        for method_name in selected_methods:
            if method_name in averages and averages[method_name] is not None:
                average_data = averages[method_name]  # Shape: (n_components, len(common_times))

                # Ensure that common_times and average_data have matching lengths
                if len(common_times) != average_data.shape[1]:
                    print(f"Length mismatch: common_times has length {len(common_times)}, "
                          f"but average_data has length {average_data.shape[1]}")
                    continue  # Skip plotting if lengths don't match

                fig = plt.figure(figsize=(10, 7))

                # Check if enough components are available for the desired projection_dim
                n_components = average_data.shape[0]
                if n_components < projection_dim:
                    print(f"Not enough components to plot {projection_dim}D projection for {method_name}. "
                          f"Available components: {n_components}")
                    continue

                # Plot based on projection_dim
                if projection_dim == 1:
                    ax = fig.add_subplot(111)
                    for i in range(len(common_times) - 1):
                        ax.plot(common_times[i:i+2], average_data[0, i:i+2], color=colors[i])
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{method_name}1')
                    ax.set_title(f'Average 1D Projection ({method_name})')

                elif projection_dim == 2:
                    ax = fig.add_subplot(111)
                    for i in range(len(common_times) - 1):
                        ax.plot(average_data[0, i:i+2], average_data[1, i:i+2], color=colors[i])
                    ax.set_xlabel(f'{method_name}1')
                    ax.set_ylabel(f'{method_name}2')
                    ax.set_title(f'Average 2D Projection ({method_name})')

                elif projection_dim == 3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(len(common_times) - 1):
                        ax.plot(
                            average_data[0, i:i+2],
                            average_data[1, i:i+2],
                            average_data[2, i:i+2],
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
                            ax.scatter(common_times[idx_t], average_data[0, idx_t], color=color, s=50, marker='o')
                        elif projection_dim == 2:
                            ax.scatter(
                                average_data[0, idx_t],
                                average_data[1, idx_t],
                                color=color,
                                s=50,
                                marker='o'
                            )
                        elif projection_dim == 3:
                            ax.scatter(
                                average_data[0, idx_t],
                                average_data[1, idx_t],
                                average_data[2, idx_t],
                                color=color,
                                s=50,
                                marker='o'
                            )

                # Add color bar with range window_start to window_end
                sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='Time (s)')

                plt.show()
