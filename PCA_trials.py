import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Fonction pour charger un fichier pkl
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Fonction pour extraire les temps de spikes pour chaque subunit
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

# Fonction pour binner les temps de spikes
def bin_spike_times(spike_times_list, bin_size, duration):
    n_neurons = len(spike_times_list)
    n_bins = int(np.ceil(duration / bin_size))
    spike_counts = np.zeros((n_neurons, n_bins))
    bin_edges = np.arange(0, duration + bin_size, bin_size)
    bin_times = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centres des bins
    for i, neuron_spike_times in enumerate(spike_times_list):
        if len(neuron_spike_times) > 0:
            counts, _ = np.histogram(neuron_spike_times, bins=bin_edges)
            spike_counts[i, :] = counts
        else:
            spike_counts[i, :] = 0
    return spike_counts, bin_times

# Fonction pour lisser les données avec un filtre gaussien
def smooth_data(data, sigma=1):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=1)
    return smoothed_data

# Fonction pour appliquer le PCA en utilisant PyTorch
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

# Fonction générique pour extraire et projeter les PC autour de chaque événement
def project_pca_around_events(pca_result, bin_times, event_times, window_start=-1.0, window_end=2.0, n_components=3):
    # Définir une grille de temps commune
    common_times = np.arange(window_start, window_end, bin_times[1] - bin_times[0])

    # Initialiser une liste pour stocker les données extraites pour chaque T_0
    extracted_pca_all_events = []

    for idx, t0 in enumerate(event_times):
        # Décaler les temps par rapport à T_0
        relative_times = bin_times - t0
        
        # Trouver les indices correspondant à la fenêtre temporelle [-1s, +2s]
        indices = np.where((relative_times >= window_start) & (relative_times <= window_end))[0]

        if len(indices) == 0:
            continue

        # Extraire les segments des données PCA pour cette fenêtre
        pca_segment = pca_result[indices, :n_components]
        times_segment = relative_times[indices]

        # Interpoler sur la grille de temps commune pour aligner les résultats
        interpolated_pca = np.zeros((len(common_times), pca_segment.shape[1]))
        for i in range(pca_segment.shape[1]):
            f = interp1d(times_segment, pca_segment[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_pca[:, i] = f(common_times)

        extracted_pca_all_events.append(interpolated_pca)

        # Visualiser la projection des 3 premières composantes principales pour cet événement
        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')
        ax.plot(interpolated_pca[:, 0], interpolated_pca[:, 1], interpolated_pca[:, 2])
        
        # Ajouter des marqueurs aux temps spécifiques (-1s, 0s, +2s)
        time_markers = {
            -1.0: 'red',
            0.0: 'green',
            2.0: 'blue'
        }
        for t_mark, color in time_markers.items():
            idx_t = np.where(np.isclose(common_times, t_mark, atol=1e-6))[0]
            if idx_t.size > 0:
                idx_t = idx_t[0]
                ax.scatter(interpolated_pca[idx_t, 0], interpolated_pca[idx_t, 1], interpolated_pca[idx_t, 2], color=color, s=50, marker='o')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Projection 3 PC for Trial at {t0:.2f}s') 
        plt.show()

    return extracted_pca_all_events

# Fonction pour visualiser la variance expliquée
def plot_variance_explained_single(explained_variance):
    components = np.arange(1, len(explained_variance) + 1)
    cumulative_variance = np.cumsum(explained_variance) * 100
    plt.figure(figsize=(8, 6))
    plt.bar(components, explained_variance * 100, alpha=0.7, label='Variance expliquée par composante')
    plt.plot(components, cumulative_variance, marker='o', color='red', label='Variance cumulative')
    plt.xlabel('Composante principale')
    plt.ylabel('Variance expliquée (%)')
    plt.title('Variance expliquée par les composantes principales')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.show()

# Code principal
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

    bin_size = 0.01
    smoothing_length = 0.02
    sigma = smoothing_length / bin_size

    smoothed_data_dict = {}

    for unit_key, spike_times in spike_times_dict.items():
        binned_data, bin_times = bin_spike_times([spike_times], bin_size, duration)
        smoothed_data = smooth_data(binned_data, sigma=sigma)
        smoothed_data_dict[unit_key] = smoothed_data

    all_smoothed_data = np.vstack([data for data in smoothed_data_dict.values()])
    all_smoothed_data_T = all_smoothed_data.T

    try:
        pca_result, explained_variance, pca_components = apply_pca_torch(all_smoothed_data_T, return_components=True)
        pca_result_3d = pca_result[:, :3]
    except Exception as e:
        print(f"PCA failed for bin_size {bin_size}s and smoothing_length {smoothing_length}s: {e}")
        exit()

    plot_variance_explained_single(explained_variance)

    project_pca_around_events(pca_result, bin_times, t_0_times, window_start=-1.0, window_end=2.0)

