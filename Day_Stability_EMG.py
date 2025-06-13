import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from random import sample
from numpy.linalg import pinv

###############################################################################
#                           GLOBAL PARAMETERS
###############################################################################

COMBINED_PICKLE_FILE = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015/combined.pkl"
)

REDUCTION_METHOD = "PCA"
SEED = 18
SAVE_MODEL_PATH = "trained_decoders_trial.pt"  # or None if you don't want to save
LOAD_MODEL_PATH = None    # or define if you want to load an existing checkpoint

# Binning / Downsampling
BIN_FACTOR = 20         # Downsample factor: group every BIN_FACTOR samples
BIN_SIZE = 0.001        # Original bin size (seconds)
SMOOTHING_LENGTH = 0.05 # In seconds
PRE_TRIAL = 1.0
POST_TRIAL = 4.0
SAMPLING_RATE = 1000    # Samples per second

# PCA dims per decoder
GRU_N_PCA    = 14
LSTM_N_PCA   = 14
LINEAR_N_PCA = 18
LIGRU_N_PCA  = 14

# RNN/Linear hidden dims & lag
GRU_HIDDEN_DIM    = 12
GRU_K_LAG         = 16  # Number of timesteps in the input sequence for GRU

LSTM_HIDDEN_DIM   = 1
LSTM_K_LAG        = 1  # Number of timesteps in the input sequence for LSTM

LINEAR_HIDDEN_DIM = 1
LINEAR_K_LAG      = 1  # Number of timesteps in the input sequence for Linear decoder

LIGRU_HIDDEN_DIM  = 1
LIGRU_K_LAG       = 1  # Number of timesteps in the input sequence for LiGRU

# Training info
NUM_EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
#   CHOOSE TO VIEW RANDOM TRIALS
###############################################################################
SHOW_GRAPHS = True

###############################################################################
#   CHOOSE TRIAL-BASED OR CONTINUOUS
###############################################################################
TRAIN_TRIAL_BASED = False  # If True => trial slicing; If False => continuous approach

###############################################################################
#   Option for PCA 
###############################################################################
RECALC_PCA_EACH_DAY = True

APPLY_ZSCORE = False  # If True, do z-score (leads to correlation-based PCA).
                      # If False, do a raw covariance-based PCA (only mean-centering).

REALIGN_PCA_TO_DAY0 = True # Toggle whether to realign each day's PCA back to Day0

###############################################################################
#                         1) SEED SETTER
###############################################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################################
#   2) DOWNSAMPLING SPIKE & EMG
###############################################################################

# def lowpass(x, fs=FS_BINS, fc=LOWPASS_FC_HZ, order=4):
#     b, a = butter(order, fc / (fs / 2.0), "low")
#     return filtfilt(b, a, x, axis=0)

def downsample_spike_and_emg(spike_df, emg_data, bin_factor=10):
    """
    Summation of spikes + average of EMG every bin_factor samples.
    
    Input:
      spike_df: DataFrame of shape (T_old, n_units)
      emg_data: EMG values (DataFrame or array) of shape (T_old, n_emg)
      
    Output:
      ds_spike_df: DataFrame (T_new, n_units) where T_new = T_old // bin_factor
      ds_emg: Downsampled EMG with shape (T_new, n_emg)
    """
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg_data

    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor

    spk_arr = spike_df.values[: T_new * bin_factor, :]
    spk_arr = spk_arr.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(spk_arr, columns=spike_df.columns)

    # Downsample EMG
    if isinstance(emg_data, pd.DataFrame):
        e_arr = emg_data.values
        col_names = emg_data.columns
    else:
        e_arr = np.array(emg_data)
        col_names = None

    if e_arr.shape[0] < bin_factor:
        return ds_spike_df, emg_data

    e_arr = e_arr[: T_new * bin_factor, ...]
    if e_arr.ndim == 2:
        # shape (T_new*bin_factor, n_emg)
        e_arr = e_arr.reshape(T_new, bin_factor, e_arr.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e_arr, columns=col_names) if col_names is not None else e_arr
    else:
        # If for some reason it's 1D or unexpected shape, handle here
        ds_emg = emg_data  # fallback
    
    return ds_spike_df, ds_emg

###############################################################################
#       3) GAUSSIAN SMOOTH + ZSCORE
###############################################################################
def gaussian_smooth_1d(x, sigma):
    return gaussian_filter1d(x.astype(float), sigma=sigma)

def smooth_spike_data(x_2d, bin_size=0.001, smoothing_length=0.05):
    sigma = (smoothing_length / bin_size) / 2
    out = np.zeros_like(x_2d, dtype=float)
    for ch in range(x_2d.shape[1]):
        out[:, ch] = gaussian_smooth_1d(x_2d[:, ch], sigma)
    return out

def safe_zscore(x_2d, axis=0, eps=1e-8):
    mean = np.mean(x_2d, axis=axis, keepdims=True)
    std  = np.std(x_2d, axis=axis, keepdims=True)
    return (x_2d - mean) / (std + eps)

def smooth_emg(emg_array, window_size=20):
    """
    Rectifies and smooths an EMG signal along axis=0 (time axis) with a simple moving average.
    
    Parameters:
      emg_array: numpy array of shape (n_samples, n_channels)
      window_size: size of the moving average window
      
    Returns:
      smoothed_emg: numpy array with the same shape as emg_array
    """
    rectified_emg = np.abs(emg_array)
    smoothed_emg = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'),
                                        axis=0, arr=rectified_emg)
    return smoothed_emg
###############################################################################
#       4) TRIAL-BASED DATASET
###############################################################################
def build_trial_based_dataset(df, bin_factor, bin_size, smoothing_length, pre, post, fs):
    """
    Builds lists of (spike_data, EMG_data) for each trial window.
    Returns:
      X_all: list of arrays (T_i, n_units)
      Y_all: list of arrays (T_i, n_emg)
      T_all: list of time arrays
    """
    X_all, Y_all, T_all = [], [], []

    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        time_frame = row["time_frame"]
        trial_starts= row["trial_start_time"]

        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            print(f"[WARNING] row {idx} => empty spike_df, skipping.")
            continue
        if emg_val is None:
            print(f"[WARNING] row {idx} => EMG is None, skipping.")
            continue
        if not isinstance(time_frame, np.ndarray) or len(time_frame) == 0:
            print(f"[WARNING] row {idx} => time_frame empty, skipping.")
            continue
        if not isinstance(trial_starts, np.ndarray) or len(trial_starts) == 0:
            print(f"[WARNING] row {idx} => no trial_starts, skipping.")
            continue

        # Downsample
        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        T_old = len(time_frame)
        T_new = T_old // bin_factor
        ds_time = time_frame[: (T_new * bin_factor)]
        ds_time = ds_time.reshape(T_new, bin_factor).mean(axis=1)

        effective_fs = fs // bin_factor

        for ts in trial_starts:
            idx_center = np.argmin(np.abs(ds_time - ts))
            start_idx = max(0, idx_center - int(pre * effective_fs))
            end_idx   = min(len(ds_time), idx_center + int(post * effective_fs))
            if end_idx <= start_idx:
                continue

            X_win = ds_spike_df.values[start_idx:end_idx, :]
            # EMG slice
            if isinstance(ds_emg, pd.DataFrame):
                e_arr = ds_emg.values
            else:
                e_arr = np.array(ds_emg)
            Y_win = e_arr[start_idx:end_idx, :]
            t_win = ds_time[start_idx:end_idx]

            if X_win.shape[0] == 0:
                continue

            X_all.append(X_win)
            Y_all.append(Y_win)
            T_all.append(t_win)
    return X_all, Y_all, T_all

###############################################################################
#       4b) CONTINUOUS DATASET (no trial slicing)
###############################################################################
def build_continuous_dataset(df, bin_factor, bin_size, smoothing_length):
    """
    Concatenate all rows (assuming each row is a chunk),
    downsample spikes + EMG, smooth spikes, optionally z-score,
    then return big_spike_arr, big_emg_arr.
    """
    all_spike_list = []
    all_emg_list   = []

    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]

        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            print(f"[WARNING] row {idx} => empty spike_df, skipping.")
            continue
        if emg_val is None:
            print(f"[WARNING] row {idx} => EMG is None, skipping.")
            continue

        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0:
            continue

        spk_arr = ds_spike_df.values
        if isinstance(ds_emg, pd.DataFrame):
            e_arr = ds_emg.values
        else:
            e_arr = np.array(ds_emg)

        # Smooth spikes
        sm = smooth_spike_data(spk_arr, bin_size*bin_factor, smoothing_length)

        # Optionally z-score spikes
        if APPLY_ZSCORE:
            final_spikes = safe_zscore(sm, axis=0)
        else:
            final_spikes = sm

        # smoothed_emg = smooth_emg(e_arr, window_size=5)
        smoothed_emg = e_arr
        all_spike_list.append(final_spikes)
        all_emg_list.append(smoothed_emg)

    if len(all_spike_list) == 0:
        return np.empty((0,)), np.empty((0,))

    big_spike_arr = np.concatenate(all_spike_list, axis=0)
    big_emg_arr   = np.concatenate(all_emg_list,   axis=0)

    return big_spike_arr, big_emg_arr

###############################################################################
#       5) DATASET BUILDERS FOR RNN/LINEAR
###############################################################################
def create_rnn_dataset(X_list, Y_list, seq_len):
    """
    For trial-based:
      X_list[i]: shape (T_i, n_features)
      Y_list[i]: shape (T_i, n_emg)
    We'll create sequences of length seq_len.
    """
    X_out, Y_out = []
    X_out, Y_out = [], []
    for X, Y in zip(X_list, Y_list):
        T_i = X.shape[0]
        if T_i <= seq_len:
            continue
        for t in range(seq_len, T_i):
            X_out.append(X[t - seq_len : t, :])
            Y_out.append(Y[t, :])  # shape (n_emg,)
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset(X_list, Y_list, seq_len):
    """
    Flatten the input window for each step => output dimension is n_emg.
    """
    X_out, Y_out = [], []
    for X, Y in zip(X_list, Y_list):
        T_i = X.shape[0]
        if T_i <= seq_len:
            continue
        for t in range(seq_len, T_i):
            window = X[t - seq_len : t, :].reshape(-1)
            X_out.append(window)
            Y_out.append(Y[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

###############################################################################
#       5b) CONTINUOUS RNN/LINEAR
###############################################################################
def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    """
    X_arr: (T, n_features), Y_arr: (T, n_emg)
    Return:
      X_out: (N, seq_len, n_features)
      Y_out: (N, n_emg)
    """
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T_i = X_arr.shape[0]
    for t in range(seq_len, T_i):
        X_out.append(X_arr[t-seq_len:t, :])
        Y_out.append(Y_arr[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    """
    Flatten each seq_len window => shape (seq_len * n_features).
    Output is (n_emg).
    """
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len * X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T_i = X_arr.shape[0]
    for t in range(seq_len, T_i):
        window = X_arr[t-seq_len:t, :].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

###############################################################################
#       6) MODEL DEFINITIONS (GRU, LSTM, LINEAR, LIGRU) => multi-output
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # final time step
        return self.fc(out)  # (B, n_emg)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)  # (B, n_emg)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        return self.lin2(x)  # (B, n_emg)

class LiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, x, h):
        z = torch.sigmoid(self.x2z(x) + self.h2z(h))
        h_candidate = torch.relu(self.x2h(x) + self.h2h(h))
        return (1 - z) * h + z * h_candidate

class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)  # (B, n_emg)

###############################################################################
#       7) TRAIN/EVAL FUNCTIONS
###############################################################################
def train_decoder(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)  # (B, n_emg)
        loss = criterion(pred, Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else float("nan")

def evaluate_decoder(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            pred = model(Xb)
            loss = criterion(pred, Yb)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else float("nan")

###############################################################################
#       8) VAF
###############################################################################
def compute_vaf_1d(y_true, y_pred):
    """Single-channel VAF."""
    var_resid = np.var(y_true - y_pred)
    var_true  = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    return 1.0 - (var_resid / var_true)

def compute_multichannel_vaf(y_true, y_pred):
    """
    For multi-channel EMG:
      y_true, y_pred: shape (N, n_emg)
    Returns array of shape (n_emg,) for each channel, then you can take mean.
    """
    if y_true.shape[0] == 0:
        return np.array([])
    n_ch = y_true.shape[1]
    vafs = []
    for ch in range(n_ch):
        vaf_ch = compute_vaf_1d(y_true[:, ch], y_pred[:, ch])
        vafs.append(vaf_ch)
    return np.array(vafs)

###############################################################################
#   Plot random segments for continuous
###############################################################################
def plot_continuous_random_segments(
    day_label, all_true, pred_gru, pred_lstm, pred_lin, pred_ligru, seg_len=400
):
    """
    Plots random segments of length seg_len for *channel 0 only*
    across actual vs predicted decoders in a continuous dataset.
    The logic is the same as your original, but for multi-channel 
    we only show channel 0 for each decoder's output.
    """
    T_total = len(all_true)
    if T_total < seg_len:
        print(f"[WARNING] day={day_label} => not enough length to plot segments.")
        return
    num_segments = 6
    possible_starts = np.arange(T_total - seg_len)
    chosen_starts = np.random.choice(
        possible_starts, size=min(num_segments, len(possible_starts)), replace=False
    )

    fig, axes = plt.subplots(3, 2, figsize=(15,12))
    axes = axes.flatten()
    for i, start_idx in enumerate(chosen_starts):
        ax = axes[i]
        end_idx = start_idx + seg_len
        # we only plot channel 0
        actual_seg = all_true[start_idx:end_idx, 0]
        gru_seg    = pred_gru[start_idx:end_idx,   0]
        lstm_seg   = pred_lstm[start_idx:end_idx,  0]
        lin_seg    = pred_lin[start_idx:end_idx,   0]
        ligru_seg  = pred_ligru[start_idx:end_idx, 0]

        vaf_gru   = compute_vaf_1d(actual_seg, gru_seg)
        vaf_lstm  = compute_vaf_1d(actual_seg, lstm_seg)
        vaf_lin   = compute_vaf_1d(actual_seg, lin_seg)
        vaf_ligru = compute_vaf_1d(actual_seg, ligru_seg)

        time_axis = np.arange(seg_len)*1e-2
        ax.plot(time_axis, actual_seg, label="Actual")
        ax.plot(time_axis, gru_seg,   label=f"GRU(VAF={vaf_gru:.2f})",   linestyle='--')
        # ax.plot(time_axis, lstm_seg,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        # ax.plot(time_axis, lin_seg,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        # ax.plot(time_axis, ligru_seg, label=f"LiGRU(VAF={vaf_ligru:.2f})",linestyle='-.')
        ax.set_title(f"Day={day_label}, segment={start_idx}:{end_idx}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG channel0 (mV)")
    plt.tight_layout()
    plt.show()

###############################################################################
#   Plot 6 random trials for trial-based
###############################################################################
def plot_trial_random_samples(
    day_label, X_all_list, Y_all_list,
    gru_preds_by_trial, lstm_preds_by_trial,
    lin_preds_by_trial, ligru_preds_by_trial
):
    """
    Plots random trial segments (channel 0 only).
    Same logic as original, except we handle multi-channel 
    by picking Y[:,0].
    """
    n_trials = len(X_all_list)
    if n_trials == 0:
        print(f"[WARNING] No trials to plot for day={day_label}")
        return
    num_plots = min(n_trials, 6)
    trial_idxs = sample(range(n_trials), num_plots)

    fig, axes = plt.subplots(3, 2, figsize=(15,12))
    axes = axes.flatten()
    for i, trial_idx in enumerate(trial_idxs):
        ax = axes[i]
        actual = Y_all_list[trial_idx][:, 0]  # channel 0 only
        t_len = len(actual)
        time_axis = np.arange(t_len)

        gru_pred  = gru_preds_by_trial[trial_idx][:,0]
        vaf_gru   = compute_vaf_1d(actual, gru_pred)

        lstm_pred = lstm_preds_by_trial[trial_idx][:,0]
        vaf_lstm  = compute_vaf_1d(actual, lstm_pred)

        lin_pred  = lin_preds_by_trial[trial_idx][:,0]
        vaf_lin   = compute_vaf_1d(actual, lin_pred)

        ligru_pred= ligru_preds_by_trial[trial_idx][:,0]
        vaf_ligru = compute_vaf_1d(actual, ligru_pred)

        ax.plot(time_axis, actual, label="Actual")
        ax.plot(time_axis, gru_pred,   label=f"GRU(VAF={vaf_gru:.2f})",   linestyle='--')
        # ax.plot(time_axis, lstm_pred,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        # ax.plot(time_axis, lin_pred,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        # ax.plot(time_axis, ligru_pred, label=f"LiGRU(VAF={vaf_ligru:.2f})", linestyle='-.')
        ax.set_title(f"Day={day_label}, trial={trial_idx}")
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("EMG channel0")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

###############################################################################
#   Helper: gather all day-level spikes for PCA
###############################################################################
def gather_day_spike_data_for_pca(day_df):
    """
    We do exactly the same logic as before, but ignoring 'force'.
    If TRAIN_TRIAL_BASED, gather from trial-based approach.
    If continuous, gather from build_continuous_dataset.
    """
    if TRAIN_TRIAL_BASED:
        all_spikes = []
        for idx, row in day_df.iterrows():
            spike_df = row["spike_counts"]
            if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
                continue
            ds_spike_df, _ = downsample_spike_and_emg(spike_df, row["EMG"], BIN_FACTOR)
            if ds_spike_df.shape[0] == 0:
                continue

            sm = smooth_spike_data(
                ds_spike_df.values,
                bin_size=BIN_SIZE*BIN_FACTOR,
                smoothing_length=SMOOTHING_LENGTH
            )

            if APPLY_ZSCORE:
                z  = safe_zscore(sm, axis=0)
                all_spikes.append(z)
            else:
                all_spikes.append(sm)

        if len(all_spikes) == 0:
            return np.empty((0,0))
        return np.concatenate(all_spikes, axis=0)

    else:
        big_spike_arr, _ = build_continuous_dataset(
            day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
        )
        return big_spike_arr

###############################################################################
#   PCA alignment: compute rotation R
###############################################################################
def compute_alignment_matrix(V_dayD, V_day0):
    return pinv(V_dayD) @ V_day0

###############################################################################
#       9) MAIN SCRIPT
###############################################################################
def main():
    print(f"[INFO] Using device: {DEVICE}")
    set_seed(SEED)

    print(f"[INFO] Loading combined DataFrame from '{COMBINED_PICKLE_FILE}' ...")
    combined_df = pd.read_pickle(COMBINED_PICKLE_FILE)
    print("[INFO] combined_df.shape =", combined_df.shape)

    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y/%m/%d")

    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!")
        return

    day0 = unique_days[0]
    test_days = [d for d in unique_days]
    print("[DEBUG] unique_days =>", unique_days)
    print("[DEBUG] day0 =>", day0)

    # Detect how many EMG channels from the first row that has valid EMG
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]
                break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]
                break
    if n_emg_channels == 0:
        print("[ERROR] Could not detect EMG channels from the DataFrame.")
        return
    print(f"[INFO] Number of EMG channels detected: {n_emg_channels}")

    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)
    print(f"[DEBUG] train_df shape={train_df.shape}")

    max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
    from sklearn.decomposition import PCA
    pca_model = None
    if REDUCTION_METHOD.upper() == "PCA":
        print(f"[DEBUG] Fitting PCA n_components={max_dim} on day0 ...")
        day0_data = gather_day_spike_data_for_pca(train_df)
        if day0_data.shape[0] == 0:
            print("[ERROR] No valid spike data in day0 after smoothing! Exiting.")
            return

        pca_model = PCA(n_components=max_dim, random_state=SEED)
        pca_model.fit(day0_data)
    else:
        print("[DEBUG] Not applying PCA (REDUCTION_METHOD != 'PCA')")

    global_pca_model = pca_model

    ###########################################################################
    # 2) Build training set using day0
    ###########################################################################
    def build_dayX_decoder_data(df, day_pca_model, n_pca, seq_len, is_linear=False):
        """
        Returns X_arr, Y_arr for training or testing.
        Replaces references to force with EMG multi-ch.
        """
        if TRAIN_TRIAL_BASED:
            X_list, Y_list, _ = build_trial_based_dataset(
                df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            X_out, Y_out = [], []
            for X_trial, Y_trial in zip(X_list, Y_list):
                if X_trial.shape[0] == 0:
                    continue
                sm = smooth_spike_data(
                    X_trial,
                    bin_size=BIN_SIZE*BIN_FACTOR,
                    smoothing_length=SMOOTHING_LENGTH
                )

                if APPLY_ZSCORE:
                    z  = safe_zscore(sm, axis=0)
                else:
                    z  = sm

                if day_pca_model is not None:
                    full_proj = day_pca_model.transform(z)
                    X_pca = full_proj[:, :n_pca]
                else:
                    X_pca = z[:, :n_pca]

                X_out.append(X_pca)
                Y_out.append(Y_trial)

            if not is_linear:
                X_arr, Y_arr = create_rnn_dataset(X_out, Y_out, seq_len)
            else:
                X_arr, Y_arr = create_linear_dataset(X_out, Y_out, seq_len)
            return X_arr, Y_arr

        else:
            big_spike_arr, big_emg_arr = build_continuous_dataset(
                df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
            )
            if big_spike_arr.shape[0] == 0:
                return np.empty((0,0)), np.empty((0,0))
            if day_pca_model is not None:
                z_full = day_pca_model.transform(big_spike_arr)
            else:
                z_full = big_spike_arr

            X_pca = z_full[:, :n_pca]

            if not is_linear:
                X_arr, Y_arr = create_rnn_dataset_continuous(X_pca, big_emg_arr, seq_len)
            else:
                X_arr, Y_arr = create_linear_dataset_continuous(X_pca, big_emg_arr, seq_len)
            return X_arr, Y_arr

    X_gru_train,   Y_gru_train   = build_dayX_decoder_data(train_df, pca_model, GRU_N_PCA,   GRU_K_LAG,   is_linear=False)
    X_lstm_train,  Y_lstm_train  = build_dayX_decoder_data(train_df, pca_model, LSTM_N_PCA,  LSTM_K_LAG,  is_linear=False)
    X_lin_train,   Y_lin_train   = build_dayX_decoder_data(train_df, pca_model, LINEAR_N_PCA,LINEAR_K_LAG,is_linear=True)
    X_ligru_train, Y_ligru_train = build_dayX_decoder_data(train_df, pca_model, LIGRU_N_PCA, LIGRU_K_LAG, is_linear=False)

    print("[DEBUG] day0 shapes => GRU:", X_gru_train.shape, Y_gru_train.shape)
    print("[DEBUG] day0 shapes => LSTM:", X_lstm_train.shape, Y_lstm_train.shape)
    print("[DEBUG] day0 shapes => Linear:", X_lin_train.shape, Y_lin_train.shape)
    print("[DEBUG] day0 shapes => LiGRU:", X_ligru_train.shape, Y_ligru_train.shape)

    ds_gru   = TensorDataset(torch.tensor(X_gru_train),   torch.tensor(Y_gru_train))
    dl_gru   = DataLoader(ds_gru,   batch_size=BATCH_SIZE, shuffle=True)
    ds_lstm  = TensorDataset(torch.tensor(X_lstm_train),  torch.tensor(Y_lstm_train))
    dl_lstm  = DataLoader(ds_lstm,  batch_size=BATCH_SIZE, shuffle=True)
    ds_lin   = TensorDataset(torch.tensor(X_lin_train),   torch.tensor(Y_lin_train))
    dl_lin   = DataLoader(ds_lin,   batch_size=BATCH_SIZE, shuffle=True)
    ds_ligru = TensorDataset(torch.tensor(X_ligru_train), torch.tensor(Y_ligru_train))
    dl_ligru = DataLoader(ds_ligru, batch_size=BATCH_SIZE, shuffle=True)

    ###########################################################################
    # 3) Initialize and train models on day0
    ###########################################################################
    # n_emg_channels detected above
    gru_model    = GRUDecoder(GRU_N_PCA,    GRU_HIDDEN_DIM,   n_emg_channels).to(DEVICE)
    lstm_model   = LSTMDecoder(LSTM_N_PCA,  LSTM_HIDDEN_DIM,  n_emg_channels).to(DEVICE)
    linear_model = LinearLagDecoder(LINEAR_K_LAG * LINEAR_N_PCA, LINEAR_HIDDEN_DIM, n_emg_channels).to(DEVICE)
    ligru_model  = LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_emg_channels).to(DEVICE)

    if LOAD_MODEL_PATH is not None and os.path.exists(LOAD_MODEL_PATH):
        print(f"[INFO] Found existing model checkpoint at {LOAD_MODEL_PATH}. Loading...")
        checkpoint = torch.load(LOAD_MODEL_PATH, map_location=DEVICE)
        gru_model.load_state_dict(checkpoint["gru_model"])
        lstm_model.load_state_dict(checkpoint["lstm_model"])
        linear_model.load_state_dict(checkpoint["linear_model"])
        ligru_model.load_state_dict(checkpoint["ligru_model"])
        print("[INFO] Models loaded. Skipping training...")
    else:
        gru_opt    = optim.Adam(gru_model.parameters(),    lr=LEARNING_RATE)
        lstm_opt   = optim.Adam(lstm_model.parameters(),   lr=LEARNING_RATE)
        lin_opt    = optim.Adam(linear_model.parameters(), lr=LEARNING_RATE)
        ligru_opt  = optim.Adam(ligru_model.parameters(),  lr=LEARNING_RATE)
        
        criterion = nn.MSELoss()

        print("[INFO] Training decoders on Day0 ...")
        for ep in range(1, NUM_EPOCHS + 1):
            loss_gru   = train_decoder(gru_model,    dl_gru,   gru_opt, criterion)
            loss_lstm  = train_decoder(lstm_model,   dl_lstm,  lstm_opt, criterion)
            loss_lin   = train_decoder(linear_model, dl_lin,   lin_opt, criterion)
            loss_ligru = train_decoder(ligru_model,  dl_ligru, ligru_opt, criterion)

            if ep % 10 == 0:
                print(f"Epoch {ep}/{NUM_EPOCHS} => "
                      f"GRU:{loss_gru:.4f}, "
                      f"LSTM:{loss_lstm:.4f}, "
                      f"Lin:{loss_lin:.4f}, "
                      f"LiGRU:{loss_ligru:.4f}")
        print("[INFO] Training complete.\n")
        # Save the trained models if we want
        if SAVE_MODEL_PATH is not None:
            print(f"[INFO] Saving trained models to {SAVE_MODEL_PATH} ...")
            torch.save({
                "gru_model": gru_model.state_dict(),
                "lstm_model": lstm_model.state_dict(),
                "linear_model": linear_model.state_dict(),
                "ligru_model": ligru_model.state_dict()
            }, SAVE_MODEL_PATH)

    ###########################################################################
    # 4) Evaluate each day
    ###########################################################################
    results_days = []
    results_gru  = []
    results_lstm = []
    results_lin  = []
    results_ligru= []

    def evaluate_day(day_df, day_label):
        local_pca_model = global_pca_model
        if REDUCTION_METHOD.upper() == "PCA" and RECALC_PCA_EACH_DAY:
            day_z_for_pca = gather_day_spike_data_for_pca(day_df)
            if day_z_for_pca.shape[0] > 0:
                from sklearn.decomposition import PCA
                new_pca = PCA(n_components=max_dim, random_state=SEED)
                new_pca.fit(day_z_for_pca)
                local_pca_model = new_pca

        V_day0_full = None
        V_dayD_full = None
        if local_pca_model is global_pca_model:
            print("[DEBUG] local_pca_model is the SAME as global_pca_model!")
        else:
            print("[DEBUG] local_pca_model is a new/different PCA model for dayD.")
        if local_pca_model is not None and global_pca_model is not None:
            V_day0_full = global_pca_model.components_.T
            V_dayD_full = local_pca_model.components_.T

        if TRAIN_TRIAL_BASED:
            X_all_list, Y_all_list, _ = build_trial_based_dataset(
                day_df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            if len(X_all_list) == 0:
                print(f"[WARNING] Day={day_label} => no trials found in trial-based approach.")
                return np.nan, np.nan, np.nan, np.nan

            def decode_trial(X_trial, n_pca, seq_len, model, is_linear=False):
                # same logic, but for multi-channel EMG
                sm = smooth_spike_data(
                    X_trial,
                    bin_size=BIN_SIZE*BIN_FACTOR,
                    smoothing_length=SMOOTHING_LENGTH
                )
                if APPLY_ZSCORE:
                    z  = safe_zscore(sm, axis=0)
                else:
                    z  = sm

                if local_pca_model is not None:
                    if (REALIGN_PCA_TO_DAY0
                        and (V_day0_full is not None)
                        and (V_dayD_full is not None)
                        and (local_pca_model is not global_pca_model)):
                        V_day0_k = V_day0_full[:, :n_pca]
                        V_dayD_k = V_dayD_full[:, :n_pca]
                        R = compute_alignment_matrix(V_dayD_k, V_day0_k)
                        z_aligned = (z @ V_dayD_k) @ R
                        X_pca = z_aligned
                    else:
                        full_proj = local_pca_model.transform(z)
                        X_pca     = full_proj[:, :n_pca]
                else:
                    X_pca = z[:, :n_pca]

                T_i = X_pca.shape[0]
                pred_out = np.full((T_i, n_emg_channels), np.nan)
                if not is_linear:
                    if T_i > seq_len:
                        X_lag = []
                        idx_list = []
                        for t in range(seq_len, T_i):
                            X_lag.append(X_pca[t-seq_len:t, :])
                            idx_list.append(t)
                        X_lag = np.array(X_lag, dtype=np.float32)
                        with torch.no_grad():
                            outp = model(torch.tensor(X_lag).to(DEVICE))
                        outp = outp.cpu().numpy()  # shape (n_windows, n_emg)
                        for k, out_idx in enumerate(idx_list):
                            pred_out[out_idx, :] = outp[k, :]
                else:
                    if T_i > seq_len:
                        X_lag = []
                        idx_list = []
                        for t in range(seq_len, T_i):
                            window = X_pca[t-seq_len:t, :].reshape(-1)
                            X_lag.append(window)
                            idx_list.append(t)
                        X_lag = np.array(X_lag, dtype=np.float32)
                        with torch.no_grad():
                            outp = model(torch.tensor(X_lag).to(DEVICE))
                        outp = outp.cpu().numpy()
                        for k, out_idx in enumerate(idx_list):
                            pred_out[out_idx, :] = outp[k, :]
                return pred_out

            gru_preds_by_trial   = []
            lstm_preds_by_trial  = []
            lin_preds_by_trial   = []
            ligru_preds_by_trial = []

            for X_trial in X_all_list:
                gru_preds_by_trial.append(
                    decode_trial(X_trial, GRU_N_PCA, GRU_K_LAG, gru_model,   is_linear=False)
                )
                lstm_preds_by_trial.append(
                    decode_trial(X_trial, LSTM_N_PCA, LSTM_K_LAG, lstm_model, is_linear=False)
                )
                lin_preds_by_trial.append(
                    decode_trial(X_trial, LINEAR_N_PCA, LINEAR_K_LAG, linear_model, is_linear=True)
                )
                ligru_preds_by_trial.append(
                    decode_trial(X_trial, LIGRU_N_PCA, LIGRU_K_LAG, ligru_model,  is_linear=False)
                )

            # flatten for VAF
            gru_all_true = np.concatenate(Y_all_list, axis=0) if Y_all_list else np.array([])
            gru_all_pred = np.concatenate(gru_preds_by_trial, axis=0) if gru_preds_by_trial else np.array([])
            if gru_all_true.size == 0 or gru_all_pred.size == 0:
                gru_vaf_day = np.nan
            else:
                gru_vafs = compute_multichannel_vaf(gru_all_true, gru_all_pred)
                gru_vaf_day = np.nanmean(gru_vafs)

            lstm_all_true= np.concatenate(Y_all_list, axis=0) if Y_all_list else np.array([])
            lstm_all_pred= np.concatenate(lstm_preds_by_trial, axis=0) if lstm_preds_by_trial else np.array([])
            if lstm_all_true.size == 0 or lstm_all_pred.size == 0:
                lstm_vaf_day = np.nan
            else:
                lstm_vafs = compute_multichannel_vaf(lstm_all_true, lstm_all_pred)
                lstm_vaf_day = np.nanmean(lstm_vafs)

            lin_all_true = np.concatenate(Y_all_list, axis=0) if Y_all_list else np.array([])
            lin_all_pred = np.concatenate(lin_preds_by_trial, axis=0) if lin_preds_by_trial else np.array([])
            if lin_all_true.size == 0 or lin_all_pred.size == 0:
                lin_vaf_day = np.nan
            else:
                lin_vafs = compute_multichannel_vaf(lin_all_true, lin_all_pred)
                lin_vaf_day = np.nanmean(lin_vafs)

            ligru_all_true= np.concatenate(Y_all_list, axis=0) if Y_all_list else np.array([])
            ligru_all_pred= np.concatenate(ligru_preds_by_trial, axis=0) if ligru_preds_by_trial else np.array([])
            if ligru_all_true.size == 0 or ligru_all_pred.size == 0:
                ligru_vaf_day = np.nan
            else:
                ligru_vafs = compute_multichannel_vaf(ligru_all_true, ligru_all_pred)
                ligru_vaf_day = np.nanmean(ligru_vafs)

            print(f"[RESULT] Day={day_label} => VAF: "
                  f"GRU={gru_vaf_day:.3f}, LSTM={lstm_vaf_day:.3f}, "
                  f"Linear={lin_vaf_day:.3f}, LiGRU={ligru_vaf_day:.3f}")

            # If we want random plots
            if SHOW_GRAPHS:
                plot_trial_random_samples(
                    day_label=day_label,
                    X_all_list=X_all_list,
                    Y_all_list=Y_all_list,
                    gru_preds_by_trial=gru_preds_by_trial,
                    lstm_preds_by_trial=lstm_preds_by_trial,
                    lin_preds_by_trial=lin_preds_by_trial,
                    ligru_preds_by_trial=ligru_preds_by_trial
                )

            return gru_vaf_day, lstm_vaf_day, lin_vaf_day, ligru_vaf_day

        else:
            # continuous approach
            big_spike_arr, big_emg_arr = build_continuous_dataset(
                day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
            )
            print(f"[DEBUG-continuous] big_spike_arr.shape = {big_spike_arr.shape}, "
                  f"big_emg_arr.shape = {big_emg_arr.shape}")

            if big_spike_arr.shape[0] == 0:
                print(f"[WARNING] Day={day_label} => no data in continuous approach.")
                return np.nan, np.nan, np.nan, np.nan

            if local_pca_model is not None:
                # Potential realignment to day0
                dayD_latents = local_pca_model.transform(big_spike_arr)
                if (REALIGN_PCA_TO_DAY0
                    and (V_day0_full is not None) 
                    and (V_dayD_full is not None)
                    and (local_pca_model is not global_pca_model)):
                    R = compute_alignment_matrix(V_dayD_full, V_day0_full)
                    day0_proj = (dayD_latents @ R)
                else:
                    day0_proj = dayD_latents
            else:
                day0_proj = big_spike_arr

            # Evaluate GRU
            X_gru_te = day0_proj[:, :GRU_N_PCA]
            Y_gru_te = big_emg_arr
            X_seq, Y_seq = create_rnn_dataset_continuous(X_gru_te, Y_gru_te, GRU_K_LAG)
            gru_all_pred = []
            gru_all_true = []
            gru_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    batch_X = X_seq[i:i+BATCH_SIZE]
                    batch_Y = Y_seq[i:i+BATCH_SIZE]
                    batch_X_t = torch.tensor(batch_X).to(DEVICE)
                    out = gru_model(batch_X_t)
                    gru_all_pred.append(out.cpu().numpy())
                    gru_all_true.append(batch_Y)
            if gru_all_pred:
                gru_all_pred = np.concatenate(gru_all_pred, axis=0)
                gru_all_true = np.concatenate(gru_all_true, axis=0)
                gru_vafs = compute_multichannel_vaf(gru_all_true, gru_all_pred)
                gru_vaf_day  = np.nanmean(gru_vafs)
            else:
                gru_vaf_day = np.nan

            # Evaluate LSTM
            X_lstm_te = day0_proj[:, :LSTM_N_PCA]
            Y_lstm_te = big_emg_arr
            X_seq, Y_seq = create_rnn_dataset_continuous(X_lstm_te, Y_lstm_te, LSTM_K_LAG)
            lstm_all_pred = []
            lstm_all_true = []
            lstm_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    batch_X = X_seq[i:i+BATCH_SIZE]
                    batch_Y = Y_seq[i:i+BATCH_SIZE]
                    batch_X_t = torch.tensor(batch_X).to(DEVICE)
                    out = lstm_model(batch_X_t)
                    lstm_all_pred.append(out.cpu().numpy())
                    lstm_all_true.append(batch_Y)
            if lstm_all_pred:
                lstm_all_pred = np.concatenate(lstm_all_pred, axis=0)
                lstm_all_true = np.concatenate(lstm_all_true, axis=0)
                lstm_vafs = compute_multichannel_vaf(lstm_all_true, lstm_all_pred)
                lstm_vaf_day  = np.nanmean(lstm_vafs)
            else:
                lstm_vaf_day = np.nan

            # Evaluate Linear
            X_lin_te = day0_proj[:, :LINEAR_N_PCA]
            Y_lin_te = big_emg_arr
            X_seq_lin, Y_seq_lin = create_linear_dataset_continuous(X_lin_te, Y_lin_te, LINEAR_K_LAG)
            lin_all_pred = []
            lin_all_true = []
            linear_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq_lin), BATCH_SIZE):
                    batch_X = X_seq_lin[i:i+BATCH_SIZE]
                    batch_Y = Y_seq_lin[i:i+BATCH_SIZE]
                    batch_X_t = torch.tensor(batch_X).to(DEVICE)
                    out = linear_model(batch_X_t)
                    lin_all_pred.append(out.cpu().numpy())
                    lin_all_true.append(batch_Y)
            if lin_all_pred:
                lin_all_pred = np.concatenate(lin_all_pred, axis=0)
                lin_all_true = np.concatenate(lin_all_true, axis=0)
                lin_vafs = compute_multichannel_vaf(lin_all_true, lin_all_pred)
                lin_vaf_day  = np.nanmean(lin_vafs)
            else:
                lin_vaf_day = np.nan

            # Evaluate LiGRU
            X_ligru_te = day0_proj[:, :LIGRU_N_PCA]
            Y_ligru_te = big_emg_arr
            X_seq_ligru, Y_seq_ligru = create_rnn_dataset_continuous(X_ligru_te, Y_ligru_te, LIGRU_K_LAG)
            ligru_all_pred = []
            ligru_all_true = []
            ligru_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq_ligru), BATCH_SIZE):
                    batch_X = X_seq_ligru[i:i+BATCH_SIZE]
                    batch_Y = Y_seq_ligru[i:i+BATCH_SIZE]
                    batch_X_t = torch.tensor(batch_X).to(DEVICE)
                    out = ligru_model(batch_X_t)
                    ligru_all_pred.append(out.cpu().numpy())
                    ligru_all_true.append(batch_Y)
            if ligru_all_pred:
                ligru_all_pred = np.concatenate(ligru_all_pred, axis=0)
                ligru_all_true = np.concatenate(ligru_all_true, axis=0)
                ligru_vafs = compute_multichannel_vaf(ligru_all_true, ligru_all_pred)
                ligru_vaf_day  = np.nanmean(ligru_vafs)
            else:
                ligru_vaf_day = np.nan

            print(f"[RESULT] Day={day_label} => VAF: "
                  f"GRU={gru_vaf_day:.3f}, LSTM={lstm_vaf_day:.3f}, "
                  f"Linear={lin_vaf_day:.3f}, LiGRU={ligru_vaf_day:.3f}")

            if SHOW_GRAPHS and len(X_seq) > 0:
                # optional random segments (channel 0)
                plot_continuous_random_segments(
                    day_label=day_label,
                    all_true=gru_all_true,  # shape (N, n_emg)
                    pred_gru=gru_all_pred,
                    pred_lstm=lstm_all_pred,
                    pred_lin=lin_all_pred,
                    pred_ligru=ligru_all_pred,
                    seg_len=400
                )

            return gru_vaf_day, lstm_vaf_day, lin_vaf_day, ligru_vaf_day

    for d in test_days:
        day_df = combined_df[combined_df["date"] == d].reset_index(drop=True)
        gru_vaf, lstm_vaf, lin_vaf, ligru_vaf = evaluate_day(day_df, d)
        results_days.append(d)
        results_gru.append(gru_vaf)
        results_lstm.append(lstm_vaf)
        results_lin.append(lin_vaf)
        results_ligru.append(ligru_vaf)

    fig, ax = plt.subplots(figsize=(8, 5))
    results_days = np.array(results_days)  # convert list -> np.array of dtype=object or datetime64
    day_nums = [
    (d.date() - results_days[0].date()).days
    for d in results_days]

    vaf_gru_day0   = results_gru[0]
    vaf_lstm_day0  = results_lstm[0]
    vaf_lin_day0   = results_lin[0]
    vaf_ligru_day0 = results_ligru[0]
    gru_loss   = [vaf_gru_day0   - v for v in results_gru]
    lstm_loss  = [vaf_lstm_day0  - v for v in results_lstm]
    lin_loss   = [vaf_lin_day0   - v for v in results_lin]
    ligru_loss = [vaf_ligru_day0 - v for v in results_ligru]
        
    ##############################################################################
    # FIGURE A: Relative VAF loss vs day index
    ##############################################################################
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(day_nums, gru_loss,   marker='o', label="GRU")
    ax.plot(day_nums, lstm_loss,  marker='o', label="LSTM")
    ax.plot(day_nums, lin_loss,   marker='o', label="Linear")
    ax.plot(day_nums, ligru_loss, marker='o', label="LiGRU")

    ax.set_xlabel("Day index")
    ax.set_ylabel("Relative VAF Loss")
    ax.set_title("Relative VAF Loss vs Day")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("relative_vaf_loss_by_day.png", dpi=700)
    plt.show()

    ##############################################################################
    # FIGURE B: Cumulative sum of VAF loss
    ##############################################################################
    # We simply do a running sum across the day_nums, so day0=0, day1=loss1, day2=loss1+loss2, etc.
    gru_loss_cumulative   = np.cumsum(gru_loss)
    lstm_loss_cumulative  = np.cumsum(lstm_loss)
    lin_loss_cumulative   = np.cumsum(lin_loss)
    ligru_loss_cumulative = np.cumsum(ligru_loss)

    fig, ax2 = plt.subplots(figsize=(6,4))

    ax2.plot(day_nums, gru_loss_cumulative,   marker='o', label="GRU")
    ax2.plot(day_nums, lstm_loss_cumulative,  marker='o', label="LSTM")
    ax2.plot(day_nums, lin_loss_cumulative,   marker='o', label="Linear")
    ax2.plot(day_nums, ligru_loss_cumulative, marker='o', label="LiGRU")

    ax2.set_xlabel("Day index")
    ax2.set_ylabel("Cumulative Relative VAF Loss")
    ax2.set_title("Cumulative VAF Loss across days")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("cumulative_vaf_loss.png", dpi=700)
    plt.show()

if __name__ == "__main__":
    main()
