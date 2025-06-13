import os
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
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
SAVE_MODEL_PATH = "trained_decoders_trial.pt"  # or None if you don't want to save
LOAD_MODEL_PATH = None                        # or define if you want to load an existing checkpoint
SEED = 42                                     # Seed used for PCA
# Binning / Downsampling
BIN_FACTOR = 20         # Downsample factor: group every BIN_FACTOR samples
BIN_SIZE = 0.001        # Original bin size (seconds)
SMOOTHING_LENGTH = 0.05 # In seconds
PRE_TRIAL = 1.0
POST_TRIAL = 4.0
SAMPLING_RATE = 1000    # Samples per second

# Seed per decoder
DECODER_SEEDS = {
    "gru":    42,
    "lstm":   4242,
    "linear": 424242,
    "ligru":  12345,
}
# PCA dims per decoder
GRU_N_PCA    = 16
LSTM_N_PCA   = 16
LINEAR_N_PCA = 18
LIGRU_N_PCA  = 14

# RNN/Linear hidden dims & lag
GRU_HIDDEN_DIM    = 17
GRU_K_LAG         = 12

LSTM_HIDDEN_DIM   = 18
LSTM_K_LAG        = 10

LINEAR_HIDDEN_DIM = 64
LINEAR_K_LAG      = 16

LIGRU_HIDDEN_DIM  = 5
LIGRU_K_LAG       = 16

# Training info
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)
# Graph toggles
SHOW_GRAPHS = True

# Trial-based or continuous
TRAIN_TRIAL_BASED = False

# PCA realignment
RECALC_PCA_EACH_DAY = True
APPLY_ZSCORE = False
REALIGN_PCA_TO_DAY0 = False

# ------------------ CROSS-VALIDATION ADDITIONS ------------------
CROSSVAL_ENABLE = False  # On/off switch for cross-validation
CROSSVAL_RUNS   = 20      # How many distinct splits
SAVE_RESULTS_PKL= 'crossval_results_recalc.pkl'   # e.g. "crossval_results.pkl" or None if not saving

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

def butter_lowpass(data, fs, order=4):
    """
    data : array (..., n_channels)
    fs   : fréquence d'échantillonnage EN HZ du signal 'data'
    Retourne le signal filtré (mêmes dimensions).
    """
    nyq   = 0.5 * fs
    norm  = 5.0 / nyq              # 5 Hz
    b, a  = butter(order, norm, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def downsample_spike_and_emg(spike_df, emg_data, bin_factor=10):
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg_data

    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor

    spk_arr = spike_df.values[: T_new * bin_factor, :]
    spk_arr = spk_arr.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(spk_arr, columns=spike_df.columns)

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
        e_arr = e_arr.reshape(T_new, bin_factor, e_arr.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e_arr, columns=col_names) if col_names is not None else e_arr
    else:
        ds_emg = emg_data
    
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

###############################################################################
#       4) TRIAL-BASED DATASET
###############################################################################
def build_trial_based_dataset(df, bin_factor, bin_size, smoothing_length, pre, post, fs):
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

        eff_fs = SAMPLING_RATE // BIN_FACTOR      
        e_arr  = butter_lowpass(e_arr, eff_fs)
        sm = smooth_spike_data(spk_arr, bin_size*bin_factor, smoothing_length)
        if APPLY_ZSCORE:
            final_spikes = safe_zscore(sm, axis=0)
        else:
            final_spikes = sm
        
        rectified_emg = np.abs(e_arr)
        # smoothed_emg = np.apply_along_axis(lambda x: np.convolve(x, np.ones(20)/20, mode='same'), axis=0, arr=rectified_emg)
        smoothed_emg = rectified_emg
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
    X_out, Y_out = [], []
    for X, Y in zip(X_list, Y_list):
        T_i = X.shape[0]
        if T_i <= seq_len:
            continue
        for t in range(seq_len, T_i):
            X_out.append(X[t - seq_len : t, :])
            Y_out.append(Y[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset(X_list, Y_list, seq_len):
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
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T_i = X_arr.shape[0]
    for t in range(seq_len, T_i):
        X_out.append(X_arr[t-seq_len:t, :])
        Y_out.append(Y_arr[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
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
#       6) MODEL DEFINITIONS (GRU, LSTM, LINEAR, LIGRU)
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        return self.lin2(x)

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
        return self.fc(h)

###############################################################################
#       7) TRAIN/EVAL FUNCTIONS
###############################################################################
def train_decoder(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
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
    var_resid = np.var(y_true - y_pred)
    var_true  = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    return 1.0 - (var_resid / var_true)

def compute_multichannel_vaf(y_true, y_pred):
    if y_true.shape[0] == 0:
        return np.array([])
    n_ch = y_true.shape[1]
    vafs = []
    for ch in range(n_ch):
        vaf_ch = compute_vaf_1d(y_true[:, ch], y_pred[:, ch])
        vafs.append(vaf_ch)
    return np.array(vafs)

def evaluate_on_split(model, X_val, Y_val, seq_len, is_linear=False):
    if X_val.shape[0] < 50:
        return np.nan

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            bx = torch.tensor(X_val[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())

    if preds:
        preds = np.concatenate(preds, axis=0)
        vafs = compute_multichannel_vaf(Y_val, preds)
        return np.nanmean(vafs)
    else:
        return np.nan
###############################################################################
#   Plot random segments for continuous
###############################################################################
def plot_continuous_random_segments(
    day_label, all_true, pred_gru, pred_lstm, pred_lin, pred_ligru, seg_len=400
):
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
        # only channel 0 for illustration
        actual_seg = all_true[start_idx:end_idx, 0]
        gru_seg    = pred_gru[start_idx:end_idx,   0]
        lstm_seg   = pred_lstm[start_idx:end_idx,  0]
        lin_seg    = pred_lin[start_idx:end_idx,   0]
        ligru_seg  = pred_ligru[start_idx:end_idx, 0]

        vaf_gru   = compute_vaf_1d(actual_seg, gru_seg)
        vaf_lstm  = compute_vaf_1d(actual_seg, lstm_seg)
        vaf_lin   = compute_vaf_1d(actual_seg, lin_seg)
        vaf_ligru = compute_vaf_1d(actual_seg, ligru_seg)

        time_axis = np.arange(seg_len)
        ax.plot(time_axis, actual_seg, label="Actual")
        ax.plot(time_axis, gru_seg,   label=f"GRU(VAF={vaf_gru:.2f})",   linestyle='--')
        ax.plot(time_axis, lstm_seg,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        ax.plot(time_axis, lin_seg,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        ax.plot(time_axis, ligru_seg, label=f"LiGRU(VAF={vaf_ligru:.2f})",linestyle='-.')
        ax.set_title(f"Day={day_label}, seg={start_idx}:{end_idx}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("EMG channel0")
    plt.tight_layout()
    plt.show()

###############################################################################
#   Plot random trials for trial-based
###############################################################################
def plot_trial_random_samples(
    day_label, X_all_list, Y_all_list,
    gru_preds_by_trial, lstm_preds_by_trial,
    lin_preds_by_trial, ligru_preds_by_trial
):
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
        actual = Y_all_list[trial_idx][:, 0]
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
        ax.plot(time_axis, lstm_pred,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        ax.plot(time_axis, lin_pred,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        ax.plot(time_axis, ligru_pred, label=f"LiGRU(VAF={vaf_ligru:.2f})", linestyle='-.')
        ax.set_title(f"Day={day_label}, trial={trial_idx}")
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("EMG channel0")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

###############################################################################
#   Helper: gather day-level spikes for PCA
###############################################################################
def gather_day_spike_data_for_pca(day_df):
    if TRAIN_TRIAL_BASED:
        all_spikes = []
        for idx, row in day_df.iterrows():
            spike_df = row["spike_counts"]
            if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
                continue
            ds_spike_df, _ = downsample_spike_and_emg(spike_df, row["EMG"], BIN_FACTOR)
            if ds_spike_df.shape[0] == 0:
                continue

            sm = smooth_spike_data(ds_spike_df.values,
                                   bin_size=BIN_SIZE*BIN_FACTOR,
                                   smoothing_length=SMOOTHING_LENGTH)
            if APPLY_ZSCORE:
                z = safe_zscore(sm, axis=0)
                all_spikes.append(z)
            else:
                all_spikes.append(sm)
        if len(all_spikes) == 0:
            return np.empty((0,0))
        return np.concatenate(all_spikes, axis=0)
    else:
        big_spike_arr, _ = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
        return big_spike_arr

###############################################################################
#   PCA alignment
###############################################################################
def compute_alignment_matrix(V_dayD, V_day0):
    return pinv(V_dayD) @ V_day0


###########################################################################
    # Utility: build dataset from day0
###########################################################################
def build_dayX_decoder_data(df, day_pca_model, n_pca, seq_len, is_linear=False):
        if TRAIN_TRIAL_BASED:
            X_list, Y_list, _ = build_trial_based_dataset(
                df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            X_out, Y_out = [], []
            for X_trial, Y_trial in zip(X_list, Y_list):
                if X_trial.shape[0] == 0:
                    continue
                sm = smooth_spike_data(X_trial, bin_size=BIN_SIZE*BIN_FACTOR, smoothing_length=SMOOTHING_LENGTH)
                if APPLY_ZSCORE:
                    z = safe_zscore(sm, axis=0)
                else:
                    z = sm
                if day_pca_model is not None:
                    full_proj = day_pca_model.transform(z)
                    X_pca = full_proj[:, :n_pca]
                else:
                    X_pca = z[:, :n_pca]
                X_out.append(X_pca)
                Y_out.append(Y_trial)
            if not is_linear:
                return create_rnn_dataset(X_out, Y_out, seq_len)
            else:
                return create_linear_dataset(X_out, Y_out, seq_len)
        else:
            big_spike_arr, big_emg_arr = build_continuous_dataset(df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
            if big_spike_arr.shape[0] == 0:
                return np.empty((0,0)), np.empty((0,0))
            if day_pca_model is not None:
                z_full = day_pca_model.transform(big_spike_arr)
                X_pca = z_full[:, :n_pca]
            else:
                X_pca = big_spike_arr[:, :n_pca]

            if not is_linear:
                return create_rnn_dataset_continuous(X_pca, big_emg_arr, seq_len)
            else:
                return create_linear_dataset_continuous(X_pca, big_emg_arr, seq_len)
# -------------------------------------------------------------------------
    # Function to do a single train pass on day0 data
# -------------------------------------------------------------------------
def train_on_day0_once(Xset_dict, model_dict, seed_dict):
    """
    Xset_dict has => 'gru':(X_gru_train, Y_gru_train), ...
    model_dict => 'gru':(model, inputdim, hidden_dim, n_emg)
    Return: trained models
    """
        # ── GRU ───────────────────────────────────────────
    set_seed(seed_dict["gru"])                 # ← NEW
    ds_gru = TensorDataset(
        torch.tensor(Xset_dict["gru"][0]),
        torch.tensor(Xset_dict["gru"][1]),
    )
    ggen   = torch.Generator().manual_seed(seed_dict["gru"])  # DataLoader RNG
    dl_gru = DataLoader(ds_gru, batch_size=BATCH_SIZE,
                        shuffle=True, generator=ggen)

    # exactly the same for LSTM / Linear / LiGRU …
    set_seed(seed_dict["lstm"])
    ds_lstm = TensorDataset(
        torch.tensor(Xset_dict["lstm"][0]),
        torch.tensor(Xset_dict["lstm"][1]),
    )
    lgen   = torch.Generator().manual_seed(seed_dict["lstm"])
    dl_lstm = DataLoader(ds_lstm, batch_size=BATCH_SIZE,
                        shuffle=True, generator=lgen)

    set_seed(seed_dict["linear"])
    ds_lin = TensorDataset(
        torch.tensor(Xset_dict["linear"][0]),
        torch.tensor(Xset_dict["linear"][1]),
    )
    ngen   = torch.Generator().manual_seed(seed_dict["linear"])
    dl_lin = DataLoader(ds_lin, batch_size=BATCH_SIZE,
                        shuffle=True, generator=ngen)

    set_seed(seed_dict["ligru"])
    ds_li = TensorDataset(
        torch.tensor(Xset_dict["ligru"][0]),
        torch.tensor(Xset_dict["ligru"][1]),
    )
    ligen  = torch.Generator().manual_seed(seed_dict["ligru"])
    dl_li  = DataLoader(ds_li, batch_size=BATCH_SIZE,
                        shuffle=True, generator=ligen)
    gru_model    = model_dict['gru']
    lstm_model   = model_dict['lstm']
    linear_model = model_dict['linear']
    ligru_model  = model_dict['ligru']

    Xg, Yg = Xset_dict['gru']
    Xl, Yl = Xset_dict['lstm']
    Xn, Yn = Xset_dict['linear']
    Xli, Yli = Xset_dict['ligru']

    ds_gru   = TensorDataset(torch.tensor(Xg),   torch.tensor(Yg))
    dl_gru   = DataLoader(ds_gru,   batch_size=BATCH_SIZE, shuffle=True)
    ds_lstm  = TensorDataset(torch.tensor(Xl),   torch.tensor(Yl))
    dl_lstm  = DataLoader(ds_lstm,  batch_size=BATCH_SIZE, shuffle=True)
    ds_lin   = TensorDataset(torch.tensor(Xn),   torch.tensor(Yn))
    dl_lin   = DataLoader(ds_lin,   batch_size=BATCH_SIZE, shuffle=True)
    ds_li    = TensorDataset(torch.tensor(Xli),  torch.tensor(Yli))
    dl_li    = DataLoader(ds_li,    batch_size=BATCH_SIZE, shuffle=True)

    gru_opt    = optim.Adam(gru_model.parameters(),    lr=LEARNING_RATE)
    lstm_opt   = optim.Adam(lstm_model.parameters(),   lr=LEARNING_RATE)
    lin_opt    = optim.Adam(linear_model.parameters(), lr=LEARNING_RATE)
    ligru_opt  = optim.Adam(ligru_model.parameters(),  lr=LEARNING_RATE)
    criterion  = nn.MSELoss()

    for ep in range(1, NUM_EPOCHS+1):
        loss_gru   = train_decoder(gru_model,    dl_gru,   gru_opt, criterion)
        loss_lstm  = train_decoder(lstm_model,   dl_lstm,  lstm_opt, criterion)
        loss_lin   = train_decoder(linear_model, dl_lin,   lin_opt, criterion)
        loss_li    = train_decoder(ligru_model,  dl_li,    ligru_opt, criterion)

        # Debug print every ~10 epochs
        if ep % 10 == 0:
            print(f"  [CV train epoch {ep}/{NUM_EPOCHS}] "
                    f"GRU={loss_gru:.4f}, LSTM={loss_lstm:.4f}, LIN={loss_lin:.4f}, LiGRU={loss_li:.4f}")
    return (gru_model, lstm_model, linear_model, ligru_model)

    ###########################################################################
    # CROSS-VALIDATION LOGIC
    ###########################################################################
    # We'll define a function to random-split day0 data (the entire X, Y).
    # For continuous data, we do a random train/test split on the row dimension.
    # For trial-based, we do a random split on the trial list. 
    # We do CROSSVAL_RUNS times. 
    ###########################################################################
def random_split_indices(n_items, train_frac=0.75):
    """
    Returns train_indices, test_indices 
    each is a list of row indices
    """
    all_indices = np.arange(n_items)
    np.random.shuffle(all_indices)
    cutoff = int(train_frac * n_items)
    train_idx = all_indices[:cutoff]
    test_idx  = all_indices[cutoff:]
    return train_idx, test_idx

###############################################################################
#   MAIN SCRIPT
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

    # Detect how many EMG channels
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
        print("[ERROR] Could not detect EMG channels from DataFrame.")
        return
    print(f"[INFO] # EMG channels detected: {n_emg_channels}")

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

    
    

    
    # We'll store crossval results in arrays:
    # shape (CROSSVAL_RUNS, len(test_days)) for each decoder
    # so e.g. results_cv_gru[i, day] is the mean VAF for the i-th crossval repeat.
    results_cv_gru   = []
    results_cv_lstm  = []
    results_cv_lin   = []
    results_cv_ligru = []

    # Build "full day0 sets" once
    X_gru_full,   Y_gru_full   = build_dayX_decoder_data(train_df, pca_model, GRU_N_PCA,   GRU_K_LAG,   is_linear=False)
    X_lstm_full,  Y_lstm_full  = build_dayX_decoder_data(train_df, pca_model, LSTM_N_PCA,  LSTM_K_LAG,  is_linear=False)
    X_lin_full,   Y_lin_full   = build_dayX_decoder_data(train_df, pca_model, LINEAR_N_PCA,LINEAR_K_LAG,is_linear=True)
    X_ligru_full, Y_ligru_full = build_dayX_decoder_data(train_df, pca_model, LIGRU_N_PCA, LIGRU_K_LAG, is_linear=False)

    # If CROSSVAL is disabled, we do 1 pass only. If enabled, do CROSSVAL_RUNS times.
    n_cv_runs = CROSSVAL_RUNS if CROSSVAL_ENABLE else 1

    # We'll hold the final arrays of shape (n_cv_runs, #days) for each model
    # after we do day-by-day evaluation
    all_gru_vafs   = np.zeros((n_cv_runs, len(test_days))) * np.nan
    all_lstm_vafs  = np.zeros((n_cv_runs, len(test_days))) * np.nan
    all_lin_vafs   = np.zeros((n_cv_runs, len(test_days))) * np.nan
    all_ligru_vafs = np.zeros((n_cv_runs, len(test_days))) * np.nan

    #  Function to "evaluate day" exactly as original
    #  We'll param it with the 4 models we currently have.
    def evaluate_day(day_df, day_label, 
                     gru_model, lstm_model, linear_model, ligru_model,
                     day_pca_model):
        V_day0_full = None
        V_dayD_full = None
        if REDUCTION_METHOD.upper() == "PCA" and RECALC_PCA_EACH_DAY:
            day_z_for_pca = gather_day_spike_data_for_pca(day_df)
            if day_z_for_pca.shape[0] > 0:
                from sklearn.decomposition import PCA
                new_pca = PCA(n_components=max_dim, random_state=SEED)
                new_pca.fit(day_z_for_pca)
                day_pca_model = new_pca

        if day_pca_model is not None and global_pca_model is not None:
            V_day0_full = global_pca_model.components_.T
            V_dayD_full = day_pca_model.components_.T

        if TRAIN_TRIAL_BASED:
            X_all_list, Y_all_list, _ = build_trial_based_dataset(
                day_df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            if len(X_all_list) == 0:
                print(f"[WARNING] Day={day_label} => no trials found (trial-based).")
                return np.nan, np.nan, np.nan, np.nan

            def decode_trial(X_trial, n_pca, seq_len, model, is_linear=False):
                sm = smooth_spike_data(X_trial, bin_size=BIN_SIZE*BIN_FACTOR, smoothing_length=SMOOTHING_LENGTH)
                if APPLY_ZSCORE:
                    z = safe_zscore(sm, axis=0)
                else:
                    z = sm

                if day_pca_model is not None:
                    if (REALIGN_PCA_TO_DAY0
                        and (V_day0_full is not None)
                        and (V_dayD_full is not None)
                        and (day_pca_model is not global_pca_model)):
                        V_day0_k = V_day0_full[:, :n_pca]
                        V_dayD_k = V_dayD_full[:, :n_pca]
                        R = compute_alignment_matrix(V_dayD_k, V_day0_k)
                        z_aligned = (z @ V_dayD_k) @ R
                        X_pca = z_aligned
                    else:
                        full_proj = day_pca_model.transform(z)
                        X_pca = full_proj[:, :n_pca]
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
                        outp = outp.cpu().numpy()
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
                    decode_trial(X_trial, GRU_N_PCA, GRU_K_LAG, gru_model,   False)
                )
                lstm_preds_by_trial.append(
                    decode_trial(X_trial, LSTM_N_PCA, LSTM_K_LAG, lstm_model, False)
                )
                lin_preds_by_trial.append(
                    decode_trial(X_trial, LINEAR_N_PCA, LINEAR_K_LAG, linear_model, True)
                )
                ligru_preds_by_trial.append(
                    decode_trial(X_trial, LIGRU_N_PCA, LIGRU_K_LAG, ligru_model, False)
                )

            Y_all_flat = np.concatenate(Y_all_list, axis=0) if Y_all_list else np.array([])
            gru_preds  = np.concatenate(gru_preds_by_trial, axis=0) if gru_preds_by_trial else np.array([])
            if gru_preds.size and Y_all_flat.size:
                g_vafs = compute_multichannel_vaf(Y_all_flat, gru_preds)
                g_vaf  = np.nanmean(g_vafs)
            else:
                g_vaf = np.nan

            lstm_preds = np.concatenate(lstm_preds_by_trial, axis=0) if lstm_preds_by_trial else np.array([])
            if lstm_preds.size and Y_all_flat.size:
                l_vafs = compute_multichannel_vaf(Y_all_flat, lstm_preds)
                l_vaf  = np.nanmean(l_vafs)
            else:
                l_vaf = np.nan

            lin_preds  = np.concatenate(lin_preds_by_trial, axis=0) if lin_preds_by_trial else np.array([])
            if lin_preds.size and Y_all_flat.size:
                n_vafs = compute_multichannel_vaf(Y_all_flat, lin_preds)
                n_vaf  = np.nanmean(n_vafs)
            else:
                n_vaf = np.nan

            ligru_preds = np.concatenate(ligru_preds_by_trial, axis=0) if ligru_preds_by_trial else np.array([])
            if ligru_preds.size and Y_all_flat.size:
                li_vafs= compute_multichannel_vaf(Y_all_flat, ligru_preds)
                li_vaf = np.nanmean(li_vafs)
            else:
                li_vaf = np.nan

            return g_vaf, l_vaf, n_vaf, li_vaf
        else:
            big_spike_arr, big_emg_arr = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
            if big_spike_arr.shape[0] == 0:
                return np.nan, np.nan, np.nan, np.nan

            if day_pca_model is not None:
                dayD_latents = day_pca_model.transform(big_spike_arr)
                if (REALIGN_PCA_TO_DAY0
                    and (V_day0_full is not None)
                    and (V_dayD_full is not None)
                    and (day_pca_model is not global_pca_model)):
                    R = compute_alignment_matrix(V_dayD_full, V_day0_full)
                    day0_proj = (dayD_latents @ R)
                else:
                    day0_proj = dayD_latents
            else:
                day0_proj = big_spike_arr

            # GRU
            Xg_te = day0_proj[:, :GRU_N_PCA]
            Yg_te = big_emg_arr
            X_seq, Y_seq = create_rnn_dataset_continuous(Xg_te, Yg_te, GRU_K_LAG)
            gru_preds, gru_true = [], []
            gru_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    bx = X_seq[i:i+BATCH_SIZE]
                    by = Y_seq[i:i+BATCH_SIZE]
                    out = gru_model(torch.tensor(bx).to(DEVICE))
                    gru_preds.append(out.cpu().numpy())
                    gru_true.append(by)
            if gru_preds:
                gru_preds = np.concatenate(gru_preds, axis=0)
                gru_true  = np.concatenate(gru_true,  axis=0)
                vafs_g    = compute_multichannel_vaf(gru_true, gru_preds)
                gf        = np.nanmean(vafs_g)
            else:
                gf = np.nan

            # LSTM
            Xl_te = day0_proj[:, :LSTM_N_PCA]
            Yl_te = big_emg_arr
            X_seq, Y_seq = create_rnn_dataset_continuous(Xl_te, Yl_te, LSTM_K_LAG)
            lstm_preds, lstm_true = [], []
            lstm_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    bx = X_seq[i:i+BATCH_SIZE]
                    by = Y_seq[i:i+BATCH_SIZE]
                    out = lstm_model(torch.tensor(bx).to(DEVICE))
                    lstm_preds.append(out.cpu().numpy())
                    lstm_true.append(by)
            if lstm_preds:
                lstm_preds = np.concatenate(lstm_preds, axis=0)
                lstm_true  = np.concatenate(lstm_true, axis=0)
                vafs_l     = compute_multichannel_vaf(lstm_true, lstm_preds)
                lf         = np.nanmean(vafs_l)
            else:
                lf = np.nan

            # Linear
            Xn_te = day0_proj[:, :LINEAR_N_PCA]
            Yn_te = big_emg_arr
            X_seq, Y_seq = create_linear_dataset_continuous(Xn_te, Yn_te, LINEAR_K_LAG)
            lin_preds, lin_true = [], []
            linear_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    bx = X_seq[i:i+BATCH_SIZE]
                    by = Y_seq[i:i+BATCH_SIZE]
                    out = linear_model(torch.tensor(bx).to(DEVICE))
                    lin_preds.append(out.cpu().numpy())
                    lin_true.append(by)
            if lin_preds:
                lin_preds = np.concatenate(lin_preds, axis=0)
                lin_true  = np.concatenate(lin_true, axis=0)
                vafs_n    = compute_multichannel_vaf(lin_true, lin_preds)
                nf        = np.nanmean(vafs_n)
            else:
                nf = np.nan

            # LiGRU
            Xli_te = day0_proj[:, :LIGRU_N_PCA]
            Yli_te = big_emg_arr
            X_seq, Y_seq = create_rnn_dataset_continuous(Xli_te, Yli_te, LIGRU_K_LAG)
            ligru_preds, ligru_true = [], []
            ligru_model.eval()
            with torch.no_grad():
                for i in range(0, len(X_seq), BATCH_SIZE):
                    bx = X_seq[i:i+BATCH_SIZE]
                    by = Y_seq[i:i+BATCH_SIZE]
                    out = ligru_model(torch.tensor(bx).to(DEVICE))
                    ligru_preds.append(out.cpu().numpy())
                    ligru_true.append(by)
            if ligru_preds:
                ligru_preds = np.concatenate(ligru_preds, axis=0)
                ligru_true  = np.concatenate(ligru_true, axis=0)
                vafs_li     = compute_multichannel_vaf(ligru_true, ligru_preds)
                li_f        = np.nanmean(vafs_li)
            else:
                li_f = np.nan

            return gf, lf, nf, li_f


    # -------------------------------------------------------------------------
    # Now do cross-validation if CROSSVAL_ENABLE is True, or single pass if not.
    # -------------------------------------------------------------------------
    for cv_iter in range(n_cv_runs):
        print(f"\n===== CROSS-VALIDATION ROUND {cv_iter+1}/{n_cv_runs} =====")

        # If not CROSSVAL_ENABLE, we simply skip any splitting and use the entire day0 set
        if CROSSVAL_ENABLE:
            # Let's do a random 75/25 split on day0 data for each decoder set
            # (We do a row-based or trial-based approach, depending on continuous vs trial).
            # For simplicity, we assume we can just do row-based splitting for the continuous sets.
            # For trial-based, you'd do a split on the list of trials. We'll do row-based for demonstration.
            def split_data(X, Y, train_frac=0.75):
                n_items = X.shape[0]
                idx_tr, idx_te = random_split_indices(n_items, train_frac)
                X_tr = X[idx_tr]
                Y_tr = Y[idx_tr]
                X_te = X[idx_te]
                Y_te = Y[idx_te]
                return (X_tr, Y_tr), (X_te, Y_te)

            # We'll keep the "test set" for day0 unused or we can see the "day0 holdout performance" if desired
            (Xg_tr, Yg_tr), (Xg_val, Yg_val) = split_data(X_gru_full,   Y_gru_full)
            (Xl_tr, Yl_tr), (Xl_val, Yl_val) = split_data(X_lstm_full,  Y_lstm_full)
            (Xn_tr, Yn_tr), (Xn_val, Yn_val) = split_data(X_lin_full,   Y_lin_full)
            (Xli_tr,Yli_tr),(Xli_val,Yli_val)= split_data(X_ligru_full, Y_ligru_full)

            # Re-init models
            gru_model    = GRUDecoder(GRU_N_PCA,    GRU_HIDDEN_DIM,   n_emg_channels).to(DEVICE)
            lstm_model   = LSTMDecoder(LSTM_N_PCA,  LSTM_HIDDEN_DIM,  n_emg_channels).to(DEVICE)
            linear_model = LinearLagDecoder(LINEAR_K_LAG*LINEAR_N_PCA, LINEAR_HIDDEN_DIM, n_emg_channels).to(DEVICE)
            ligru_model  = LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_emg_channels).to(DEVICE)

            # Train on 75% day0
            sets_dict = {
                'gru':   (Xg_tr, Yg_tr),
                'lstm':  (Xl_tr, Yl_tr),
                'linear':(Xn_tr, Yn_tr),
                'ligru': (Xli_tr,Yli_tr),
            }
            model_dict = {
                'gru': gru_model,
                'lstm': lstm_model,
                'linear': linear_model,
                'ligru': ligru_model,
            }

            seed_dict  = DECODER_SEEDS
            (gru_model, lstm_model, linear_model, ligru_model) = train_on_day0_once(sets_dict, model_dict, seed_dict)
        else:
            # Single pass => use entire day0
            # Re-init models
            gru_model    = GRUDecoder(GRU_N_PCA,    GRU_HIDDEN_DIM,   n_emg_channels).to(DEVICE)
            lstm_model   = LSTMDecoder(LSTM_N_PCA,  LSTM_HIDDEN_DIM,  n_emg_channels).to(DEVICE)
            linear_model = LinearLagDecoder(LINEAR_K_LAG*LINEAR_N_PCA, LINEAR_HIDDEN_DIM, n_emg_channels).to(DEVICE)
            ligru_model  = LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_emg_channels).to(DEVICE)

            sets_dict = {
                'gru':   (X_gru_full, Y_gru_full),
                'lstm':  (X_lstm_full, Y_lstm_full),
                'linear':(X_lin_full, Y_lin_full),
                'ligru': (X_ligru_full,Y_ligru_full),
            }
            model_dict = {
                'gru': gru_model,
                'lstm': lstm_model,
                'linear': linear_model,
                'ligru': ligru_model,
            }
            seed_dict  = DECODER_SEEDS
            (gru_model, lstm_model, linear_model, ligru_model) = train_on_day0_once(sets_dict, model_dict, seed_dict)

        # Evaluate each day
        for d_i, d_val in enumerate(test_days):
            if d_val == day0:
                # For day0, evaluate on the held-out validation splits.
                g_vaf = evaluate_on_split(gru_model, Xg_val, Yg_val, GRU_K_LAG, is_linear=False)
                l_vaf = evaluate_on_split(lstm_model, Xl_val, Yl_val, LSTM_K_LAG, is_linear=False)
                n_vaf = evaluate_on_split(linear_model, Xn_val, Yn_val, LINEAR_K_LAG, is_linear=True)
                li_vaf = evaluate_on_split(ligru_model, Xli_val, Yli_val, LIGRU_K_LAG, is_linear=False)
            else:
                day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
                g_vaf, l_vaf, n_vaf, li_vaf = evaluate_day(day_df, d_val, 
                                                        gru_model, lstm_model, linear_model, ligru_model,
                                                        global_pca_model)
            all_gru_vafs[cv_iter, d_i]   = g_vaf
            all_lstm_vafs[cv_iter, d_i]  = l_vaf
            all_lin_vafs[cv_iter, d_i]   = n_vaf
            all_ligru_vafs[cv_iter, d_i] = li_vaf
            # print quick result
            print(f"   -> Day={d_val}: GRU={g_vaf:.3f}, LSTM={l_vaf:.3f}, Lin={n_vaf:.3f}, LiGRU={li_vaf:.3f}")

    ###########################################################################
    # Save crossval results (optional)
    ###########################################################################
    if SAVE_RESULTS_PKL is not None:
        results_dict = {
            'test_days': test_days,
            'gru_vafs': all_gru_vafs,   # shape (n_cv_runs, n_days)
            'lstm_vafs': all_lstm_vafs,
            'lin_vafs': all_lin_vafs,
            'ligru_vafs': all_ligru_vafs,
        }
        pd.to_pickle(results_dict, SAVE_RESULTS_PKL)
        print(f"[INFO] Crossval results saved to {SAVE_RESULTS_PKL}")

    ###########################################################################
    # Original "final figure" (Mean across folds if crossval?)
    ###########################################################################
    # We'll plot the average across crossval (axis=0)
    mean_gru = np.nanmean(all_gru_vafs, axis=0)
    mean_lstm= np.nanmean(all_lstm_vafs, axis=0)
    mean_lin = np.nanmean(all_lin_vafs,  axis=0)
    mean_li  = np.nanmean(all_ligru_vafs,axis=0)

    # Also store std
    std_gru = np.nanstd(all_gru_vafs, axis=0)
    std_lstm= np.nanstd(all_lstm_vafs, axis=0)
    std_lin = np.nanstd(all_lin_vafs,  axis=0)
    std_li  = np.nanstd(all_ligru_vafs,axis=0)

    # Convert days to integer differences
    # day_nums = np.arange(len(test_days)) # or actual date differences
    base_day = test_days[0]
    day_nums = [(d.date() - base_day.date()).days for d in test_days]

    # Plot the original style with lines for each model
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day_nums, mean_gru,   marker='o', label="GRU")
    ax.plot(day_nums, mean_lstm,  marker='o', label="LSTM")
    ax.plot(day_nums, mean_lin,   marker='o', label="Linear")
    ax.plot(day_nums, mean_li,    marker='o', label="LiGRU")

    ax.set_xlabel("Days from day0 (int)")
    ax.set_ylabel("VAF (mean across CV & EMG channels)")
    ax.set_title("VAF over Days (Mean of CV runs)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('day_evo_recalc_Jango.png', dpi=700)
    plt.show()

    ###########################################################################
    # Additional Figures:
    #   1) Relative VAF Loss vs Days (mean±std shading)
    #   2) Cumulative sum of VAF Loss
    #   3) Boxplot or scatter+box for each day
    ###########################################################################
    # 1) Compute day0-based loss: loss = day0 - dayD
    # We'll do it for each CV run, each day
    gru_loss   = all_gru_vafs[:,:,] * 0.0  # same shape, fill in
    lstm_loss  = all_lstm_vafs[:,:,] * 0.0
    lin_loss   = all_lin_vafs[:,:,] * 0.0
    ligru_loss = all_ligru_vafs[:,:,] * 0.0

    for i_cv in range(n_cv_runs):
        day0_gru = all_gru_vafs[i_cv, 0]
        day0_lstm= all_lstm_vafs[i_cv, 0]
        day0_lin = all_lin_vafs[i_cv, 0]
        day0_li  = all_ligru_vafs[i_cv, 0]
        for d_i in range(len(test_days)):
            gru_loss[i_cv, d_i]   = day0_gru  - all_gru_vafs[i_cv, d_i]
            lstm_loss[i_cv, d_i]  = day0_lstm - all_lstm_vafs[i_cv, d_i]
            lin_loss[i_cv, d_i]   = day0_lin  - all_lin_vafs[i_cv, d_i]
            ligru_loss[i_cv, d_i] = day0_li   - all_ligru_vafs[i_cv, d_i]

    # mean+std across crossval
    m_gru_loss = np.nanmean(gru_loss, axis=0)
    s_gru_loss = np.nanstd(gru_loss, axis=0)
    m_lstm_loss= np.nanmean(lstm_loss, axis=0)
    s_lstm_loss= np.nanstd(lstm_loss, axis=0)
    m_lin_loss = np.nanmean(lin_loss, axis=0)
    s_lin_loss = np.nanstd(lin_loss, axis=0)
    m_li_loss  = np.nanmean(ligru_loss,axis=0)
    s_li_loss  = np.nanstd(ligru_loss,axis=0)

    # Figure: Relative VAF Loss vs days
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.plot(day_nums, m_gru_loss,   '-o', label='GRU')
    ax2.fill_between(day_nums, m_gru_loss - s_gru_loss, m_gru_loss + s_gru_loss, alpha=0.2)

    ax2.plot(day_nums, m_lstm_loss,  '-o', label='LSTM')
    ax2.fill_between(day_nums, m_lstm_loss - s_lstm_loss, m_lstm_loss + s_lstm_loss, alpha=0.2)

    ax2.plot(day_nums, m_lin_loss,   '-o', label='Linear')
    ax2.fill_between(day_nums, m_lin_loss - s_lin_loss, m_lin_loss + s_lin_loss, alpha=0.2)

    ax2.plot(day_nums, m_li_loss,    '-o', label='LiGRU')
    ax2.fill_between(day_nums, m_li_loss - s_li_loss, m_li_loss + s_li_loss, alpha=0.2)

    ax2.set_xlabel("Days from day0")
    ax2.set_ylabel("Relative VAF Loss (day0 - dayX)")
    ax2.set_title("Relative VAF Loss vs. Days (Mean ± Std across CV)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("relative_vaf_loss_recalc.png", dpi=700)
    plt.show()

    # 2) Cumulative sum
    # We'll do it for each CV run, then average
    csum_gru = np.cumsum(gru_loss, axis=1)   # shape (n_cv, n_days)
    csum_lstm= np.cumsum(lstm_loss, axis=1)
    csum_lin = np.cumsum(lin_loss, axis=1)
    csum_li  = np.cumsum(ligru_loss, axis=1)

    m_gru_csum = np.nanmean(csum_gru, axis=0)
    s_gru_csum = np.nanstd(csum_gru, axis=0)
    m_lstm_csum= np.nanmean(csum_lstm, axis=0)
    s_lstm_csum= np.nanstd(csum_lstm, axis=0)
    m_lin_csum = np.nanmean(csum_lin, axis=0)
    s_lin_csum = np.nanstd(csum_lin, axis=0)
    m_li_csum  = np.nanmean(csum_li, axis=0)
    s_li_csum  = np.nanstd(csum_li, axis=0)

    fig3, ax3 = plt.subplots(figsize=(7,5))
    ax3.plot(day_nums, m_gru_csum,   '-o', label='GRU')
    ax3.fill_between(day_nums, m_gru_csum - s_gru_csum, m_gru_csum + s_gru_csum, alpha=0.2)

    ax3.plot(day_nums, m_lstm_csum,  '-o', label='LSTM')
    ax3.fill_between(day_nums, m_lstm_csum - s_lstm_csum, m_lstm_csum + s_lstm_csum, alpha=0.2)

    ax3.plot(day_nums, m_lin_csum,   '-o', label='Linear')
    ax3.fill_between(day_nums, m_lin_csum - s_lin_csum, m_lin_csum + s_lin_csum, alpha=0.2)

    ax3.plot(day_nums, m_li_csum,    '-o', label='LiGRU')
    ax3.fill_between(day_nums, m_li_csum - s_li_csum, m_li_csum + s_li_csum, alpha=0.2)

    ax3.set_xlabel("Days from day0")
    ax3.set_ylabel("Cumulative Relative VAF Loss")
    ax3.set_title("Cumulative VAF Loss (Mean ± Std across CV)")
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig("cumulative_vaf_loss_recalc.png", dpi=700)
    plt.show()

    # 3) Boxplot/Scatter for each day+decoder
    # We'll produce a dataframe for easy boxplot in seaborn or pandas
    import seaborn as sns

    # Gather all crossval runs, each day
    # Flatten each model into a list of (decoder, day, vaf)
    # day will be day_nums[i], or the actual i, whichever
    day_labels = [ (d.date() - base_day.date()).days for d in test_days ]
    box_data = []
    decoders = ["GRU","LSTM","Linear","LiGRU"]
    for i_cv in range(n_cv_runs):
        for i_day, d_lab in enumerate(day_labels):
            box_data.append(["GRU",   d_lab, all_gru_vafs[i_cv, i_day]])
            box_data.append(["LSTM",  d_lab, all_lstm_vafs[i_cv, i_day]])
            box_data.append(["Linear",d_lab, all_lin_vafs[i_cv, i_day]])
            box_data.append(["LiGRU", d_lab, all_ligru_vafs[i_cv, i_day]])

    df_box = pd.DataFrame(box_data, columns=["Decoder","Day","VAF"])

    fig4, ax4 = plt.subplots(figsize=(9,6))
    sns.boxplot(data=df_box, x="Day", y="VAF", hue="Decoder", ax=ax4, whis=[5,95])
    # optionally add scatter/swarm
    sns.stripplot(data=df_box, x="Day", y="VAF", hue="Decoder", dodge=True, ax=ax4,
                  alpha=0.5, color="black")
    ax4.legend_.remove()
    handles, labels = ax4.get_legend_handles_labels()
    unique_handles = handles[:4]
    unique_labels  = labels[:4]
    ax4.legend(unique_handles, unique_labels, title="Decoder", loc="upper right",
           frameon=True)
    ax4.set_title("Boxplot of VAF per Day/Decoder (with scatter overlay)")
    plt.tight_layout()
    plt.savefig("boxplot_vaf_recalc.png", dpi=700)
    plt.show()

if __name__ == "__main__":
    main()
