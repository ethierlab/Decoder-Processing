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

from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from random import sample

###############################################################################
#                           GLOBAL PARAMETERS
###############################################################################

COMBINED_PICKLE_FILE = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Spike_ISO_2012/combined.pkl"
)

SHOW_GRAPHS = True

REDUCTION_METHOD = "PCA"
SEED = 42

# Binning / Downsampling
BIN_FACTOR = 10
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
PRE_TRIAL = 1.0
POST_TRIAL = 4.0
SAMPLING_RATE = 1000

# PCA dims per decoder
GRU_N_PCA    = 14
LSTM_N_PCA   = 14
LINEAR_N_PCA = 18
LIGRU_N_PCA  = 14

# RNN/Linear hidden dims & lag
GRU_HIDDEN_DIM    = 5
GRU_K_LAG         = 16

LSTM_HIDDEN_DIM   = 16
LSTM_K_LAG        = 16

LINEAR_HIDDEN_DIM = 64
LINEAR_K_LAG      = 16

LIGRU_HIDDEN_DIM  = 5
LIGRU_K_LAG       = 16

NUM_EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
#   CHOOSE TRIAL-BASED OR CONTINUOUS
###############################################################################
TRAIN_TRIAL_BASED = False  # If True => trial slicing; If False => continuous approach

###############################################################################
#   Option to re-calc PCA each day
###############################################################################
RECALC_PCA_EACH_DAY = False

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
#          2) DOWNSAMPLING SPIKE & FORCE
###############################################################################
def downsample_spike_and_force(spike_df, force_data, bin_factor=10):
    """
    Summation of spikes + average of force every bin_factor samples.
    """
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, force_data

    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor

    spk_arr = spike_df.values[: T_new * bin_factor, :]
    spk_arr = spk_arr.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(spk_arr, columns=spike_df.columns)

    if isinstance(force_data, pd.DataFrame):
        f_arr = force_data.values
        col_names = force_data.columns
    else:
        f_arr = np.array(force_data)
    if f_arr.shape[0] < bin_factor:
        return ds_spike_df, force_data

    f_arr = f_arr[: T_new * bin_factor, ...]
    if f_arr.ndim == 1:
        f_arr = f_arr.reshape(T_new, bin_factor).mean(axis=1)
        ds_force = pd.Series(f_arr)
    elif f_arr.ndim == 2:
        f_arr = f_arr.reshape(T_new, bin_factor, f_arr.shape[1]).mean(axis=1)
        ds_force = (
            pd.DataFrame(f_arr, columns=col_names)
            if isinstance(force_data, pd.DataFrame)
            else f_arr
        )
    else:
        ds_force = force_data
    return ds_spike_df, ds_force

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
    """
    For each row => slice around trial_start_time => return lists X_all, Y_all, T_all
    """
    X_all, Y_all, T_all = [], [], []

    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        force_val= row["force"]
        time_frame = row["time_frame"]
        trial_starts= row["trial_start_time"]

        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            print(f"[WARNING] row {idx} => empty spike_df, skipping.")
            continue
        if force_val is None:
            print(f"[WARNING] row {idx} => force is None, skipping.")
            continue
        if not isinstance(time_frame, np.ndarray) or len(time_frame) == 0:
            print(f"[WARNING] row {idx} => time_frame empty, skipping.")
            continue
        if not isinstance(trial_starts, np.ndarray) or len(trial_starts) == 0:
            print(f"[WARNING] row {idx} => no trial_starts, skipping.")
            continue

        if bin_factor > 1:
            ds_spike_df, ds_force = downsample_spike_and_force(spike_df, force_val, bin_factor)
            T_old = len(time_frame)
            T_new = T_old // bin_factor
            ds_time = time_frame[: (T_new * bin_factor)]
            ds_time = ds_time.reshape(T_new, bin_factor).mean(axis=1)
        else:
            ds_spike_df = spike_df
            ds_force    = force_val
            ds_time     = time_frame

        effective_fs = fs // bin_factor

        for ts in trial_starts:
            idx_center = np.argmin(np.abs(ds_time - ts))
            start_idx = max(0, idx_center - int(pre * effective_fs))
            end_idx   = min(len(ds_time), idx_center + int(post * effective_fs))
            if end_idx <= start_idx:
                continue

            X_win = ds_spike_df.values[start_idx:end_idx, :]
            if isinstance(ds_force, pd.DataFrame):
                if "x" in ds_force.columns:
                    Y_arr = ds_force["x"].values
                else:
                    Y_arr = ds_force.values[:, 0]
            else:
                Y_arr = np.array(ds_force)
            Y_win = Y_arr[start_idx:end_idx]
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
    Merges all rows => 1 big chunk => downsample => smooth => returns big_spike_smooth, big_force.
    """
    all_spike_list = []
    all_force_list = []

    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        force_val= row["force"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            print(f"[WARNING] row {idx} => empty spike_df, skipping.")
            continue
        if force_val is None:
            print(f"[WARNING] row {idx} => force is None, skipping.")
            continue

        ds_spike_df, ds_force = downsample_spike_and_force(spike_df, force_val, bin_factor)
        if ds_spike_df.shape[0] == 0:
            continue

        if isinstance(ds_force, pd.DataFrame):
            if "x" in ds_force.columns:
                force_arr = ds_force["x"].values
            else:
                force_arr = ds_force.values[:,0]
        else:
            force_arr = np.array(ds_force)

        all_spike_list.append(ds_spike_df.values)
        all_force_list.append(force_arr)

    if len(all_spike_list) == 0:
        return np.empty((0,)), np.empty((0,))
    big_spike_arr = np.concatenate(all_spike_list, axis=0)
    big_force_arr = np.concatenate(all_force_list, axis=0)

    big_spike_smooth = smooth_spike_data(big_spike_arr, bin_size, smoothing_length)
    return big_spike_smooth, big_force_arr

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
            Y_out.append(Y[t])
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
            Y_out.append(Y[t])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

###############################################################################
#       5b) CONTINUOUS RNN/LINEAR
###############################################################################
def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0,))
    X_out, Y_out = [], []
    T_i = X_arr.shape[0]
    for t in range(seq_len, T_i):
        X_out.append(X_arr[t-seq_len:t, :])
        Y_out.append(Y_arr[t])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len * X_arr.shape[1])), np.empty((0,))
    X_out, Y_out = [], []
    T_i = X_arr.shape[0]
    for t in range(seq_len, T_i):
        window = X_arr[t-seq_len:t, :].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[t])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

###############################################################################
#       6) MODEL DEFINITIONS (GRU, LSTM, LINEAR, LIGRU)
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 1)
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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, 1)
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
def compute_vaf(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    var_resid = np.nanvar(y_true - y_pred)
    var_true  = np.nanvar(y_true)
    return 1.0 - (var_resid / var_true) if var_true > 1e-12 else np.nan

###############################################################################
#   Plot random segments for continuous
###############################################################################
def plot_continuous_random_segments(day_label, all_true, pred_gru, pred_lstm, pred_lin, pred_ligru, seg_len=400):
    """
    all_true/pred_* each shape (T,)
    seg_len => # of samples per segment
    """
    T_total = len(all_true)
    if T_total < seg_len:
        print(f"[WARNING] day={day_label} => not enough length to plot segments.")
        return
    num_segments = 6
    possible_starts = np.arange(T_total - seg_len)
    chosen_starts = np.random.choice(possible_starts, size=min(num_segments, len(possible_starts)), replace=False)

    fig, axes = plt.subplots(3, 2, figsize=(15,12))
    axes = axes.flatten()
    for i, start_idx in enumerate(chosen_starts):
        ax = axes[i]
        end_idx = start_idx + seg_len
        actual_seg = all_true[start_idx:end_idx]
        gru_seg    = pred_gru[start_idx:end_idx]
        lstm_seg   = pred_lstm[start_idx:end_idx]
        lin_seg    = pred_lin[start_idx:end_idx]
        ligru_seg  = pred_ligru[start_idx:end_idx]

        vaf_gru   = compute_vaf(actual_seg, gru_seg)
        vaf_lstm  = compute_vaf(actual_seg, lstm_seg)
        vaf_lin   = compute_vaf(actual_seg, lin_seg)
        vaf_ligru = compute_vaf(actual_seg, ligru_seg)

        time_axis = np.arange(seg_len)
        ax.plot(time_axis, actual_seg, label="Actual")
        ax.plot(time_axis, gru_seg,   label=f"GRU(VAF={vaf_gru:.2f})",   linestyle='--')
        ax.plot(time_axis, lstm_seg,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        ax.plot(time_axis, lin_seg,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        ax.plot(time_axis, ligru_seg, label=f"LiGRU(VAF={vaf_ligru:.2f})",linestyle='-.')
        ax.set_title(f"Day={day_label}, segment={start_idx}:{end_idx}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Force (z-score)")
    plt.tight_layout()
    plt.show()

###############################################################################
#   Plot 6 random trials for trial-based
###############################################################################
def plot_trial_random_samples(day_label, X_all_list, Y_all_list,
                              gru_preds_by_trial, lstm_preds_by_trial,
                              lin_preds_by_trial, ligru_preds_by_trial):
    """
    Plots 6 random trials from the trial-based approach.
    Each trial is shape (T_i,).
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
        actual = Y_all_list[trial_idx]
        t_len = len(actual)
        time_axis = np.arange(t_len)

        gru_pred  = gru_preds_by_trial[trial_idx]
        vaf_gru   = compute_vaf(actual, gru_pred)

        lstm_pred = lstm_preds_by_trial[trial_idx]
        vaf_lstm  = compute_vaf(actual, lstm_pred)

        lin_pred  = lin_preds_by_trial[trial_idx]
        vaf_lin   = compute_vaf(actual, lin_pred)

        ligru_pred= ligru_preds_by_trial[trial_idx]
        vaf_ligru = compute_vaf(actual, ligru_pred)

        ax.plot(time_axis, actual, label="Actual")
        ax.plot(time_axis, gru_pred,   label=f"GRU(VAF={vaf_gru:.2f})",   linestyle='--')
        ax.plot(time_axis, lstm_pred,  label=f"LSTM(VAF={vaf_lstm:.2f})", linestyle='--')
        ax.plot(time_axis, lin_pred,   label=f"Lin(VAF={vaf_lin:.2f})",   linestyle=':')
        ax.plot(time_axis, ligru_pred, label=f"LiGRU(VAF={vaf_ligru:.2f})", linestyle='-.')
        ax.set_title(f"Day={day_label}, trial={trial_idx}")
        ax.set_xlabel("Time (bins)")
        ax.set_ylabel("Force (z-scored)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

###############################################################################
#   Helper: gather all day-level spikes (trial or continuous) for PCA
###############################################################################
def gather_day_spike_data_for_pca(day_df):
    """
    If you are in TRIAL mode, gather *all trials’ spike data*
    If in CONTINUOUS mode, gather data by the continuous method.

    Returns a single 2D numpy array [T x Channels] after smoothing + zscoring,
    so we can fit a PCA if needed.
    """
    if TRAIN_TRIAL_BASED:
        # For trial-based, we gather each row, downsample, just like build_trial_based_dataset,
        # but we only want to accumulate spikes in a single array to do PCA.
        all_spikes = []
        for idx, row in day_df.iterrows():
            spike_df = row["spike_counts"]
            if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
                continue
            ds_spike_df, _ = downsample_spike_and_force(spike_df, row["force"], BIN_FACTOR)
            if ds_spike_df.shape[0] == 0:
                continue
            # smooth + zscore
            sm = smooth_spike_data(ds_spike_df.values,
                                   bin_size=BIN_SIZE*BIN_FACTOR,
                                   smoothing_length=SMOOTHING_LENGTH)
            z  = safe_zscore(sm, axis=0)
            all_spikes.append(z)
        if len(all_spikes) == 0:
            return np.empty((0,0))
        return np.concatenate(all_spikes, axis=0)

    else:
        # For continuous approach:
        big_spike_smooth, _ = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
        if big_spike_smooth.shape[0] == 0:
            return np.empty((0,0))
        z  = safe_zscore(big_spike_smooth, axis=0)
        return z


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

    day1 = unique_days[0]
    test_days = [d for d in unique_days]  # We'll evaluate *all* days, including day1
    print("[DEBUG] unique_days =>", unique_days)
    print("[DEBUG] day1 =>", day1)

    #--- Build train_df as day1 data & gather PCA (unless we recalc each day).
    train_df = combined_df[combined_df["date"] == day1].reset_index(drop=True)
    print(f"[DEBUG] train_df shape={train_df.shape}")

    # Gather day1 spike data for PCA
    max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
    pca_model = None
    if REDUCTION_METHOD.upper() == "PCA":
        print(f"[DEBUG] Fitting PCA n_components={max_dim} on day1 ...")
        # gather day1 spikes
        day1_z = gather_day_spike_data_for_pca(train_df)
        if day1_z.shape[0] == 0:
            print("[ERROR] No valid spike data in day1 after smoothing/zscore!")
            return
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=max_dim, random_state=SEED)
        pca_model.fit(day1_z)
    else:
        print("[DEBUG] Not applying PCA (REDUCTION_METHOD != 'PCA')")

    ###########################################################################
    # 2) Build training set using day1
    ###########################################################################
    def build_dayX_decoder_data(df, day_pca_model, n_pca, seq_len, is_linear=False):
        """
        Build the final X, Y arrays for a single day (df).
        That day could be day1 (training) or dayN (if we wanted to re-train).
        But here we just use it for day1 training.
        """
        if TRAIN_TRIAL_BASED:
            # trial-based
            X_list, Y_list, _ = build_trial_based_dataset(
                df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            # transform each trial
            X_out, Y_out = [], []
            for X_trial, Y_trial in zip(X_list, Y_list):
                if X_trial.shape[0] == 0:
                    continue
                # smooth + zscore
                sm = smooth_spike_data(X_trial, bin_size=BIN_SIZE*BIN_FACTOR,
                                       smoothing_length=SMOOTHING_LENGTH)
                z  = safe_zscore(sm, axis=0)
                # PCA if specified
                if day_pca_model is not None:
                    full_proj = day_pca_model.transform(z)
                    X_pca = full_proj[:, :n_pca]
                else:
                    X_pca = z[:, :n_pca]
                X_out.append(X_pca)
                Y_out.append(Y_trial)

            if not is_linear:
                # RNN
                X_arr, Y_arr = create_rnn_dataset(X_out, Y_out, seq_len)
            else:
                # linear
                X_arr, Y_arr = create_linear_dataset(X_out, Y_out, seq_len)
            return X_arr, Y_arr

        else:
            # continuous
            big_spike_smooth, big_force_arr = build_continuous_dataset(
                df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
            )
            if big_spike_smooth.shape[0] == 0:
                return np.empty((0,0)), np.empty((0,))
            z  = safe_zscore(big_spike_smooth, axis=0)
            if day_pca_model is not None:
                z_full = day_pca_model.transform(z)
            else:
                z_full = z
            # slice the correct # of pca dims
            X_pca = z_full[:, :n_pca]
            if not is_linear:
                X_arr, Y_arr = create_rnn_dataset_continuous(X_pca, big_force_arr, seq_len)
            else:
                X_arr, Y_arr = create_linear_dataset_continuous(X_pca, big_force_arr, seq_len)
            return X_arr, Y_arr

    # Build training sets for day1
    X_gru_train,   Y_gru_train   = build_dayX_decoder_data(train_df, pca_model, GRU_N_PCA,   GRU_K_LAG,   is_linear=False)
    X_lstm_train,  Y_lstm_train  = build_dayX_decoder_data(train_df, pca_model, LSTM_N_PCA,  LSTM_K_LAG,  is_linear=False)
    X_lin_train,   Y_lin_train   = build_dayX_decoder_data(train_df, pca_model, LINEAR_N_PCA,LINEAR_K_LAG,is_linear=True)
    X_ligru_train, Y_ligru_train = build_dayX_decoder_data(train_df, pca_model, LIGRU_N_PCA, LIGRU_K_LAG, is_linear=False)

    print("[DEBUG] day1 shapes => GRU:", X_gru_train.shape, Y_gru_train.shape)
    print("[DEBUG] day1 shapes => LSTM:", X_lstm_train.shape, Y_lstm_train.shape)
    print("[DEBUG] day1 shapes => Linear:", X_lin_train.shape, Y_lin_train.shape)
    print("[DEBUG] day1 shapes => LiGRU:", X_ligru_train.shape, Y_ligru_train.shape)

    ds_gru   = TensorDataset(torch.tensor(X_gru_train),   torch.tensor(Y_gru_train).unsqueeze(-1))
    dl_gru   = DataLoader(ds_gru, batch_size=BATCH_SIZE, shuffle=True)
    ds_lstm  = TensorDataset(torch.tensor(X_lstm_train),  torch.tensor(Y_lstm_train).unsqueeze(-1))
    dl_lstm  = DataLoader(ds_lstm, batch_size=BATCH_SIZE, shuffle=True)
    ds_lin   = TensorDataset(torch.tensor(X_lin_train),   torch.tensor(Y_lin_train).unsqueeze(-1))
    dl_lin   = DataLoader(ds_lin, batch_size=BATCH_SIZE,  shuffle=True)
    ds_ligru = TensorDataset(torch.tensor(X_ligru_train), torch.tensor(Y_ligru_train).unsqueeze(-1))
    dl_ligru = DataLoader(ds_ligru, batch_size=BATCH_SIZE, shuffle=True)

    ###########################################################################
    # 3) Initialize and train models on day1
    ###########################################################################
    gru_model    = GRUDecoder(GRU_N_PCA,    GRU_HIDDEN_DIM).to(DEVICE)
    lstm_model   = LSTMDecoder(LSTM_N_PCA,  LSTM_HIDDEN_DIM).to(DEVICE)
    linear_model = LinearLagDecoder(LINEAR_K_LAG * LINEAR_N_PCA, LINEAR_HIDDEN_DIM).to(DEVICE)
    ligru_model  = LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM).to(DEVICE)

    gru_opt    = optim.Adam(gru_model.parameters(),    lr=LEARNING_RATE)
    lstm_opt   = optim.Adam(lstm_model.parameters(),   lr=LEARNING_RATE)
    lin_opt    = optim.Adam(linear_model.parameters(), lr=LEARNING_RATE)
    ligru_opt  = optim.Adam(ligru_model.parameters(),  lr=LEARNING_RATE)

    criterion = nn.MSELoss()

    print("[INFO] Training decoders on Day1 ...")
    for ep in range(1, NUM_EPOCHS + 1):
        loss_gru   = train_decoder(gru_model,    dl_gru,   gru_opt, criterion)
        loss_lstm  = train_decoder(lstm_model,   dl_lstm,  lstm_opt, criterion)
        loss_lin   = train_decoder(linear_model, dl_lin,   lin_opt, criterion)
        loss_ligru = train_decoder(ligru_model,  dl_ligru, ligru_opt, criterion)

        if ep % 10 == 0:
            print(f"Epoch {ep}/{NUM_EPOCHS} => "
                  f"GRU:{loss_gru:.4f}, LSTM:{loss_lstm:.4f}, "
                  f"Lin:{loss_lin:.4f}, LiGRU:{loss_ligru:.4f}")
    print("[INFO] Training complete.\n")

    ###########################################################################
    # 4) Evaluate each day. We store day => VAF results, then plot them.
    ###########################################################################
    results_days = []
    results_gru  = []
    results_lstm = []
    results_lin  = []
    results_ligru= []

    def evaluate_day(day_df, day_label):
        """
        Evaluate the decoders on a single day.
        If RECALC_PCA_EACH_DAY is True, we recalc PCA using that day’s entire data
        (and still feed it into the day1-trained model).
        Then compute VAF on that day.

        We also optionally do random plots.
        """
        # Possibly recalc PCA
        local_pca_model = pca_model
        if REDUCTION_METHOD.upper() == "PCA" and RECALC_PCA_EACH_DAY:
            day_z_for_pca = gather_day_spike_data_for_pca(day_df)
            if day_z_for_pca.shape[0] > 0:
                from sklearn.decomposition import PCA
                # fit new PCA up to max_dim
                new_pca = PCA(n_components=max_dim, random_state=SEED)
                new_pca.fit(day_z_for_pca)
                local_pca_model = new_pca

        if TRAIN_TRIAL_BASED:
            # TRIAL-BASED EVAL + PLOTTING
            X_all_list, Y_all_list, _ = build_trial_based_dataset(
                day_df, BIN_FACTOR, BIN_SIZE*BIN_FACTOR, SMOOTHING_LENGTH,
                PRE_TRIAL, POST_TRIAL, SAMPLING_RATE
            )
            if len(X_all_list) == 0:
                print(f"[WARNING] Day={day_label} => no trials found in trial-based approach.")
                return np.nan, np.nan, np.nan, np.nan

            def decode_trial(X_trial, n_pca, seq_len, model, is_linear=False):
                sm = smooth_spike_data(X_trial,
                                       bin_size=BIN_SIZE*BIN_FACTOR,
                                       smoothing_length=SMOOTHING_LENGTH)
                z  = safe_zscore(sm, axis=0)
                if local_pca_model is not None:
                    full_proj = local_pca_model.transform(z)
                    X_pca = full_proj[:, :n_pca]
                else:
                    X_pca = z[:, :n_pca]

                T_i = X_pca.shape[0]
                pred_out = np.full((T_i,), np.nan)
                if not is_linear:
                    # RNN
                    if T_i > seq_len:
                        X_lag = []
                        idx_list = []
                        for t in range(seq_len, T_i):
                            X_lag.append(X_pca[t-seq_len:t, :])
                            idx_list.append(t)
                        X_lag = np.array(X_lag, dtype=np.float32)
                        with torch.no_grad():
                            outp = model(torch.tensor(X_lag).to(DEVICE))
                        outp = outp.cpu().numpy().flatten()
                        for k, out_idx in enumerate(idx_list):
                            pred_out[out_idx] = outp[k]
                else:
                    # Linear
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
                        outp = outp.cpu().numpy().flatten()
                        for k, out_idx in enumerate(idx_list):
                            pred_out[out_idx] = outp[k]
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

            # Flatten => day-level VAF
            gru_all_true = np.concatenate(Y_all_list) if Y_all_list else np.array([])
            gru_all_pred = np.concatenate(gru_preds_by_trial) if gru_preds_by_trial else np.array([])
            gru_vaf_day  = compute_vaf(gru_all_true, gru_all_pred)

            lstm_all_true= np.concatenate(Y_all_list) if Y_all_list else np.array([])
            lstm_all_pred= np.concatenate(lstm_preds_by_trial) if lstm_preds_by_trial else np.array([])
            lstm_vaf_day = compute_vaf(lstm_all_true, lstm_all_pred)

            lin_all_true = np.concatenate(Y_all_list) if Y_all_list else np.array([])
            lin_all_pred = np.concatenate(lin_preds_by_trial) if lin_preds_by_trial else np.array([])
            lin_vaf_day  = compute_vaf(lin_all_true, lin_all_pred)

            ligru_all_true= np.concatenate(Y_all_list) if Y_all_list else np.array([])
            ligru_all_pred= np.concatenate(ligru_preds_by_trial) if ligru_preds_by_trial else np.array([])
            ligru_vaf_day = compute_vaf(ligru_all_true, ligru_all_pred)

            print(f"[RESULT] Day={day_label} => VAF: "
                  f"GRU={gru_vaf_day:.3f}, LSTM={lstm_vaf_day:.3f}, "
                  f"Linear={lin_vaf_day:.3f}, LiGRU={ligru_vaf_day:.3f}")

            # Optional plots
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
            # CONTINUOUS approach
            big_spike_smooth, big_force_arr = build_continuous_dataset(
                day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
            )
            if big_spike_smooth.shape[0] == 0:
                print(f"[WARNING] Day={day_label} => no data in continuous approach.")
                return np.nan, np.nan, np.nan, np.nan

            z  = safe_zscore(big_spike_smooth, axis=0)
            if local_pca_model is not None:
                z_full = local_pca_model.transform(z)
            else:
                z_full = z

            # Evaluate GRU
            X_gru_te, Y_gru_te = create_rnn_dataset_continuous(z_full[:, :GRU_N_PCA], big_force_arr, GRU_K_LAG)
            dl_gru_te = DataLoader(
                TensorDataset(torch.tensor(X_gru_te), torch.tensor(Y_gru_te).unsqueeze(-1)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            preds_list, ytrue_list = [], []
            gru_model.eval()
            with torch.no_grad():
                for Xb, Yb in dl_gru_te:
                    Xb = Xb.to(DEVICE)
                    out = gru_model(Xb)
                    preds_list.append(out.cpu().numpy().flatten())
                    ytrue_list.append(Yb.numpy().flatten())
            gru_all_pred = np.concatenate(preds_list) if preds_list else np.array([])
            gru_all_true = np.concatenate(ytrue_list) if ytrue_list else np.array([])
            gru_vaf_day  = compute_vaf(gru_all_true, gru_all_pred)

            # LSTM
            X_lstm_te, Y_lstm_te = create_rnn_dataset_continuous(z_full[:, :LSTM_N_PCA], big_force_arr, LSTM_K_LAG)
            dl_lstm_te = DataLoader(
                TensorDataset(torch.tensor(X_lstm_te), torch.tensor(Y_lstm_te).unsqueeze(-1)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            preds_list, ytrue_list = [], []
            lstm_model.eval()
            with torch.no_grad():
                for Xb, Yb in dl_lstm_te:
                    Xb = Xb.to(DEVICE)
                    out = lstm_model(Xb)
                    preds_list.append(out.cpu().numpy().flatten())
                    ytrue_list.append(Yb.numpy().flatten())
            lstm_all_pred = np.concatenate(preds_list) if preds_list else np.array([])
            lstm_all_true = np.concatenate(ytrue_list) if ytrue_list else np.array([])
            lstm_vaf_day  = compute_vaf(lstm_all_true, lstm_all_pred)

            # Linear
            X_lin_te, Y_lin_te = create_linear_dataset_continuous(z_full[:, :LINEAR_N_PCA],
                                                                  big_force_arr, LINEAR_K_LAG)
            dl_lin_te = DataLoader(
                TensorDataset(torch.tensor(X_lin_te), torch.tensor(Y_lin_te).unsqueeze(-1)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            preds_list, ytrue_list = [], []
            linear_model.eval()
            with torch.no_grad():
                for Xb, Yb in dl_lin_te:
                    Xb = Xb.to(DEVICE)
                    out = linear_model(Xb)
                    preds_list.append(out.cpu().numpy().flatten())
                    ytrue_list.append(Yb.numpy().flatten())
            lin_all_pred = np.concatenate(preds_list) if preds_list else np.array([])
            lin_all_true = np.concatenate(ytrue_list) if ytrue_list else np.array([])
            lin_vaf_day  = compute_vaf(lin_all_true, lin_all_pred)

            # LiGRU
            X_ligru_te, Y_ligru_te = create_rnn_dataset_continuous(z_full[:, :LIGRU_N_PCA],
                                                                   big_force_arr, LIGRU_K_LAG)
            dl_ligru_te = DataLoader(
                TensorDataset(torch.tensor(X_ligru_te), torch.tensor(Y_ligru_te).unsqueeze(-1)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            preds_list, ytrue_list = [], []
            ligru_model.eval()
            with torch.no_grad():
                for Xb, Yb in dl_ligru_te:
                    Xb = Xb.to(DEVICE)
                    out = ligru_model(Xb)
                    preds_list.append(out.cpu().numpy().flatten())
                    ytrue_list.append(Yb.numpy().flatten())
            ligru_all_pred = np.concatenate(preds_list) if preds_list else np.array([])
            ligru_all_true = np.concatenate(ytrue_list) if ytrue_list else np.array([])
            ligru_vaf_day  = compute_vaf(ligru_all_true, ligru_all_pred)

            print(f"[RESULT] Day={day_label} => VAF: "
                  f"GRU={gru_vaf_day:.3f}, LSTM={lstm_vaf_day:.3f}, "
                  f"Linear={lin_vaf_day:.3f}, LiGRU={ligru_vaf_day:.3f}")

            # Optional: plot random segments
            if SHOW_GRAPHS and len(gru_all_true) > 0:
                plot_continuous_random_segments(
                    day_label=day_label,
                    all_true=gru_all_true,
                    pred_gru=gru_all_pred,
                    pred_lstm=lstm_all_pred,
                    pred_lin=lin_all_pred,
                    pred_ligru=ligru_all_pred,
                    seg_len=400
                )

            return gru_vaf_day, lstm_vaf_day, lin_vaf_day, ligru_vaf_day

    # Evaluate each day in ascending order
    for d in test_days:
        day_df = combined_df[combined_df["date"] == d].reset_index(drop=True)
        gru_vaf, lstm_vaf, lin_vaf, ligru_vaf = evaluate_day(day_df, d)
        results_days.append(d)
        results_gru.append(gru_vaf)
        results_lstm.append(lstm_vaf)
        results_lin.append(lin_vaf)
        results_ligru.append(ligru_vaf)

    ###########################################################################
    # 5) Plot the VAF vs. Day (a linear plot with x-axis as the real Date)
    ###########################################################################
    fig, ax = plt.subplots(figsize=(8, 5))
    # Convert your dates to a numeric format for plotting
    x_vals = mdates.date2num(results_days)

    ax.plot(x_vals, results_gru,  marker='o', label="GRU")
    ax.plot(x_vals, results_lstm, marker='o', label="LSTM")
    ax.plot(x_vals, results_lin,  marker='o', label="Linear")
    ax.plot(x_vals, results_ligru,marker='o', label="LiGRU")

    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlabel("Day")
    ax.set_ylabel("VAF")
    ax.set_title("VAF over Days")
    ax.legend()
    plt.gcf().autofmt_xdate()
    if SHOW_GRAPHS:
        plt.savefig('day_evo.png',dpi=700)
        plt.show()

if __name__ == "__main__":
    main()
