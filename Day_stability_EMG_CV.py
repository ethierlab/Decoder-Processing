import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
#                               CONFIG
###############################################################################

COMBINED_PICKLE_FILE = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015/combined.pkl"
)
SAVE_RESULTS_PKL = 'crossval_results_dimred.pkl'
SEED = 42
BIN_FACTOR = 20
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
SAMPLING_RATE = 1000
CROSSVAL_RUNS = 20
TRAIN_FRAC = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECODER_SEEDS = {"gru": 42, "lstm": 4242, "linear": 424242, "ligru": 12345}

# Dims
GRU_N_PCA    = 16
LSTM_N_PCA   = 16
LINEAR_N_PCA = 18
LIGRU_N_PCA  = 14
GRU_HIDDEN_DIM    = 17
GRU_K_LAG         = 12
LSTM_HIDDEN_DIM   = 18
LSTM_K_LAG        = 10
LINEAR_HIDDEN_DIM = 64
LINEAR_K_LAG      = 16
LIGRU_HIDDEN_DIM  = 5
LIGRU_K_LAG       = 16
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001

###############################################################################
#                               HELPERS
###############################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def butter_lowpass(data, fs, order=4):
    nyq = 0.5 * fs
    norm = 5.0 / nyq
    b, a = butter(order, norm, btype='low', analog=False)
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

def build_continuous_dataset(df, bin_factor, bin_size, smoothing_length):
    all_spike_list, all_emg_list = [], []
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        if emg_val is None:
            continue
        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0:
            continue
        spk_arr = ds_spike_df.values
        if isinstance(ds_emg, pd.DataFrame):
            e_arr = ds_emg.values
        else:
            e_arr = np.array(ds_emg)
        eff_fs = SAMPLING_RATE // bin_factor
        e_arr  = butter_lowpass(e_arr, eff_fs)
        sm = smooth_spike_data(spk_arr, bin_size*bin_factor, smoothing_length)
        all_spike_list.append(sm)
        all_emg_list.append(np.abs(e_arr))
    if len(all_spike_list) == 0:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(all_spike_list, axis=0), np.concatenate(all_emg_list, axis=0)

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
#   MODEL DEFINITIONS (GRU, LSTM, LINEAR, LIGRU)
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
#   DIM RED FUNCTION (PCA or UMAP)
###############################################################################
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        model = PCA(n_components=n_components, random_state=seed)
        model.fit(data)
        return model
    elif method.upper() == "UMAP":
        model = umap.UMAP(n_components=n_components, random_state=seed)
        model.fit(data)
        return model
    else:
        raise ValueError(f"Unknown dim. reduction: {method}")

def transform_dimred(model, data, method):
    if method.upper() in ("PCA", "UMAP"):
        return model.transform(data)
    else:
        raise ValueError(f"Unknown dim. reduction: {method}")

###############################################################################
#   VAF + STATISTICS
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

###############################################################################
#   MAIN PIPELINE
###############################################################################

def random_split_indices(n_items, train_frac=0.75):
    all_indices = np.arange(n_items)
    np.random.shuffle(all_indices)
    cutoff = int(train_frac * n_items)
    train_idx = all_indices[:cutoff]
    test_idx  = all_indices[cutoff:]
    return train_idx, test_idx

def train_model(model, X_train, Y_train):
    set_seed(SEED)
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        for Xb, Yb in dl:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X, Y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            bx = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
        vafs = compute_multichannel_vaf(Y, preds)
        return np.nanmean(vafs), vafs
    else:
        return np.nan, np.full((Y.shape[1],), np.nan)

def main():
    set_seed(SEED)
    print(f"[INFO] Using device: {DEVICE}")

    print(f"[INFO] Loading combined DataFrame from '{COMBINED_PICKLE_FILE}' ...")
    combined_df = pd.read_pickle(COMBINED_PICKLE_FILE)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!")
        return

    day0 = unique_days[0]
    test_days = [d for d in unique_days]

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

    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)
    max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)

    # For statistics
    results = []

    # Try both dim. reduction methods
    for dim_red_method in ["PCA", "UMAP"]:
        print(f"\n[INFO] === Running for {dim_red_method} ===")
        day0_spike, day0_emg = build_continuous_dataset(train_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
        dimred_model_day0 = get_dimred_model(day0_spike, dim_red_method, max_dim, SEED)
        z0 = transform_dimred(dimred_model_day0, day0_spike, dim_red_method)

        # Create all datasets once
        X_gru_full,   Y_gru_full   = create_rnn_dataset_continuous(z0[:, :GRU_N_PCA],   day0_emg,   GRU_K_LAG)
        X_lstm_full,  Y_lstm_full  = create_rnn_dataset_continuous(z0[:, :LSTM_N_PCA],  day0_emg,   LSTM_K_LAG)
        X_lin_full,   Y_lin_full   = create_linear_dataset_continuous(z0[:, :LINEAR_N_PCA], day0_emg, LINEAR_K_LAG)
        X_ligru_full, Y_ligru_full = create_rnn_dataset_continuous(z0[:, :LIGRU_N_PCA], day0_emg,   LIGRU_K_LAG)

        for cv_fold in range(CROSSVAL_RUNS):
            # Split
            def split_data(X, Y, train_frac=TRAIN_FRAC):
                n_items = X.shape[0]
                idx_tr, idx_te = random_split_indices(n_items, train_frac)
                X_tr = X[idx_tr]
                Y_tr = Y[idx_tr]
                X_te = X[idx_te]
                Y_te = Y[idx_te]
                return (X_tr, Y_tr), (X_te, Y_te)

            (Xg_tr, Yg_tr), (Xg_val, Yg_val) = split_data(X_gru_full,   Y_gru_full)
            (Xl_tr, Yl_tr), (Xl_val, Yl_val) = split_data(X_lstm_full,  Y_lstm_full)
            (Xn_tr, Yn_tr), (Xn_val, Yn_val) = split_data(X_lin_full,   Y_lin_full)
            (Xli_tr,Yli_tr),(Xli_val,Yli_val)= split_data(X_ligru_full, Y_ligru_full)

            # Train
            gru_model    = train_model(GRUDecoder(GRU_N_PCA, GRU_HIDDEN_DIM, n_emg_channels).to(DEVICE), Xg_tr, Yg_tr)
            lstm_model   = train_model(LSTMDecoder(LSTM_N_PCA, LSTM_HIDDEN_DIM, n_emg_channels).to(DEVICE), Xl_tr, Yl_tr)
            linear_model = train_model(LinearLagDecoder(LINEAR_K_LAG*LINEAR_N_PCA, LINEAR_HIDDEN_DIM, n_emg_channels).to(DEVICE), Xn_tr, Yn_tr)
            ligru_model  = train_model(LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_emg_channels).to(DEVICE), Xli_tr, Yli_tr)

            # Evaluate all test days
            for day_i, d_val in enumerate(test_days):
                day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
                spike, emg = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                if spike.shape[0] == 0:
                    continue

                # Fit dimred model for this day, transform
                dimred_model_dayX = get_dimred_model(spike, dim_red_method, max_dim, SEED)
                zx = transform_dimred(dimred_model_dayX, spike, dim_red_method)

                # Alignment (works well for PCA; for UMAP, it is a control)
                V_day0_full = z0.T
                V_dayX_full = zx.T
                # Keep only the matching dimensions for each decoder
                # Alignment is done per decoder for proper dimension
                for decoder, n_dim, model, seq_len, X_test, Y_test in [
                    ('GRU', GRU_N_PCA, gru_model, GRU_K_LAG, zx[:, :GRU_N_PCA], emg),
                    ('LSTM', LSTM_N_PCA, lstm_model, LSTM_K_LAG, zx[:, :LSTM_N_PCA], emg),
                    ('Linear', LINEAR_N_PCA, linear_model, LINEAR_K_LAG, zx[:, :LINEAR_N_PCA], emg),
                    ('LiGRU', LIGRU_N_PCA, ligru_model, LIGRU_K_LAG, zx[:, :LIGRU_N_PCA], emg),
                ]:
                    # Realignment: forcibly apply R = pinv(V_dayX_k) @ V_day0_k
                    # (works well for PCA, not justified for UMAP, but applied for comparison)
                    V_day0_k = V_day0_full[:n_dim, :].T
                    V_dayX_k = V_dayX_full[:n_dim, :].T
                    try:
                        R = pinv(V_dayX_k) @ V_day0_k
                        zx_aligned = (X_test @ R)
                    except Exception as e:
                        zx_aligned = X_test  # fallback

                    # Build test set
                    if decoder == 'Linear':
                        X_seq, Y_seq = create_linear_dataset_continuous(zx_aligned, Y_test, seq_len)
                    else:
                        X_seq, Y_seq = create_rnn_dataset_continuous(zx_aligned, Y_test, seq_len)
                    if X_seq.shape[0] == 0: continue

                    vaf, vaf_ch = evaluate_model(model, X_seq, Y_seq)
                    for ch_idx, vaf_single in enumerate(vaf_ch):
                        results.append({
                            "day": d_val,
                            "day_int": (d_val - day0).days,
                            "cv": cv_fold,
                            "decoder": decoder,
                            "dim_red": dim_red_method,
                            "emg_channel": ch_idx,
                            "vaf": vaf_single
                        })


                print(f" [cv={cv_fold+1:02d}] {dim_red_method:4} | day={str(d_val.date())} | done")

    # Final DataFrame
    df_results = pd.DataFrame(results)
    pd.to_pickle(df_results, SAVE_RESULTS_PKL)
    print(f"\n[INFO] Saved all results to {SAVE_RESULTS_PKL}")

    # Example plots/statistics
    print("[INFO] Example plot: VAF per day for each decoder/dim_red method.")
    for dim_red in ["PCA", "UMAP"]:
        plt.figure(figsize=(10,6))
        for dec in ["GRU","LSTM","Linear","LiGRU"]:
            sub = df_results[(df_results['decoder']==dec)&(df_results['dim_red']==dim_red)]
            means = sub.groupby("day_int")["vaf"].mean()
            stds = sub.groupby("day_int")["vaf"].std()
            plt.errorbar(means.index, means.values, yerr=stds.values, label=f"{dec} ({dim_red})")
        plt.legend()
        plt.xlabel("Days from day0")
        plt.ylabel("Mean VAF (crossval)")
        plt.title(f"VAF per day/decoder ({dim_red})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Example: boxplot of all runs
    sns.boxplot(data=df_results, x="day_int", y="vaf", hue="dim_red")
    plt.title("VAF across days (all decoders pooled)")
    plt.show()

if __name__ == "__main__":
    main()
