import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from scipy.linalg import orthogonal_procrustes  # NEW: for UMAP alignment
from sklearn.decomposition import PCA
import umap
import warnings
import argparse

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")
###############################################################################
# CONFIG
###############################################################################

COMBINED_PICKLE_FILE = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015/combined.pkl"
)
SAVE_RESULTS_PKL = 'multi_day_align_results_cv.pkl'
SEED = 42
BIN_FACTOR = 20
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
SAMPLING_RATE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
N_FOLDS = 10  # <-- CROSSVAL : nombre de splits sur day0
ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}


###############################################################################
# HELPERS
###############################################################################

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_split_indices(n_items, train_frac=0.75, rng=None):
    all_indices = np.arange(n_items)
    if rng is None:
        rng = np.random
    rng.shuffle(all_indices)
    cutoff = int(train_frac * n_items)
    train_idx = all_indices[:cutoff]
    test_idx = all_indices[cutoff:]
    return train_idx, test_idx
    

def get_all_unit_names(combined_df):
    unit_set = set()
    for idx, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

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

def build_continuous_dataset(df, bin_factor, bin_size, smoothing_length, all_units=None):
    all_spike_list, all_emg_list = [], []
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        if emg_val is None:
            continue

        # Harmonize spike_df to have all units (missing units get 0)
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)

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
# MODELS
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
# DIM REDUCTION (PCA or UMAP)
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
# VAF + TRAIN/EVAL
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

def train_model(model, X_train, Y_train, num_epochs=200, lr=0.001):
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, num_epochs+1):
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

###############################################################################
# ALIGNMENT HELPERS
###############################################################################

def align_umap_procrustes(Zx: np.ndarray, Z0: np.ndarray) -> np.ndarray:
    """Align Zx (UMAP trained on dayX) onto Z0 (UMAP(day0).transform(dayX))
    using orthogonal Procrustes (rotation + global scale), with mean-centering.

    Returns Zx_aligned in the target coordinate system (approx Z0).
    """
    if Zx.shape != Z0.shape:
        raise ValueError(f"Procrustes requires same shape: got {Zx.shape} vs {Z0.shape}")
    # Mean-center
    Zx_mean = Zx.mean(axis=0, keepdims=True)
    Z0_mean = Z0.mean(axis=0, keepdims=True)
    Zx_c = Zx - Zx_mean
    Z0_c = Z0 - Z0_mean
    # Orthogonal Procrustes (R: rotation, s: global scale)
    R, s = orthogonal_procrustes(Zx_c, Z0_c)
    Zx_aligned = (Zx_c @ R) * s + Z0_mean
    return Zx_aligned

###############################################################################
# MAIN PIPELINE (day0 → all days, avec et sans alignement)
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, required=True, choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument('--dimred', type=str, default="PCA", choices=["PCA", "UMAP"]) 
    parser.add_argument('--crossval_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--train_frac', type=float, default=0.75)
    parser.add_argument('--combined_pickle', type=str, default="combined.pkl")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Hyperparams
    hp = ARCH_HYPERPARAMS[args.decoder]
    N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = (
        hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
    )

    # Data
    combined_df = pd.read_pickle(args.combined_pickle)
    ALL_UNITS = get_all_unit_names(combined_df)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!")
        return

    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)
    test_days = [d for d in unique_days]

    # Detect EMG channels
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

    # Build day0 (train)
    BIN_FACTOR = 20
    BIN_SIZE = 0.001
    SMOOTHING_LENGTH = 0.05
    SAMPLING_RATE = 1000

    day0_spike, day0_emg = build_continuous_dataset(train_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, all_units=ALL_UNITS)
    max_dim = N_PCA
    dimred_model_day0 = get_dimred_model(day0_spike, args.dimred, max_dim, args.seed)
    z0 = transform_dimred(dimred_model_day0, day0_spike, args.dimred)

    # Build datasets for decoder
    if args.decoder == "Linear":
        X_full, Y_full = create_linear_dataset_continuous(z0[:, :N_PCA], day0_emg, K_LAG)
    else:
        X_full, Y_full = create_rnn_dataset_continuous(z0[:, :N_PCA], day0_emg, K_LAG)

    # Cross-validation on day0
    results = []
    for fold in range(args.crossval_runs):
        rng = np.random.default_rng(args.seed + fold)
        idx = np.arange(len(X_full))
        rng.shuffle(idx)
        split = int(args.train_frac * len(idx))
        tr_idx, te_idx = idx[:split], idx[split:]

        X_tr, Y_tr = X_full[tr_idx], Y_full[tr_idx]
        X_te, Y_te = X_full[te_idx], Y_full[te_idx]

        # Model
        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LiGRU":
            model = LiGRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)

        train_model(model, X_tr, Y_tr, num_epochs=NUM_EPOCHS, lr=LR)
        vaf_te, vaf_ch_te = evaluate_model(model, X_te, Y_te)
        for ch_idx, vaf_single in enumerate(vaf_ch_te):
            results.append({
                "day": day0,
                "day_int": 0,
                "align": "crossval",
                "decoder": args.decoder,
                "dim_red": args.dimred,
                "fold": fold,
                "emg_channel": ch_idx,
                "vaf": vaf_single
            })

        # Test cross-days
        for day_i, d_val in enumerate(test_days):
            if d_val == day0:
                continue
            day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
            spike, emg = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, all_units=ALL_UNITS)
            if spike.shape[0] == 0:
                continue

            # Direct: project with day0 model
            zx_direct = transform_dimred(dimred_model_day0, spike, args.dimred)
            # DayX own model
            dimred_model_dayX = get_dimred_model(spike, args.dimred, N_PCA, args.seed)
            zx_dayX = transform_dimred(dimred_model_dayX, spike, args.dimred)

            for align_mode in ["direct", "aligned"]:
                if align_mode == "direct":
                    zx_test = zx_direct[:, :N_PCA]
                else:
                    if args.dimred == "PCA":
                        V_day0 = dimred_model_day0.components_[:N_PCA, :].T
                        V_dayX = dimred_model_dayX.components_[:N_PCA, :].T
                        try:
                            R = pinv(V_dayX) @ V_day0
                            zx_test = zx_dayX[:, :N_PCA] @ R
                        except Exception:
                            zx_test = zx_dayX[:, :N_PCA]
                    else:  # UMAP aligned via Procrustes onto zx_direct
                        try:
                            Z0 = zx_direct[:, :N_PCA]
                            Zx = zx_dayX[:, :N_PCA]
                            zx_test = align_umap_procrustes(Zx, Z0)
                        except Exception as e:
                            print(f"[WARN] UMAP Procrustes failed ({e}); falling back to unaligned dayX embedding.")
                            zx_test = zx_dayX[:, :N_PCA]

                if args.decoder == "Linear":
                    X_seq, Y_seq = create_linear_dataset_continuous(zx_test, emg, K_LAG)
                else:
                    X_seq, Y_seq = create_rnn_dataset_continuous(zx_test, emg, K_LAG)
                if X_seq.shape[0] == 0:
                    continue

                vaf, vaf_ch = evaluate_model(model, X_seq, Y_seq)
                for ch_idx, vaf_single in enumerate(vaf_ch):
                    results.append({
                        "day": d_val,
                        "day_int": (d_val - day0).days,
                        "align": align_mode,
                        "decoder": args.decoder,
                        "dim_red": args.dimred,
                        "fold": fold,
                        "emg_channel": ch_idx,
                        "vaf": vaf_single,
                        "mean_vaf": vaf
                    })
            print(f"[fold={fold}] {str(pd.to_datetime(d_val).date())} done.")

    # Save
    save_path = os.path.join(args.save_dir, f"crossday_results_{args.decoder}_{args.dimred}.pkl")
    pd.to_pickle(pd.DataFrame(results), save_path)
    print(f"\n[INFO] Saved all results to {save_path}")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from scipy.linalg import orthogonal_procrustes  # NEW: for UMAP alignment
from sklearn.decomposition import PCA
import umap
import warnings
import argparse

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")
###############################################################################
# CONFIG
###############################################################################

COMBINED_PICKLE_FILE = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015/combined.pkl"
)
SAVE_RESULTS_PKL = 'multi_day_align_results_cv.pkl'
SEED = 42
BIN_FACTOR = 20
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
SAMPLING_RATE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
N_FOLDS = 10  # <-- CROSSVAL : nombre de splits sur day0
ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}


###############################################################################
# HELPERS
###############################################################################

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_split_indices(n_items, train_frac=0.75, rng=None):
    all_indices = np.arange(n_items)
    if rng is None:
        rng = np.random
    rng.shuffle(all_indices)
    cutoff = int(train_frac * n_items)
    train_idx = all_indices[:cutoff]
    test_idx = all_indices[cutoff:]
    return train_idx, test_idx
    

def get_all_unit_names(combined_df):
    unit_set = set()
    for idx, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

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

def build_continuous_dataset(df, bin_factor, bin_size, smoothing_length, all_units=None):
    all_spike_list, all_emg_list = [], []
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        if emg_val is None:
            continue

        # Harmonize spike_df to have all units (missing units get 0)
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)

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
# MODELS
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
# DIM REDUCTION (PCA or UMAP)
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
# VAF + TRAIN/EVAL
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

def train_model(model, X_train, Y_train, num_epochs=200, lr=0.001):
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, num_epochs+1):
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

###############################################################################
# ALIGNMENT HELPERS
###############################################################################

def align_umap_procrustes(Zx: np.ndarray, Z0: np.ndarray) -> np.ndarray:
    """Align Zx (UMAP trained on dayX) onto Z0 (UMAP(day0).transform(dayX))
    using orthogonal Procrustes (rotation + global scale), with mean-centering.

    Returns Zx_aligned in the target coordinate system (approx Z0).
    """
    if Zx.shape != Z0.shape:
        raise ValueError(f"Procrustes requires same shape: got {Zx.shape} vs {Z0.shape}")
    # Mean-center
    Zx_mean = Zx.mean(axis=0, keepdims=True)
    Z0_mean = Z0.mean(axis=0, keepdims=True)
    Zx_c = Zx - Zx_mean
    Z0_c = Z0 - Z0_mean
    # Orthogonal Procrustes (R: rotation, s: global scale)
    R, s = orthogonal_procrustes(Zx_c, Z0_c)
    Zx_aligned = (Zx_c @ R) * s + Z0_mean
    return Zx_aligned

###############################################################################
# MAIN PIPELINE (day0 → all days, avec et sans alignement)
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, required=True, choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument('--dimred', type=str, default="PCA", choices=["PCA", "UMAP"]) 
    parser.add_argument('--crossval_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--train_frac', type=float, default=0.75)
    parser.add_argument('--combined_pickle', type=str, default="combined.pkl")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Hyperparams
    hp = ARCH_HYPERPARAMS[args.decoder]
    N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = (
        hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
    )

    # Data
    combined_df = pd.read_pickle(args.combined_pickle)
    ALL_UNITS = get_all_unit_names(combined_df)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!")
        return

    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)
    test_days = [d for d in unique_days]

    # Detect EMG channels
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

    # Build day0 (train)
    BIN_FACTOR = 20
    BIN_SIZE = 0.001
    SMOOTHING_LENGTH = 0.05
    SAMPLING_RATE = 1000

    day0_spike, day0_emg = build_continuous_dataset(train_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, all_units=ALL_UNITS)
    max_dim = N_PCA
    dimred_model_day0 = get_dimred_model(day0_spike, args.dimred, max_dim, args.seed)
    z0 = transform_dimred(dimred_model_day0, day0_spike, args.dimred)

    # Build datasets for decoder
    if args.decoder == "Linear":
        X_full, Y_full = create_linear_dataset_continuous(z0[:, :N_PCA], day0_emg, K_LAG)
    else:
        X_full, Y_full = create_rnn_dataset_continuous(z0[:, :N_PCA], day0_emg, K_LAG)

    # Cross-validation on day0
    results = []
    for fold in range(args.crossval_runs):
        rng = np.random.default_rng(args.seed + fold)
        idx = np.arange(len(X_full))
        rng.shuffle(idx)
        split = int(args.train_frac * len(idx))
        tr_idx, te_idx = idx[:split], idx[split:]

        X_tr, Y_tr = X_full[tr_idx], Y_full[tr_idx]
        X_te, Y_te = X_full[te_idx], Y_full[te_idx]

        # Model
        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LiGRU":
            model = LiGRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)

        train_model(model, X_tr, Y_tr, num_epochs=NUM_EPOCHS, lr=LR)
        vaf_te, vaf_ch_te = evaluate_model(model, X_te, Y_te)
        for ch_idx, vaf_single in enumerate(vaf_ch_te):
            results.append({
                "day": day0,
                "day_int": 0,
                "align": "crossval",
                "decoder": args.decoder,
                "dim_red": args.dimred,
                "fold": fold,
                "emg_channel": ch_idx,
                "vaf": vaf_single
            })

        # Test cross-days
        for day_i, d_val in enumerate(test_days):
            if d_val == day0:
                continue
            day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
            spike, emg = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, all_units=ALL_UNITS)
            if spike.shape[0] == 0:
                continue

            # Direct: project with day0 model
            zx_direct = transform_dimred(dimred_model_day0, spike, args.dimred)
            # DayX own model
            dimred_model_dayX = get_dimred_model(spike, args.dimred, N_PCA, args.seed)
            zx_dayX = transform_dimred(dimred_model_dayX, spike, args.dimred)

            for align_mode in ["direct", "aligned"]:
                if align_mode == "direct":
                    zx_test = zx_direct[:, :N_PCA]
                else:
                    if args.dimred == "PCA":
                        V_day0 = dimred_model_day0.components_[:N_PCA, :].T
                        V_dayX = dimred_model_dayX.components_[:N_PCA, :].T
                        try:
                            R = pinv(V_dayX) @ V_day0
                            zx_test = zx_dayX[:, :N_PCA] @ R
                        except Exception:
                            zx_test = zx_dayX[:, :N_PCA]
                    else:  # UMAP aligned via Procrustes onto zx_direct
                        try:
                            Z0 = zx_direct[:, :N_PCA]
                            Zx = zx_dayX[:, :N_PCA]
                            zx_test = align_umap_procrustes(Zx, Z0)
                        except Exception as e:
                            print(f"[WARN] UMAP Procrustes failed ({e}); falling back to unaligned dayX embedding.")
                            zx_test = zx_dayX[:, :N_PCA]

                if args.decoder == "Linear":
                    X_seq, Y_seq = create_linear_dataset_continuous(zx_test, emg, K_LAG)
                else:
                    X_seq, Y_seq = create_rnn_dataset_continuous(zx_test, emg, K_LAG)
                if X_seq.shape[0] == 0:
                    continue

                vaf, vaf_ch = evaluate_model(model, X_seq, Y_seq)
                for ch_idx, vaf_single in enumerate(vaf_ch):
                    results.append({
                        "day": d_val,
                        "day_int": (d_val - day0).days,
                        "align": align_mode,
                        "decoder": args.decoder,
                        "dim_red": args.dimred,
                        "fold": fold,
                        "emg_channel": ch_idx,
                        "vaf": vaf_single,
                        "mean_vaf": vaf
                    })
            print(f"[fold={fold}] {str(pd.to_datetime(d_val).date())} done.")

    # Save
    save_path = os.path.join(args.save_dir, f"crossday_results_{args.decoder}_{args.dimred}.pkl")
    pd.to_pickle(pd.DataFrame(results), save_path)
    print(f"\n[INFO] Saved all results to {save_path}")

if __name__ == "__main__":
    main()
