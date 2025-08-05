import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from numpy.linalg import pinv

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import umap
import pickle

# ============================================
# Config global décoder
# ============================================
DECODER_CONFIG = {
    "GRU":    {"N_PCA": 32, "HIDDEN_DIM": 96, "K_LAG": 25, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 200, "BATCH_SIZE": 64},
    "LSTM":   {"N_PCA": 24, "HIDDEN_DIM": 128, "K_LAG": 25, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 300, "BATCH_SIZE": 64},
    "LIN":    {"N_PCA": 32, "HIDDEN_DIM": 64, "K_LAG": 16, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 100, "BATCH_SIZE": 64},
    "LiGRU":  {"N_PCA": 32, "HIDDEN_DIM": 5,  "K_LAG": 16, "LEARNING_RATE": 0.001, "NUM_EPOCHS": 200, "BATCH_SIZE": 64},
}

# ============================================
# Utils : seed, model, VAF, etc.
# ============================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# ============================================
# RNN/Linear architectures
# ============================================
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

# ============================================
# Cross-validation split for day0
# ============================================
def get_day0_cv_splits(day0_df, n_folds, fold, seed):
    """Returns (train_idx, val_idx) on the trials/rows of day0_df for this fold."""
    n_trials = day0_df.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(n_trials)))
    train_idx, val_idx = splits[fold]
    return train_idx, val_idx

# ============================================
# PCA/UMAP gathering (per day)
# ============================================
def gather_day_spike_data_for_pca(day_df, downsample_spike_and_emg, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, APPLY_ZSCORE):
    all_spikes = []
    for idx, row in day_df.iterrows():
        spike_df = row["spike_counts"]
        emg_val = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        ds_spike_df, _ = downsample_spike_and_emg(spike_df, emg_val, BIN_FACTOR)
        if ds_spike_df.shape[0] == 0:
            continue
        sm = smooth_spike_data(
            ds_spike_df.values,
            bin_size=BIN_SIZE * BIN_FACTOR,
            smoothing_length=SMOOTHING_LENGTH
        )
        if APPLY_ZSCORE:
            z = safe_zscore(sm, axis=0)
            all_spikes.append(z)
        else:
            all_spikes.append(sm)
    if len(all_spikes) == 0:
        return np.empty((0, 0))
    return np.concatenate(all_spikes, axis=0)

# ============================================
# Rest of the helpers (downsampling, smoothing, etc.)
# ============================================
from scipy.ndimage import gaussian_filter1d

def downsample_spike_and_emg(spike_df, emg_data, bin_factor=10):
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

def build_continuous_dataset(df, bin_factor, bin_size, smoothing_length, APPLY_ZSCORE):
    all_spike_list = []
    all_emg_list = []
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val = row["EMG"]
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
        sm = smooth_spike_data(spk_arr, bin_size * bin_factor, smoothing_length)
        if APPLY_ZSCORE:
            final_spikes = safe_zscore(sm, axis=0)
        else:
            final_spikes = sm
        smoothed_emg = e_arr
        all_spike_list.append(final_spikes)
        all_emg_list.append(smoothed_emg)
    if len(all_spike_list) == 0:
        return np.empty((0,)), np.empty((0,))
    big_spike_arr = np.concatenate(all_spike_list, axis=0)
    big_emg_arr   = np.concatenate(all_emg_list, axis=0)
    return big_spike_arr, big_emg_arr

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

def compute_alignment_matrix(V_dayD, V_day0):
    return pinv(V_dayD) @ V_day0

# ============================================
# Training and evaluation functions
# ============================================
def train_decoder(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else float("nan")

def evaluate_decoder(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            out = model(Xb)
            preds.append(out.cpu().numpy())
            targets.append(Yb.numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
    else:
        preds = np.empty((0,))
        targets = np.empty((0,))
    return preds, targets

# ============================================
# MAIN (Argparse, job control)
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_pickle_file", type=str, required=True)
    parser.add_argument("--reduction_method", type=str, default="PCA", choices=["PCA", "UMAP"])
    parser.add_argument("--realignment_mode", type=str, default="naif", choices=["naif", "recalcule", "realign"])
    parser.add_argument("--decoder", type=str, default="GRU", choices=["GRU", "LSTM", "LIN", "LiGRU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./results/")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BIN_FACTOR = 20
    BIN_SIZE = 0.001
    SMOOTHING_LENGTH = 0.05
    APPLY_ZSCORE = False

    # 1. Load data
    df = pd.read_pickle(args.combined_pickle_file)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    unique_days = sorted(df["date"].unique())
    if len(unique_days) == 0:
        raise RuntimeError("No days found in combined_df!")
    day0 = unique_days[0]

    # 2. Cross-validation split sur les essais/trials du jour 0
    day0_df = df[df["date"] == day0].reset_index(drop=True)
    train_idx, val_idx = get_day0_cv_splits(day0_df, args.n_folds, args.fold, args.seed)
    train_day0 = day0_df.iloc[train_idx].reset_index(drop=True)
    val_day0   = day0_df.iloc[val_idx].reset_index(drop=True)

    # 3. Détermination des paramètres du décodeur choisi
    config = DECODER_CONFIG[args.decoder]
    n_pca = config["N_PCA"]
    hidden_dim = config["HIDDEN_DIM"]
    k_lag = config["K_LAG"]
    learning_rate = config["LEARNING_RATE"]
    num_epochs = config["NUM_EPOCHS"]
    batch_size = config["BATCH_SIZE"]

    # 4. Détecter nombre de canaux EMG
    n_emg_channels = 0
    for _, row in df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]
                break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]
                break
    if n_emg_channels == 0:
        raise RuntimeError("Could not detect EMG channels.")

    # 5. Fit reduction (PCA ou UMAP) sur le split d'entraînement de day0 (toujours)
    X_pca_train, _ = build_continuous_dataset(train_day0, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, APPLY_ZSCORE)
    if args.reduction_method.upper() == "PCA":
        reducer = PCA(n_components=max([v["N_PCA"] for v in DECODER_CONFIG.values()]), random_state=args.seed)
    else:
        reducer = umap.UMAP(n_components=max([v["N_PCA"] for v in DECODER_CONFIG.values()]), random_state=args.seed)
    reducer.fit(X_pca_train)
    V_train_full = reducer.components_.T if hasattr(reducer, "components_") else None

    # 6. Construction dataset pour ce décodeur et ce split
    X_train_full, Y_train_full = build_continuous_dataset(train_day0, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, APPLY_ZSCORE)
    X_val_full, Y_val_full = build_continuous_dataset(val_day0, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, APPLY_ZSCORE)
    X_train = reducer.transform(X_train_full)[:, :n_pca]
    X_val = reducer.transform(X_val_full)[:, :n_pca]

    # Séquences pour RNN/LSTM/LiGRU ou fenêtre pour LIN
    if args.decoder == "LIN":
        X_train_seq, Y_train_seq = create_linear_dataset_continuous(X_train, Y_train_full, k_lag)
        X_val_seq, Y_val_seq = create_linear_dataset_continuous(X_val, Y_val_full, k_lag)
    else:
        X_train_seq, Y_train_seq = create_rnn_dataset_continuous(X_train, Y_train_full, k_lag)
        X_val_seq, Y_val_seq = create_rnn_dataset_continuous(X_val, Y_val_full, k_lag)

    # Torch datasets/loaders
    ds_train = TensorDataset(torch.tensor(X_train_seq), torch.tensor(Y_train_seq))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ds_val = TensorDataset(torch.tensor(X_val_seq), torch.tensor(Y_val_seq))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # 7. Init modèle
    if args.decoder == "GRU":
        model = GRUDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
    elif args.decoder == "LSTM":
        model = LSTMDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
    elif args.decoder == "LIN":
        model = LinearLagDecoder(k_lag * n_pca, hidden_dim, n_emg_channels).to(DEVICE)
    elif args.decoder == "LiGRU":
        model = LiGRUDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
    else:
        raise RuntimeError(f"Decoder {args.decoder} not recognized")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 8. Train sur day0 split
    for ep in range(1, num_epochs+1):
        train_loss = train_decoder(model, dl_train, optimizer, criterion, DEVICE)
        if ep % 50 == 0 or ep == 1 or ep == num_epochs:
            print(f"Epoch {ep}/{num_epochs} - Loss: {train_loss:.4f}", flush=True)

    # 9. Évaluation sur day0 val
    preds_val, targets_val = evaluate_decoder(model, dl_val, DEVICE)
    vaf_val = np.nanmean(compute_multichannel_vaf(targets_val, preds_val))

    # 10. Test sur tous les jours (selon la logique du mode de realignement)
    results_per_day = []
    for day in unique_days:
        day_df = df[df["date"] == day].reset_index(drop=True)
        X_day, Y_day = build_continuous_dataset(day_df, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, APPLY_ZSCORE)
        # Choix du reducer à appliquer :
        # - Naif: utiliser reducer (fit sur train)
        # - Recalcule: fit sur X_day (test), appliquer à X_day
        # - Realign: fit sur X_day, puis aligner vers subspace train
        if args.realignment_mode == "naif":
            X_proj = reducer.transform(X_day)[:, :n_pca]
        else:
            # On refit la réduction sur le test set du jour en cours
            if args.reduction_method.upper() == "PCA":
                reducer_test = PCA(n_components=max([v["N_PCA"] for v in DECODER_CONFIG.values()]), random_state=args.seed)
            else:
                reducer_test = umap.UMAP(n_components=max([v["N_PCA"] for v in DECODER_CONFIG.values()]), random_state=args.seed)
            reducer_test.fit(X_day)
            V_test_full = reducer_test.components_.T if hasattr(reducer_test, "components_") else None
            if args.realignment_mode == "recalcule":
                X_proj = reducer_test.transform(X_day)[:, :n_pca]
            elif args.realignment_mode == "realign":
                # aligner vers train
                if hasattr(reducer_test, "components_") and hasattr(reducer, "components_"):
                    V_train_k = V_train_full[:, :n_pca]
                    V_test_k  = V_test_full[:, :n_pca]
                    R = compute_alignment_matrix(V_test_k, V_train_k)
                    X_proj = (reducer_test.transform(X_day)[:, :n_pca]) @ R
                else:
                    # UMAP n'a pas components_; impossible d'aligner proprement, fallback
                    X_proj = reducer_test.transform(X_day)[:, :n_pca]
            else:
                raise RuntimeError("Invalid realignment_mode")
        # Format pour modèle
        if args.decoder == "LIN":
            X_seq, Y_seq = create_linear_dataset_continuous(X_proj, Y_day, k_lag)
        else:
            X_seq, Y_seq = create_rnn_dataset_continuous(X_proj, Y_day, k_lag)
        ds_day = TensorDataset(torch.tensor(X_seq), torch.tensor(Y_seq))
        dl_day = DataLoader(ds_day, batch_size=batch_size, shuffle=False)
        preds, targets = evaluate_decoder(model, dl_day, DEVICE)
        vaf_day = np.nanmean(compute_multichannel_vaf(targets, preds))
        results_per_day.append({
            "date": day,
            "vaf": vaf_day,
            "preds": preds,  # optionally comment out to save space
            "targets": targets  # optionally comment out to save space
        })
        print(f"Day {day} - VAF={vaf_day:.3f}")

    # 11. Save ALL results for this fold/config to file (pickle)
    output = {
        "args": vars(args),
        "config": config,
        "val_fold": {
            "val_indices": val_idx.tolist(),
            "vaf": vaf_val,
            "preds": preds_val,
            "targets": targets_val,
        },
        "results_per_day": results_per_day,
        "decoder": args.decoder,
        "seed": args.seed,
        "reduction_method": args.reduction_method,
        "realignment_mode": args.realignment_mode,
        "n_folds": args.n_folds,
        "fold": args.fold,
    }
    out_fname = f"{args.decoder}_method{args.reduction_method}_align{args.realignment_mode}_fold{args.fold}_seed{args.seed}.pkl"
    with open(os.path.join(args.output_path, out_fname), "wb") as f:
        pickle.dump(output, f)

    print(f"Saved results: {os.path.join(args.output_path, out_fname)}")

if __name__ == "__main__":
    main()
