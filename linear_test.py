#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid (K-fold no-leak) with two decoders:
 - WienerCascadeEthier (FIR + quadratic static nonlinearity)
 - KalmanDecoder (state = EMG, observation = neural features)

Outputs:
  - gridsearch_results_{jobid}.pkl  (summary per config)
  - gridsearch_rows_{jobid}.pkl     (long-format per fold×channel×seed)
"""

import os, gc, time, argparse, warnings, pickle, itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

# ============================ CONFIG ============================
SEED = 42
BIN_FACTOR = 20           # 1 kHz -> /20 -> 50 Hz
BIN_SIZE = 0.001          # s, original bins (1ms). Effective bin after downsample = BIN_FACTOR * BIN_SIZE
SMOOTHING_LENGTH = 0.05   # s, Gaussian smoothing window length
SAMPLING_RATE = 1000      # Hz (original)
GAUSS_TRUNCATE = 4.0      # embargo covers ~truncate*sigma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True          # True = fast (AMP+TF32), False = strict reproducibility

# ============================ PERF / SEED ============================
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if PERF_MODE:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def auto_num_workers(default=8):
    try:
        n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if n > 0: return max(2, n - 1)
    except Exception:
        pass
    return default

def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)

def _dl_generator():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g

# ============================ DATA HELPERS ============================
def get_all_unit_names(combined_df: pd.DataFrame) -> List[str]:
    unit_set = set()
    for _, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

def get_emg_labels_from_df(df: pd.DataFrame, fallback_n: int) -> List[str]:
    for emg in df["EMG"]:
        if isinstance(emg, pd.DataFrame) and not emg.empty:
            return list(map(str, emg.columns))
    return [f"ch{c}" for c in range(fallback_n)]

def butter_lowpass(data, fs, order=4, cutoff_hz=5.0):
    nyq = 0.5 * fs
    norm = cutoff_hz / nyq
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

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    """Concat trials (downsampled) WITHOUT smoothing/filtering; return X_raw, Y_raw, cuts."""
    spikes_all, emg_all, lengths = [], [], []
    for _, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty: continue
        if emg_val is None: continue
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)
        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0: continue

        Xr = ds_spike_df.values.astype(np.float32)
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(np.float32)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)
        spikes_all.append(Xr); emg_all.append(Yr); lengths.append(len(Xr))

    if not spikes_all:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(spikes_all, axis=0), np.concatenate(emg_all, axis=0), cuts

def smooth_spike_data(x_2d, eff_bin, smoothing_length):
    # vectorized Gaussian across channels
    sigma = (smoothing_length / eff_bin) / 2.0
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    """Per-segment smoothing/filtering (no leakage across segments)."""
    eff_fs  = SAMPLING_RATE // bin_factor
    eff_bin = bin_factor * bin_size
    Xs = smooth_spike_data(Xseg, eff_bin, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_bin = bin_factor * bin_size
    return (smoothing_length / eff_bin) / 2.0

def embargo_bins(k_lag, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(k_lag, emb)

def time_kfold_splits(n_time, n_splits) -> List[Tuple[int,int]]:
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None: seg_len = end - start
    new_start = trim_left
    new_end   = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, start=0, end=None, stride=1):
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    if not idx:
        if is_linear:
            return np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32), np.empty((0, Y.shape[1]), dtype=np.float32)
        else:
            return np.empty((0, K_LAG, Z.shape[1]), dtype=np.float32), np.empty((0, Y.shape[1]), dtype=np.float32)
    if is_linear:
        X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0).astype(np.float32)
    else:
        X = np.stack([Z[t-K_LAG:t, :] for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx], axis=0).astype(np.float32)
    return X, Yb

def contiguous_blocks(n_time: int, cuts: List[int]) -> List[Tuple[int,int]]:
    """Convert cut indices into contiguous [start,end) blocks."""
    if n_time <= 0: return []
    points = [0] + sorted([c for c in cuts if 0 < c < n_time]) + [n_time]
    return [(points[i], points[i+1]) for i in range(len(points)-1)]

# ============================ MODELS ============================
class WienerCascadeEthier(nn.Module):
    """
    FIR multi-entrée -> non-linéarité quadratique statique (par canal EMG).
    Entrée attendue: (B, K*D) ou (B, K, D).
    y_d = c0_d + c1_d z_d + c2_d z_d^2
    """
    def __init__(self, input_dim: int, n_out: int, poly_order: int = 2, bias: bool = True):
        super().__init__()
        assert poly_order in (1, 2)
        self.poly_order = poly_order
        self.lin = nn.Linear(input_dim, n_out, bias=bias)
        self.c0 = nn.Parameter(torch.zeros(n_out))
        self.c1 = nn.Parameter(torch.ones(n_out))
        if self.poly_order == 2:
            self.c2 = nn.Parameter(0.1 * torch.ones(n_out))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        z = self.lin(x)  # (B, n_out)
        y = self.c0 + self.c1 * z
        if self.poly_order == 2:
            y = y + self.c2 * (z * z)
        return y

# ---- Kalman (NumPy) ----
class KFParams:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A, self.H, self.Q, self.R, self.x0, self.P0 = A, H, Q, R, x0, P0

class KalmanDecoder:
    """
    KF pour EMG (état) à partir des features neuronaux (observations).
    Apprentissage des paramètres par MCO sur TRAIN contigu (respect des cuts).
    """
    def __init__(self):
        self.params: KFParams | None = None

    @staticmethod
    def _fit_A(X_blocks: List[np.ndarray]) -> np.ndarray:
        # concat sans franchir de cuts pour la régression X_t ~ A X_{t-1}
        Xt_list, Xt1_list = [], []
        for X in X_blocks:
            if len(X) < 2: continue
            Xt_list.append(X[:-1]); Xt1_list.append(X[1:])
        if not Xt_list:  # fallback identité
            n_out = X_blocks[0].shape[1]
            return np.eye(n_out)
        Xt = np.vstack(Xt_list)   # (M, n_out)
        Xt1= np.vstack(Xt1_list)  # (M, n_out)
        # A^T = pinv(Xt) @ Xt1  -> A = Xt1^T @ pinv(Xt)^T
        A = (np.linalg.pinv(Xt) @ Xt1).T
        return A

    @staticmethod
    def _fit_H(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        # Z ≈ H X  -> H = Z X^+   (pas besoin de découper par cuts pour H)
        Xp = np.linalg.pinv(X)
        H = Z.T @ Xp.T
        return H

    def fit(self, Z_train: np.ndarray, Y_train: np.ndarray, cuts_train: List[int]):
        """
        Z_train: (T, n_feat), Y_train: (T, n_out), cuts_train: indices coupures [0..T]
        """
        # blocs contigus pour la dynamique d'état
        blocks = contiguous_blocks(len(Y_train), cuts_train)
        X_blocks = [Y_train[s:e] for (s,e) in blocks]

        # A
        A = self._fit_A(X_blocks)

        # H
        H = self._fit_H(Z_train, Y_train)

        # Q (état) : résidus intra-bloc uniquement
        W_list = []
        for X in X_blocks:
            if len(X) < 2: continue
            X_pred = X[:-1] @ A.T
            W_list.append(X[1:] - X_pred)
        if W_list:
            W = np.vstack(W_list)
            Q = np.cov(W.T) + 1e-6*np.eye(W.shape[1])
        else:
            Q = 1e-3*np.eye(Y_train.shape[1])

        # R (mesure)
        Z_pred = Y_train @ H.T
        V = Z_train - Z_pred
        R = np.cov(V.T) + 1e-6*np.eye(Z_train.shape[1])

        x0 = Y_train[0].copy()
        P0 = np.cov(Y_train.T) + 1e-3*np.eye(Y_train.shape[1])

        self.params = KFParams(A=A, H=H, Q=Q, R=R, x0=x0, P0=P0)

    def filter_block(self, Z: np.ndarray, x0: np.ndarray, P0: np.ndarray) -> np.ndarray:
        """Filtrage KF sur un bloc contigu d'observations Z."""
        A,H,Q,R = self.params.A, self.params.H, self.params.Q, self.params.R
        x, P = x0.copy(), P0.copy()
        nT, n_feat = Z.shape
        n_out = x.shape[0]
        X_hat = np.zeros((nT, n_out), dtype=float)
        AT, HT = A.T, H.T
        I = np.eye(n_out)
        for t in range(nT):
            # Predict
            x = A @ x
            P = A @ P @ AT + Q
            # Update
            z = Z[t]
            S = H @ P @ HT + R
            K = P @ HT @ np.linalg.inv(S)
            y = z - (H @ x)
            x = x + K @ y
            P = (I - K @ H) @ P
            X_hat[t] = x
        return X_hat, x, P

    def filter(self, Z: np.ndarray, cuts_val: List[int]) -> np.ndarray:
        """Filtre par blocs contigus si cuts_val non vide."""
        assert self.params is not None
        blocks = contiguous_blocks(len(Z), cuts_val)
        x, P = self.params.x0, self.params.P0
        out = np.zeros((len(Z), len(x)), dtype=float)
        for s,e in blocks:
            Xb, x, P = self.filter_block(Z[s:e], x, P)
            out[s:e] = Xb
        return out

# ============================ DIM-RED ============================
def fit_pca(X, n_components, seed):
    model = PCA(n_components=max(n_components, 2), random_state=seed)
    model.fit(X)
    return model

def pca_transform(model, X):
    return model.transform(X)

# ============================ METRICS + TRAIN/EVAL ============================
def eval_vaf_full(model, X_np, Y_np, batch_size, use_amp=True):
    if X_np.shape[0] == 0:
        return float("nan"), np.full((Y_np.shape[1],), np.nan, dtype=np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(DEVICE, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type=="cuda")):
                out = model(xb).cpu().numpy()
            preds.append(out)
    Yp = np.concatenate(preds, axis=0)
    vafs = []
    for ch in range(Y_np.shape[1]):
        yt, yp = Y_np[:, ch], Yp[:, ch]
        vt = np.var(yt)
        if vt < 1e-12: vafs.append(np.nan)
        else:          vafs.append(1.0 - np.var(yt - yp)/vt)
    vafs = np.asarray(vafs, dtype=np.float32)
    return float(np.nanmean(vafs)), vafs

def train_torch_model(model, X_train, Y_train, num_epochs, lr,
                      batch_size, num_workers=None, use_amp=True):
    if num_workers is None:
        num_workers = auto_num_workers()

    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset  = TensorDataset(x_cpu, y_cpu)

    loader = DataLoader(
        dset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        generator=None if PERF_MODE else _dl_generator(),
        worker_init_fn=None if PERF_MODE else _seed_worker
    )

    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    model.train()
    for ep in range(1, num_epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = crit(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep}/{num_epochs}  loss={total/len(loader):.4f}")
    return model

# ============================ GRID (Wiener + Kalman) ============================
GRID: Dict[str, Dict[str, List[Any]]] = {
    "wiener": dict(  # FIR + quad (hidden_dim not used)
        n_pca=[16,32,64,96],
        k_lag=[15,20,25,30,40],   # à 50 Hz: 25 ~ 500 ms
        hidden_dim=[1],           # placeholder non utilisé
        num_epochs=[100,200],
        lr=[1e-3,3e-3],
    ),
    "kalman": dict(  # pas d'entraînement torch; n_pca impacte seulement H
        n_pca=[16,32,64,96],
        k_lag=[1],               # pas de lags côté obs pour KF
        hidden_dim=[1],
        num_epochs=[1],          # ignoré
        lr=[1e-3],               # ignoré
    ),
}

DECODER_DISPLAY = {"wiener":"WienerCascade", "kalman":"Kalman"}

def get_model(decoder: str, n_pca: int, k_lag: int, hidden_dim: int, n_out: int):
    if decoder == "wiener":
        return WienerCascadeEthier(k_lag * n_pca, n_out).to(DEVICE)
    else:
        return None  # Kalman géré à part

# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--decoders", nargs="+", default=["wiener", "kalman"])
    ap.add_argument("--seeds",    type=int, default=1, help="number of seeds (0..seeds-1)")
    ap.add_argument("--folds",    type=int, default=5)
    ap.add_argument("--progress", type=int, default=20, help="print heartbeat every N runs")
    ap.add_argument("--combined_pickle", type=str, default="combined.pkl")
    # cluster partitioning
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--num_per_job", type=int, default=999999)
    # perf
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 => auto from $SLURM_CPUS_PER_TASK")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--perf_mode", action="store_true")
    args = ap.parse_args()

    global PERF_MODE
    if args.perf_mode:
        PERF_MODE = True

    # ---------- Load data ----------
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")
    unique_days = sorted(combined_df["date"].dropna().unique())
    if not unique_days:
        raise RuntimeError("No days in combined_df")
    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    ALL_UNITS = get_all_unit_names(combined_df)

    # detect EMG channels
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]; break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]; break
    if n_emg_channels == 0:
        raise RuntimeError("Could not detect EMG channels.")

    EMG_LABELS = get_emg_labels_from_df(combined_df, n_emg_channels)

    # raw day0 arrays (no smoothing yet)
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0:
        raise RuntimeError("Empty day0 after downsampling.")

    # precompute contiguous CV splits over time
    splits = time_kfold_splits(X0_raw.shape[0], args.folds)

    # perf knobs
    set_seed(SEED)
    BATCH = args.batch_size
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # build all combos
    combos = []
    for dec in args.decoders:
        for cfg in itertools.product(*GRID[dec].values()):
            cfg_dict = dict(zip(GRID[dec].keys(), cfg))
            for seed in range(args.seeds):
                combos.append((dec, cfg_dict, seed))

    start = args.start_idx
    end   = min(len(combos), start + args.num_per_job)
    my_combos = combos[start:end]
    print(f"This job handles combos [{start}..{end-1}] of {len(combos)} total.")

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    out_summary = Path(f"gridsearch_results_{job_id}.pkl")
    out_rows    = Path(f"gridsearch_rows_{job_id}.pkl")

    # load previous
    results: List[Dict[str, Any]] = pickle.load(open(out_summary, "rb")) if out_summary.exists() else []
    long_rows: List[Dict[str, Any]] = pickle.load(open(out_rows, "rb")) if out_rows.exists() else []

    done_keys = {(r["decoder"], r["n_pca"], r["k_lag"],
                  r.get("hidden_dim", 0), r["num_epochs"], r["lr"], r["seed"])
                 for r in results}

    total = 0
    for decoder, cfg, seed in my_combos:
        key = (decoder, int(cfg["n_pca"]), int(cfg["k_lag"]),
               int(cfg["hidden_dim"]), int(cfg["num_epochs"]), float(cfg["lr"]), int(seed))
        if key in done_keys:
            continue
        total += 1
        if total % args.progress == 0:
            print(f"  …{total} runs done in this job")

        try:
            set_seed(seed)
            n_pca = int(cfg["n_pca"])
            k_lag = int(cfg["k_lag"])
            hidden_dim = int(cfg["hidden_dim"])
            num_epochs = int(cfg["num_epochs"])
            lr = float(cfg["lr"])
            is_linear_like = (decoder == "wiener")
            stride = 1
            EMB = embargo_bins(k_lag, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE) if is_linear_like else embargo_bins(1, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE)

            vafs_fold = []
            fold_times = []
            param_count = None

            for i_fold, (val_start, val_end) in enumerate(splits):
                # raw segments
                X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
                X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
                X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

                # preprocess segments independently (no leakage)
                # LEFT (trim right by EMB)
                Xl, Yl = (preprocess_segment(X_left_raw, Y_left_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                          if len(X_left_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xl) > EMB:
                    Xl = Xl[:len(Xl)-EMB]; Yl = Yl[:len(Yl)-EMB]
                    cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=0, trim_right=EMB, seg_len=len(X_left_raw))
                else:
                    Xl = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yl = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_left = []

                # VAL (trim both sides by EMB)
                Xv, Yv = (preprocess_segment(X_val_raw, Y_val_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                          if len(X_val_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xv) > 2*EMB:
                    Xv = Xv[EMB:len(Xv)-EMB]; Yv = Yv[EMB:len(Yv)-EMB]
                    cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=EMB, trim_right=EMB, seg_len=len(X_val_raw))
                else:
                    Xv = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yv = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_val = []

                # RIGHT (trim left by EMB)
                Xr, Yr = (preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                          if len(X_right_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xr) > EMB:
                    Xr = Xr[EMB:]; Yr = Yr[EMB:]
                    cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=EMB, trim_right=0, seg_len=len(X_right_raw))
                else:
                    Xr = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yr = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_right = []

                # concatenate train
                if Xl.size and Xr.size:
                    X_train_time = np.vstack([Xl, Xr])
                    Y_train_time = np.vstack([Yl, Yr])
                    cuts_train = cuts_left + [len(Xl)] + [c + len(Xl) for c in cuts_right]
                elif Xl.size:
                    X_train_time, Y_train_time, cuts_train = Xl, Yl, cuts_left
                else:
                    X_train_time, Y_train_time, cuts_train = Xr, Yr, cuts_right

                # PCA (train only)
                if X_train_time.shape[0] == 0 or Xv.shape[0] == 0:
                    continue
                pca_model = fit_pca(X_train_time, n_components=n_pca, seed=seed + i_fold)
                Z_tr_full = pca_transform(pca_model, X_train_time)[:, :n_pca]
                Z_va_full = pca_transform(pca_model, Xv)[:, :n_pca]

                if decoder == "wiener":
                    # --- sequences laggées / respect cuts ---
                    if X_train_time.shape[0] <= k_lag or Xv.shape[0] <= k_lag:
                        continue
                    X_tr, Y_tr = build_seq_with_cuts(Z_tr_full, Y_train_time, k_lag, cuts_train, stride, True)
                    X_va, Y_va = build_seq_with_cuts(Z_va_full, Yv,            k_lag, cuts_val,   stride, True)
                    if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                        continue

                    model = get_model(decoder, n_pca, k_lag, hidden_dim, n_emg_channels)
                    if param_count is None:
                        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    t0 = time.perf_counter()
                    model = train_torch_model(model, X_tr, Y_tr, num_epochs=num_epochs, lr=lr,
                                              batch_size=BATCH, num_workers=WORKERS, use_amp=USE_AMP)
                    fold_times.append(time.perf_counter() - t0)

                    mean_vaf, vaf_ch = eval_vaf_full(model, X_va, Y_va, batch_size=BATCH, use_amp=USE_AMP)

                    # long rows
                    for ch_idx, v in enumerate(vaf_ch):
                        long_rows.append(dict(
                            decoder=DECODER_DISPLAY.get(decoder, decoder),
                            dim_red="PCA",
                            align="crossval",
                            day_int=0,
                            fold=i_fold,
                            emg_channel=int(ch_idx),
                            emg_label=str(EMG_LABELS[ch_idx]) if ch_idx < len(EMG_LABELS) else f"ch{ch_idx}",
                            vaf=float(v),
                            seed=int(seed),
                            n_pca=int(n_pca),
                            k_lag=int(k_lag),
                            hidden_dim=int(hidden_dim),
                            num_epochs=int(num_epochs),
                            lr=float(lr)
                        ))

                    del model; torch.cuda.empty_cache(); gc.collect()

                elif decoder == "kalman":
                    # --- KF : pas de lags côté obs -> utiliser Z_tr_full/Z_va_full directement ---
                    # Fit sur train en respectant les cuts (pour A/Q). H/R via régression complète
                    kf = KalmanDecoder()
                    t0 = time.perf_counter()
                    kf.fit(Z_tr_full, Y_train_time, cuts_train)
                    # Filtrage sur val par blocs contigus si cuts_val
                    Y_hat = kf.filter(Z_va_full, cuts_val)
                    fold_times.append(time.perf_counter() - t0)

                    # VAF
                    vafs = []
                    for ch in range(Yv.shape[1]):
                        yt = Yv[:, ch]; yp = Y_hat[:, ch]
                        vt = np.var(yt)
                        vafs.append(np.nan if vt < 1e-12 else 1.0 - np.var(yt - yp)/vt)
                    vafs = np.asarray(vafs, dtype=np.float32)
                    mean_vaf = float(np.nanmean(vafs))
                    vaf_ch = vafs

                    # "param_count" approx (A, H, Q upper, R upper, x0, P0 diag) — indicatif
                    if param_count is None:
                        n_out = Y_train_time.shape[1]; n_feat = Z_tr_full.shape[1]
                        param_count = n_out*n_out + n_feat*n_out + n_out*(n_out+1)//2 + n_feat*(n_feat+1)//2 + n_out + n_out

                    for ch_idx, v in enumerate(vaf_ch):
                        long_rows.append(dict(
                            decoder=DECODER_DISPLAY.get(decoder, decoder),
                            dim_red="PCA",
                            align="crossval",
                            day_int=0,
                            fold=i_fold,
                            emg_channel=int(ch_idx),
                            emg_label=str(EMG_LABELS[ch_idx]) if ch_idx < len(EMG_LABELS) else f"ch{ch_idx}",
                            vaf=float(v),
                            seed=int(seed),
                            n_pca=int(n_pca),
                            k_lag=int(k_lag),
                            hidden_dim=int(hidden_dim),
                            num_epochs=int(num_epochs),
                            lr=float(lr)
                        ))

                else:
                    raise ValueError(f"Unknown decoder: {decoder}")

                vafs_fold.append(float(mean_vaf if not np.isnan(mean_vaf) else -1.0))

            if not vafs_fold:
                raise RuntimeError("No valid folds (after embargo/windowing) for this config.")

            # append summary
            results.append(dict(
                decoder=decoder,
                seed=int(seed),
                num_params=int(param_count) if param_count is not None else None,
                mean_vaf=float(np.mean(vafs_fold)),
                fold_vafs=[float(x) for x in vafs_fold],
                fold_times=[float(s) for s in fold_times],
                mean_time=float(np.mean(fold_times)) if fold_times else np.nan,
                n_pca=int(n_pca),
                k_lag=int(k_lag),
                hidden_dim=int(hidden_dim),
                num_epochs=int(num_epochs),
                lr=float(lr)
            ))
            done_keys.add(key)

            # checkpoint both files frequently
            pickle.dump(results, open(out_summary, "wb"))
            pickle.dump(long_rows, open(out_rows, "wb"))

        except RuntimeError as e:
            warnings.warn(f"{key} failed: {e}")
            continue

    print(f"\nJob done. Saved:")
    print("  ", out_summary.resolve())
    print("  ", out_rows.resolve())

if __name__ == "__main__":
    main()
