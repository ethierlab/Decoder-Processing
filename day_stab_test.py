#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from sklearn.decomposition import PCA

# Optional UMAP
try:
    import umap
except Exception:
    umap = None

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

# ---------------- Runtime / perf ----------------
SEED = 42
BIN_FACTOR = 20
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
SAMPLING_RATE = 1000
GAUSS_TRUNCATE = 4.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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
        if n > 0: return max(2, n-1)
    except Exception:
        pass
    return default

# ---------------- Data helpers ----------------
def get_all_unit_names(df):
    s = set()
    for _, r in df.iterrows():
        sc = r.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame): s.update(sc.columns)
    return sorted(s)

def butter_lowpass(data, fs, order=4, cutoff_hz=5.0):
    nyq = 0.5*fs
    b,a = butter(order, cutoff_hz/nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def downsample_spike_and_emg(spike_df, emg_data, bin_factor=10):
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg_data
    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor
    spk = spike_df.values[:T_new*bin_factor, :].reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(spk, columns=spike_df.columns)
    if isinstance(emg_data, pd.DataFrame):
        e = emg_data.values; cols = emg_data.columns
    else:
        e = np.asarray(emg_data); cols = None
    if e.shape[0] < bin_factor: return ds_spike_df, emg_data
    e = e[:T_new*bin_factor, ...]
    if e.ndim == 2:
        e = e.reshape(T_new, bin_factor, e.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e, columns=cols) if cols is not None else e
    else:
        ds_emg = emg_data
    return ds_spike_df, ds_emg

def smooth_spike_data(x_2d, bin_size=0.001, smoothing_length=0.05):
    sigma = (smoothing_length / bin_size) / 2.0
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    spikes_all, emg_all, lengths = [], [], []
    for _, row in df.iterrows():
        spike_df = row["spike_counts"]; emg_val = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty: continue
        if emg_val is None: continue
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)
        ds_spk, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spk.shape[0] == 0: continue
        Xr = ds_spk.values.astype(np.float32)
        Yr = ds_emg.values.astype(np.float32) if isinstance(ds_emg, pd.DataFrame) else np.asarray(ds_emg, dtype=np.float32)
        spikes_all.append(Xr); emg_all.append(Yr); lengths.append(len(Xr))
    if not spikes_all:
        return np.empty((0,), np.float32), np.empty((0,), np.float32), []
    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(spikes_all,0), np.concatenate(emg_all,0), cuts

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_fs = SAMPLING_RATE // bin_factor
    Xs = smooth_spike_data(Xseg, bin_size*bin_factor, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def preprocess_within_cuts(X_raw, Y_raw, cuts, bin_factor):
    if not cuts: return preprocess_segment(X_raw, Y_raw, bin_factor)
    Xs_list, Ys_list = [], []
    start = 0
    for c in cuts + [len(X_raw)]:
        Xs, Ys = preprocess_segment(X_raw[start:c], Y_raw[start:c], bin_factor)
        Xs_list.append(Xs); Ys_list.append(Ys); start = c
    return np.concatenate(Xs_list,0), np.concatenate(Ys_list,0)

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    return (smoothing_length / (bin_size*bin_factor)) / 2.0

def embargo_bins(K_LAG, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(K_LAG, emb)

def time_kfold_splits(n_time, n_splits):
    block = n_time // n_splits
    out = []
    for k in range(n_splits):
        v0 = k*block
        v1 = (k+1)*block if k < n_splits-1 else n_time
        out.append((v0, v1))
    return out

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None: seg_len = end - start
    new_start = trim_left; new_end = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    end = n_time if end is None else end
    idx = []
    for t in range(start + k, end, stride):
        if any(t-k < c < t for c in cuts): continue
        idx.append(t)
    return idx

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    if not idx:
        if is_linear:
            return np.empty((0, K_LAG*Z.shape[1]), np.float32), np.empty((0, Y.shape[1]), np.float32)
        else:
            return np.empty((0, K_LAG, Z.shape[1]), np.float32), np.empty((0, Y.shape[1]), np.float32)
    if is_linear:
        X = np.stack([Z[t-K_LAG:t,:].reshape(-1) for t in idx],0).astype(np.float32)
    else:
        X = np.stack([Z[t-K_LAG:t,:]          for t in idx],0).astype(np.float32)
    Yb = np.stack([Y[t,:] for t in idx],0).astype(np.float32)
    return X, Yb

# ---------------- Models ----------------
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out,_ = self.gru(x); out = out[:,-1,:]; return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out,_ = self.lstm(x); out = out[:,-1,:]; return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        x = self.act(self.lin1(x)); return self.lin2(x)

class LiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, x, h):
        z = torch.sigmoid(self.x2z(x) + self.h2z(h))
        h_tilde = torch.relu(self.x2h(x) + self.h2h(h))
        return (1 - z)*h + z*h_tilde

class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        B,T,_ = x.size()
        h = torch.zeros(B, self.hidden, device=x.device)
        for t in range(T):
            h = self.cell(x[:,t,:], h)
        return self.fc(h)

# ---------------- Dimred & Align ----------------
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        m = PCA(n_components=n_components, random_state=seed); m.fit(data); return m
    elif method.upper() == "UMAP":
        if umap is None: raise RuntimeError("umap-learn not installed (pip install umap-learn)")
        m = umap.UMAP(n_components=n_components, random_state=seed); m.fit(data); return m
    else:
        raise ValueError("Unknown dimred")

def transform_dimred(model, data, method):
    return model.transform(data)

def align_linear_pinv(Zx, Z0, lam=1e-6):
    if Zx.shape[1] != Z0.shape[1]:
        d = min(Zx.shape[1], Z0.shape[1])
        Zx = Zx[:,:d]; Z0 = Z0[:,:d]
    X = Zx - Zx.mean(0, keepdims=True)
    Y = Z0 - Z0.mean(0, keepdims=True)
    d = X.shape[1]
    A = np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ Y)
    return (Zx - Zx.mean(0, keepdims=True)) @ A + Z0.mean(0, keepdims=True)

def pca_basis_change(zx_dayX, pca_x, pca_0, n_comp):
    Vx = pca_x.components_[:n_comp, :].T
    V0 = pca_0.components_[:n_comp, :].T
    R  = pinv(Vx) @ V0
    return zx_dayX @ R

# ---------------- Metrics / Train/Eval ----------------
def compute_vaf_1d(y, yhat):
    vt = np.var(y)
    if vt < 1e-12: return np.nan
    return 1.0 - np.var(y - yhat)/vt

def compute_multichannel_vaf(Y, Yhat):
    if Y.shape[0] == 0: return np.array([])
    return np.array([compute_vaf_1d(Y[:,i], Yhat[:,i]) for i in range(Y.shape[1])])

def train_model(model, X_train, Y_train, num_epochs=200, lr=0.003, batch_size=256, num_workers=None, use_amp=True):
    if num_workers is None: num_workers = auto_num_workers()
    dset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                         torch.as_tensor(Y_train, dtype=torch.float32))
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers>0), prefetch_factor=4)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    model.train()
    for ep in range(1, num_epochs+1):
        tot = 0.0
        for xb,yb in loader:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb); loss = crit(pred, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{num_epochs} - loss={tot/len(loader):.4f}")
    return model

def evaluate_model(model, X, Y, batch_size=256, use_amp=True):
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
                out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        P = np.concatenate(preds,0); vafs = compute_multichannel_vaf(Y, P)
        return float(np.nanmean(vafs)), vafs
    return np.nan, np.full((Y.shape[1],), np.nan)

# ---------------- Hyperparams ----------------
ARCH = {
    "GRU":    dict(N=32, K=25, H=96,  E=60,  LR=0.003),
    "LSTM":   dict(N=24, K=25, H=128, E=80,  LR=0.003),
    "Linear": dict(N=32, K=16, H=64,  E=60,  LR=0.003),
    "LiGRU":  dict(N=32, K=16, H=64,  E=60,  LR=0.001),
}

# ---------------- MAIN ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decoder', type=str, required=True, choices=list(ARCH.keys()))
    p.add_argument('--dimred',  type=str, default="PCA", choices=["PCA", "UMAP"])
    p.add_argument('--align_mode', type=str, default="none", choices=["none","basis","latent"])
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', type=str, default="results_emg_cross_day")
    p.add_argument('--combined_pickle', type=str, default="combined.pkl")

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=-1)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--perf_mode', action='store_true')
    p.add_argument('--stride_mul', type=float, default=1.0)

    args = p.parse_args()

    global PERF_MODE
    PERF_MODE = bool(args.perf_mode)
    set_seed(args.seed)

    df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    ALL_UNITS = get_all_unit_names(df)
    unique_days = sorted(pd.to_datetime(df["date"]).unique())
    if len(unique_days) == 0: raise RuntimeError("No days found in combined_df")

    # EMG dims
    n_emg = 0
    for _, row in df.iterrows():
        E = row.get("EMG", None)
        if isinstance(E, pd.DataFrame) and not E.empty: n_emg = E.shape[1]; break
        if isinstance(E, np.ndarray) and E.size>0:       n_emg = E.shape[1]; break
    if n_emg == 0: raise RuntimeError("Could not detect EMG channels")

    # Train on day0 only
    day0 = unique_days[0]
    train_df = df[df["date"] == day0].reset_index(drop=True)

    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0: raise RuntimeError("empty day0")

    splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)

    hp = ARCH[args.decoder]
    N, K, H, EPOCHS, LR = hp["N"], hp["K"], hp["H"], hp["E"], hp["LR"]
    STRIDE = max(1, int(args.stride_mul * K))
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    print(f"[INFO] train day={pd.to_datetime(day0).date()} decoder={args.decoder} dimred={args.dimred} align={args.align_mode}")

    results = []

    for fold_idx, (v0, v1) in enumerate(splits):
        # carve segments
        Xl_raw, Yl_raw = X0_raw[:v0], Y0_raw[:v0]
        Xv_raw, Yv_raw = X0_raw[v0:v1], Y0_raw[v0:v1]
        Xr_raw, Yr_raw = X0_raw[v1:],  Y0_raw[v1:]

        EMB = embargo_bins(K, BIN_FACTOR)

        # preprocess + embargo trims
        # LEFT
        Xl, Yl = preprocess_segment(Xl_raw, Yl_raw, BIN_FACTOR) if len(Xl_raw) else (np.empty((0,)), np.empty((0,)))
        if len(Xl) > EMB: Xl, Yl = Xl[:-EMB], Yl[:-EMB]
        else: Xl, Yl = np.empty((0, X0_raw.shape[1])), np.empty((0, Y0_raw.shape[1]))
        cuts_left = adjust_cuts_for_segment(0, len(Xl_raw), cuts0, trim_left=0, trim_right=EMB, seg_len=len(Xl_raw))

        # VAL (ALWAYS none)
        Xv, Yv = preprocess_segment(Xv_raw, Yv_raw, BIN_FACTOR) if len(Xv_raw) else (np.empty((0,)), np.empty((0,)))
        if len(Xv) > 2*EMB: Xv, Yv = Xv[EMB:-EMB], Yv[EMB:-EMB]
        else: Xv, Yv = np.empty((0, X0_raw.shape[1])), np.empty((0, Y0_raw.shape[1]))
        cuts_val = adjust_cuts_for_segment(v0, v1, cuts0, trim_left=EMB, trim_right=EMB, seg_len=len(Xv_raw))

        # RIGHT
        Xr, Yr = preprocess_segment(Xr_raw, Yr_raw, BIN_FACTOR) if len(Xr_raw) else (np.empty((0,)), np.empty((0,)))
        if len(Xr) > EMB: Xr, Yr = Xr[EMB:], Yr[EMB:]
        else: Xr, Yr = np.empty((0, X0_raw.shape[1])), np.empty((0, Y0_raw.shape[1]))
        cuts_right = adjust_cuts_for_segment(v1, len(X0_raw), cuts0, trim_left=EMB, trim_right=0, seg_len=len(Xr_raw))

        # concat train
        if Xl.size and Xr.size:
            Xtr = np.vstack([Xl,Xr]); Ytr = np.vstack([Yl,Yr])
            cuts_tr = cuts_left + [c + len(Xl) for c in cuts_right] + [len(Xl)]
        elif Xl.size:
            Xtr, Ytr, cuts_tr = Xl, Yl, cuts_left
        else:
            Xtr, Ytr, cuts_tr = Xr, Yr, cuts_right

        if Xtr.shape[0] <= K or Xv.shape[0] <= K:
            print(f"[WARN] fold {fold_idx}: not enough after embargo; skip")
            continue

        # Train manifold on TRAIN only
        dimred_train = get_dimred_model(Xtr, args.dimred, max(N,2), args.seed + fold_idx)
        Z_tr = transform_dimred(dimred_train, Xtr, args.dimred)[:,:N]
        Z_va = transform_dimred(dimred_train, Xv,  args.dimred)[:,:N]

        # Windowing
        is_linear = (args.decoder == "Linear")
        X_tr_seq, Y_tr_seq = build_seq_with_cuts(Z_tr, Ytr, K, cuts_tr, STRIDE, is_linear)
        X_va_seq, Y_va_seq = build_seq_with_cuts(Z_va, Yv,  K, cuts_val, STRIDE, is_linear)
        if X_tr_seq.shape[0]==0 or X_va_seq.shape[0]==0:
            print(f"[WARN] fold {fold_idx}: empty after windowing; skip")
            continue

        # Model
        if args.decoder == "GRU":
            model = GRUDecoder(N, ARCH["GRU"]["H"], n_emg).to(DEVICE)
            epochs, lr = ARCH["GRU"]["E"], ARCH["GRU"]["LR"]
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N, ARCH["LSTM"]["H"], n_emg).to(DEVICE)
            epochs, lr = ARCH["LSTM"]["E"], ARCH["LSTM"]["LR"]
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K*N, ARCH["Linear"]["H"], n_emg).to(DEVICE)
            epochs, lr = ARCH["Linear"]["E"], ARCH["Linear"]["LR"]
        else:
            model = LiGRUDecoder(N, ARCH["LiGRU"]["H"], n_emg).to(DEVICE)
            epochs, lr = ARCH["LiGRU"]["E"], ARCH["LiGRU"]["LR"]

        print(f"[INFO] fold {fold_idx+1}/{args.n_folds} - training {args.decoder} ...")
        model = train_model(model, X_tr_seq, Y_tr_seq, num_epochs=epochs, lr=lr,
                            batch_size=args.batch_size, num_workers=WORKERS, use_amp=USE_AMP)
        vaf_val, _ = evaluate_model(model, X_va_seq, Y_va_seq, batch_size=args.batch_size, use_amp=USE_AMP)
        print(f"[RESULT] internal CV fold={fold_idx} | VAF={vaf_val:.4f}")

        # -------- Cross-days with selected align_mode --------
        for d_val in unique_days:
            if pd.to_datetime(d_val) == pd.to_datetime(day0): 
                continue
            day_df = df[df["date"] == d_val].reset_index(drop=True)
            X_raw, Y_raw, cuts_test = build_continuous_dataset_raw(day_df, BIN_FACTOR, all_units=ALL_UNITS)
            if X_raw.shape[0] == 0: continue
            Xp, Yp = preprocess_within_cuts(X_raw, Y_raw, cuts_test, BIN_FACTOR)

            # direct projection with train manifold
            Z_direct = transform_dimred(dimred_train, Xp, args.dimred)[:, :N]

            # if align_mode != none → fit manifold on test day and align
            if args.align_mode == "none":
                Z_test = Z_direct
                mode_used = "none"
            else:
                dimred_tgt = get_dimred_model(Xp, args.dimred, N, args.seed + fold_idx + 777)
                Z_tgt_space = transform_dimred(dimred_tgt, Xp, args.dimred)[:, :N]
                if args.align_mode == "basis":
                    if args.dimred.upper() != "PCA":
                        Z_test = align_linear_pinv(Z_tgt_space, Z_direct, lam=1e-6)
                        mode_used = "latent"
                    else:
                        Z_test = pca_basis_change(Z_tgt_space, dimred_tgt, dimred_train, N)
                        mode_used = "basis"
                else:  # latent
                    Z_test = align_linear_pinv(Z_tgt_space, Z_direct, lam=1e-6)
                    mode_used = "latent"

            X_seq, Y_seq = build_seq_with_cuts(Z_test, Yp, K, cuts_test, STRIDE, is_linear)
            if X_seq.shape[0] == 0: continue
            vaf_cd, _ = evaluate_model(model, X_seq, Y_seq, batch_size=args.batch_size, use_amp=USE_AMP)
            print(f"[RESULT] cross-day {pd.to_datetime(d_val).date()} | {args.dimred}+{mode_used} | fold={fold_idx} | VAF={vaf_cd:.4f}")

            results.append({
                "train_day": pd.to_datetime(day0),
                "test_day": pd.to_datetime(d_val),
                "fold": fold_idx,
                "decoder": args.decoder,
                "dimred": args.dimred,
                "align_mode": mode_used,
                "vaf": vaf_cd
            })

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"crossday_results_{args.decoder}_{args.dimred}_{args.align_mode}_kf{args.n_folds}_seed{args.seed}.pkl")
    pd.to_pickle(pd.DataFrame(results), out)
    print(f"[INFO] Saved → {out}")

if __name__ == "__main__":
    main()
