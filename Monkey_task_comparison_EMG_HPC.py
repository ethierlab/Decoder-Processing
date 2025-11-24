#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-task EMG decoder validation with STRICT no-leakage CV, PCA/UMAP + alignment,
independent decoders, and perf tweaks.

What you get:
  • Scenario engine (within-monkey cross-task, same-day by default).
  • Seeds × K-fold blocked CV on the TRAIN day/time series.
  • Strict anti-leakage:
      - build raw with trial `cuts`
      - per-segment preprocessing (left / val / right)
      - embargo trims based on smoothing tail & K_LAG
      - windowing that never crosses trial cuts
      - dimred fit on TRAIN-only per fold (VAL never touches fit)
  • Dimensionality reduction: PCA or UMAP.
  • Alignments:
      - none: direct projection via train manifold
      - latent: Moore–Penrose in latent space (works for PCA/UMAP)
      - pca_basis: pinv on PCA bases (PCA only; true change of basis)
  • AMP/TF32, DataLoader workers/pinning, stride windowing.

CLI example:
  python Monkey_task_comparison_EMG_HPC.py \
    --input output.pkl --scenario Jaco_mgpt --train_day_idx 0 \
    --decoder GRU --dimred pca --align pca_basis \
    --n_folds 5 --seeds 42,43,44 --perf_mode

"""

import os, sys, argparse, random, warnings, datetime, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────── Runtime / Seeds ─────────────────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True   # True => speed (TF32, non-deterministic cudnn), False => stricter repro

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

# ─────────────────────── Hyperparams per decoder ───────────────────────
DECODER_CONFIG = {
    "GRU":    {"N_PCA": 32, "HIDDEN": 96,  "K_LAG": 25, "LR": 0.003, "EPOCHS": 200},
    "LSTM":   {"N_PCA": 24, "HIDDEN": 128, "K_LAG": 25, "LR": 0.003, "EPOCHS": 300},
    "Linear": {"N_PCA": 32, "HIDDEN": 64,  "K_LAG": 16, "LR": 0.003, "EPOCHS": 100},
    "LiGRU":  {"N_PCA": 32, "HIDDEN": 5,   "K_LAG": 16, "LR": 0.001, "EPOCHS": 200},
}
BATCH_SIZE_DEFAULT = 256
BIN_SIZE = 0.02   # 20 ms -> fs ~ 50Hz
SMOOTHING = 0.05  # 50 ms smoothing on spikes
GAUSS_TRUNCATE = 4.0  # embargo tail ~ truncate*sigma

# ─────────────────────── EMG mapping ───────────────────────
TARGET = {"FCR","FDS","FDP","FCU","ECR","EDC","ECU"}
GLOBAL_MUSCLE_MAP = {
    'ECR_1':'ECR','ECR_2':'ECR','EDC_1':'EDC','EDC_2':'EDC',
    'FCR_1':'FCR','FCU_1':'FCU','FDS_1':'FDS','FDS_2':'FDS',
    'FDP_1':'FDP','FDP_2':'FDP','ECU_1':'ECU'
}

def map_emg_labels(emg_df: pd.DataFrame) -> pd.DataFrame:
    out, cnt = {}, defaultdict(int)
    for col in emg_df.columns:
        raw = col.strip().upper(); tmp = raw
        while tmp and tmp not in GLOBAL_MUSCLE_MAP: tmp = tmp[:-1]
        base = GLOBAL_MUSCLE_MAP.get(tmp, None)
        if base and base in TARGET:
            cnt[base]+=1; out[f"{base}_{cnt[base]}"] = emg_df[col]
    return pd.DataFrame(out)

def filter_and_map_emg(df: pd.DataFrame):
    rows, cols = [], set()
    for _,r in df.iterrows():
        emg=r.get("EMG")
        if isinstance(emg, pd.DataFrame) and not emg.empty:
            m=map_emg_labels(emg); r=r.copy(); r["EMG"]=m; cols.update(m.columns)
        rows.append(r)
    df2=pd.DataFrame(rows); cols=sorted(cols)
    for i,r in df2.iterrows():
        emg=r.get("EMG")
        if isinstance(emg, pd.DataFrame):
            r["EMG"]=emg.reindex(cols, fill_value=0)
    return df2, cols

# ───────────────────────── Helpers for units ─────────────────────────
def get_all_unit_names(df):
    unit_set = set()
    for _, row in df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

# ───────────────────────── Preprocessing (strict) ─────────────────────────
def smooth_spike_data(X, bin_size=BIN_SIZE, smoothing=SMOOTHING):
    sigma=(smoothing/bin_size)/2.0
    return gaussian_filter1d(X.astype(np.float32), sigma=sigma, axis=0)

def butter_lowpass_abs(Y, fs_hz, cutoff_hz=5.0, order=4):
    rect = np.abs(Y)
    nyq = 0.5*fs_hz
    b, a = butter(order, cutoff_hz/nyq, btype='low', analog=False)
    return filtfilt(b, a, rect, axis=0)

def downsample_pair(spike_df, emg, bin_factor=1):
    # here BIN_FACTOR defaults to 1 (already binned). If you want >1, this supports it.
    if bin_factor == 1:
        return spike_df, emg
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg
    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor
    sp = spike_df.values[:T_new*bin_factor, :]
    sp = sp.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(sp, columns=spike_df.columns)

    if isinstance(emg, pd.DataFrame):
        e = emg.values
        cols = emg.columns
    else:
        e = np.asarray(emg)
        cols = None
    if e.shape[0] < bin_factor:
        return ds_spike_df, emg
    e = e[:T_new*bin_factor, ...]
    if e.ndim == 2:
        e = e.reshape(T_new, bin_factor, e.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e, columns=cols) if cols is not None else e
    else:
        ds_emg = emg
    return ds_spike_df, ds_emg

def build_continuous_dataset_raw(df, bin_factor=1, all_units=None, ref_emg_cols=None):
    """Concat trials, keep trial boundaries as 'cuts'. Return X_raw, Y_raw, cuts."""
    Xs, Ys, lengths = [], [], []
    expected = all_units if all_units is not None else [f"neuron{i}" for i in range(1,97)]
    for _, r in df.iterrows():
        sp = r.get("spike_counts"); emg = r.get("EMG")
        if not isinstance(sp, pd.DataFrame) or sp.empty or emg is None: continue
        sp = sp.reindex(columns=expected, fill_value=0)
        sp, emg = downsample_pair(sp, emg, bin_factor=bin_factor)
        if isinstance(emg, pd.DataFrame):
            if ref_emg_cols is not None:
                emg = emg.reindex(ref_emg_cols, axis=1, fill_value=0)
            e_val = emg.values.astype(np.float32)
        else:
            e_val = np.asarray(emg, dtype=np.float32)
        Xs.append(sp.values.astype(np.float32))
        Ys.append(e_val)
        lengths.append(len(sp))
    if not Xs:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0), cuts

def preprocess_segment(Xseg, Yseg, bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING, emg_cutoff=5.0):
    fs_eff = int(1.0 / (bin_size * bin_factor))
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing)
    Ys = butter_lowpass_abs(Yseg, fs_eff, cutoff_hz=emg_cutoff)
    return Xs, Ys

def preprocess_within_cuts(X_raw, Y_raw, cuts, bin_factor=1):
    """Apply preprocessing independently within each trial block => no bleed across trials."""
    if not cuts:
        return preprocess_segment(X_raw, Y_raw, bin_factor)
    Xp, Yp = [], []
    start = 0
    for c in cuts + [len(X_raw)]:
        Xs, Ys = preprocess_segment(X_raw[start:c], Y_raw[start:c], bin_factor)
        Xp.append(Xs); Yp.append(Ys)
        start = c
    return np.concatenate(Xp, 0), np.concatenate(Yp, 0)

def sigma_bins(bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING):
    return (smoothing / (bin_size * bin_factor)) / 2.0

def embargo_bins(K_LAG, bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing)))
    return max(K_LAG, emb)

def time_kfold_splits(n_time, n_splits):
    """Contiguous time folds (no shuffle)."""
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    """Shift global cuts into [start,end), then into local indices after trimming."""
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None:
        seg_len = end - start
    new_start = trim_left
    new_end = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    """Yield end indices t of windows [t-k, t) that DO NOT cross any cut."""
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

# ───────────────────── Decoders ─────────────────────
class GRUDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.r=nn.GRU(i,h,batch_first=True); self.f=nn.Linear(h,o)
    def forward(self,x): o,_=self.r(x); return self.f(o[:,-1,:])
class LSTMDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.r=nn.LSTM(i,h,batch_first=True); self.f=nn.Linear(h,o)
    def forward(self,x): o,_=self.r(x); return self.f(o[:,-1,:])
class LinearLagDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.l1=nn.Linear(i,h); self.a=nn.ReLU(); self.l2=nn.Linear(h,o)
    def forward(self,x): return self.l2(self.a(self.l1(x)))
class LiGRUCell(nn.Module):
    def __init__(self, i, h): super().__init__(); self.x2z=nn.Linear(i,h); self.h2z=nn.Linear(h,h,bias=False); self.x2h=nn.Linear(i,h); self.h2h=nn.Linear(h,h,bias=False)
    def forward(self,x,h): z=torch.sigmoid(self.x2z(x)+self.h2z(h)); hc=torch.relu(self.x2h(x)+self.h2h(h)); return (1-z)*h+z*hc
class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        B, T, _ = x.size()
        h = x.new_zeros(B, self.hidden_size)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

def get_model(name, n_in, h, n_out):
    if name=="GRU": return GRUDecoder(n_in,h,n_out).to(DEVICE)
    if name=="LSTM": return LSTMDecoder(n_in,h,n_out).to(DEVICE)
    if name=="Linear": return LinearLagDecoder(n_in,h,n_out).to(DEVICE)
    if name=="LiGRU": return LiGRUDecoder(n_in,h,n_out).to(DEVICE)
    raise ValueError("Unknown decoder")

# ───────────────────────── Train / Eval (AMP + DL) ─────────────────────────
def train_model(model, X_train, Y_train, epochs, lr, batch_size=BATCH_SIZE_DEFAULT,
                num_workers=None, use_amp=True):
    if num_workers is None: num_workers = auto_num_workers()
    ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                       torch.as_tensor(Y_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=4)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    model.train()
    for ep in range(1, epochs+1):
        s=0.0
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb); loss = crit(pred, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            s += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} - loss={s/len(dl):.4f}")
    return model

def compute_vaf_1d(y, yp):
    vt=np.var(y); 
    return np.nan if vt<1e-12 else 1.0 - np.var(y-yp)/vt

def eval_decoder(model, X, Y, batch_size=BATCH_SIZE_DEFAULT, use_amp=True):
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
                out = model(bx)
            preds.append(out.cpu().numpy())
    P = np.concatenate(preds, axis=0) if preds else np.empty((0,))
    ch = np.array([compute_vaf_1d(Y[:,c], P[:,c]) for c in range(Y.shape[1])], dtype=np.float32)
    return float(np.nanmean(ch)), ch

# ─────────────────── DimRed: PCA / UMAP (+ align) ───────────────────
def fit_dimred(X, method='pca', n_components=10, random_state=SEED):
    if method == 'pca':
        m = PCA(n_components=n_components, random_state=random_state); m.fit(X); return m
    elif method == 'umap':
        try:
            import umap
        except Exception as e:
            raise RuntimeError("umap-learn non installé. `pip install umap-learn`") from e
        m = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=30, min_dist=0.1)
        m.fit(X); return m
    else:
        raise ValueError("dimred must be 'pca' or 'umap'")

def transform_dimred(model, X, method='pca'):
    if method in ('pca','umap'):
        return model.transform(X)
    raise ValueError("unknown dimred method")

def align_latent_pinv(Zx: np.ndarray, Z0: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Moore–Penrose in latent space (works for PCA & UMAP). Shapes must match (T x d)."""
    if Zx.shape != Z0.shape:
        raise ValueError(f"latent pinv requires same shapes, got {Zx.shape} vs {Z0.shape}")
    X = Zx - Zx.mean(axis=0, keepdims=True)
    Y = Z0 - Z0.mean(axis=0, keepdims=True)
    d = X.shape[1]
    A = np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ Y)
    return (Zx - Zx.mean(axis=0, keepdims=True)) @ A + Z0.mean(axis=0, keepdims=True)

def align_pca_basis(pca_train: PCA, pca_test: PCA, X_test: np.ndarray, n_comp: int) -> np.ndarray:
    """True change of basis for PCA: project test with its own PCA, then map into train basis."""
    V0 = pca_train.components_[:n_comp, :].T   # p x k (train)
    Vx = pca_test.components_[:n_comp, :].T    # p x k (test)
    R  = pinv(Vx) @ V0                         # k x k
    Z_local = pca_test.transform(X_test)[:, :n_comp]
    return Z_local @ R

# ───────────────────────── Scenarios (full) ─────────────────────────
def build_scenarios():
    s=[]
    # Within-monkey Jango/JacB (iso/wm/spr)
    for m in ["Jango","JacB"]:
        for tr in ["iso","wm","spr"]:
            others=[t for t in ["iso","wm","spr"] if t!=tr]
            s.append({
                "name": f"{m}_{tr}",
                "train_filter": lambda r,m=m,tr=tr: (r["monkey"]==m) and (r["task"].strip().lower()==tr),
                "tests": [{"name":o, "test_filter": lambda r,m=m,o=o: (r["monkey"]==m) and (r["task"].strip().lower()==o)} for o in others],
                "force_same_day": True
            })
    # Jaco/Theo (mgpt/ball)
    s.append({"name":"Jaco_mgpt","train_filter":lambda r:(r["monkey"]=="Jaco") and (r["task"].strip().lower() in ["mgpt","mg-pt"]),
              "tests":[{"name":"ball","test_filter":lambda r:(r["monkey"]=="Jaco") and (r["task"].strip().lower()=="ball")}], "force_same_day":True})
    s.append({"name":"Jaco_ball","train_filter":lambda r:(r["monkey"]=="Jaco") and (r["task"].strip().lower()=="ball"),
              "tests":[{"name":"mgpt","test_filter":lambda r:(r["monkey"]=="Jaco") and (r["task"].strip().lower() in ["mgpt","mg-pt"])}], "force_same_day":True})
    s.append({"name":"Theo_mgpt","train_filter":lambda r:(r["monkey"]=="Theo") and (r["task"].strip().lower() in ["mgpt","mg-pt"]),
              "tests":[{"name":"ball","test_filter":lambda r:(r["monkey"]=="Theo") and (r["task"].strip().lower()=="ball")}], "force_same_day":True})
    s.append({"name":"Theo_ball","train_filter":lambda r:(r["monkey"]=="Theo") and (r["task"].strip().lower()=="ball"),
              "tests":[{"name":"mgpt","test_filter":lambda r:(r["monkey"]=="Theo") and (r["task"].strip().lower() in ["mgpt","mg-pt"])}], "force_same_day":True})
    return s

def parse_test_task(tname):
    tasks = ['iso','wm','spr','mgpt','ball','mg-pt']
    t = tname.lower()
    for k in tasks:
        if k in t: return 'mgpt' if k=='mg-pt' else k
    return "unknown"

# ───────────────────────── Main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--scenario', required=True, choices=[s["name"] for s in build_scenarios()])
    ap.add_argument('--train_day_idx', type=int, required=True)
    ap.add_argument('--decoder', required=True, choices=list(DECODER_CONFIG.keys()))
    ap.add_argument('--dimred', default='pca', choices=['pca','umap'])
    ap.add_argument('--align', default='none', choices=['none','latent','pca_basis'])
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seeds', type=str, default='42', help='comma-separated seeds, e.g., "42,43,44"')
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--stride_mul', type=float, default=1.0, help='stride = max(1, int(stride_mul*K_LAG))')
    ap.add_argument('--num_workers', type=int, default=-1)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--perf_mode', action='store_true')
    ap.add_argument('--bin_factor', type=int, default=1)
    ap.add_argument('--output_dir', default='results_emg_cross_task_strict')
    args = ap.parse_args()

    # perf/repro
    global PERF_MODE
    if args.perf_mode: PERF_MODE = True
    os.makedirs(args.output_dir, exist_ok=True)

    # load & EMG mapping
    df = pd.read_pickle(args.input)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df, emg_cols = filter_and_map_emg(df)

    # scenario selection
    scen_map = {s["name"]:s for s in build_scenarios()}
    sc = scen_map[args.scenario]

    df_train_task = df[df.apply(sc["train_filter"], axis=1)].copy()
    days = sorted(df_train_task["date"].dropna().unique())
    if args.train_day_idx >= len(days):
        raise IndexError("train_day_idx out of range")
    day = days[args.train_day_idx]
    df_train_day = df_train_task[df_train_task["date"]==day].copy()
    if df_train_day.empty:
        print("[ERROR] no training rows for selected scenario/day"); sys.exit(1)

    # decoder config
    cfg = DECODER_CONFIG[args.decoder]
    N = cfg["N_PCA"]; K = cfg["K_LAG"]; H = cfg["HIDDEN"]
    LR = args.lr if args.lr is not None else cfg["LR"]
    E  = args.epochs if args.epochs is not None else cfg["EPOCHS"]
    STRIDE = max(1, int(args.stride_mul * K))
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # units set for consistency
    ALL_UNITS = get_all_unit_names(df_train_day)

    # Build raw (train day) with cuts
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(df_train_day, bin_factor=args.bin_factor,
                                                         all_units=ALL_UNITS, ref_emg_cols=emg_cols)
    if X0_raw.size == 0:
        print("[ERROR] empty raw on train day"); sys.exit(1)

    # Seeds × K-fold strict CV
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()!='']
    results = []
    for seed in seeds:
        set_seed(seed)
        splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)
        EMB = embargo_bins(K, bin_factor=args.bin_factor, bin_size=BIN_SIZE, smoothing=SMOOTHING, truncate=GAUSS_TRUNCATE)

        for fold_idx, (val_start, val_end) in enumerate(splits):
            # raw segments
            X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
            X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
            X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

            # per-segment preprocessing (NO bleed)
            # LEFT: keep everything except right embargo
            trimL, trimR = 0, EMB
            X_left_p, Y_left_p = preprocess_segment(X_left_raw, Y_left_raw, args.bin_factor) if len(X_left_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_left_p) > trimR:
                X_left_p, Y_left_p = X_left_p[:len(X_left_p)-trimR], Y_left_p[:len(Y_left_p)-trimR]
                cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_left_raw))
            else:
                X_left_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_left_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_left = []

            # VAL: trim both sides by EMB
            trimL, trimR = EMB, EMB
            X_val_p, Y_val_p = preprocess_segment(X_val_raw, Y_val_raw, args.bin_factor) if len(X_val_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_val_p) > (trimL + trimR):
                X_val_p, Y_val_p = X_val_p[trimL:len(X_val_p)-trimR], Y_val_p[trimL:len(Y_val_p)-trimR]
                cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_val_raw))
            else:
                X_val_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_val_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_val = []

            # RIGHT: drop left embargo
            trimL, trimR = EMB, 0
            X_right_p, Y_right_p = preprocess_segment(X_right_raw, Y_right_raw, args.bin_factor) if len(X_right_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_right_p) > trimL:
                X_right_p, Y_right_p = X_right_p[trimL:], Y_right_p[trimL:]
                cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_right_raw))
            else:
                X_right_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_right_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_right = []

            # TRAIN = LEFT ⊕ RIGHT (keep a cut at the junction)
            if X_left_p.size and X_right_p.size:
                X_tr_time = np.vstack([X_left_p, X_right_p])
                Y_tr_time = np.vstack([Y_left_p, Y_right_p])
                cuts_train = cuts_left + [len(X_left_p)] + [c + len(X_left_p) for c in cuts_right]
            elif X_left_p.size:
                X_tr_time, Y_tr_time, cuts_train = X_left_p, Y_left_p, cuts_left
            else:
                X_tr_time, Y_tr_time, cuts_train = X_right_p, Y_right_p, cuts_right

            if X_tr_time.shape[0] <= K or X_val_p.shape[0] <= K:
                print(f"[WARN] seed={seed} fold={fold_idx}: not enough samples after embargo; skipping")
                continue

            # Fit manifold on TRAIN-only
            dim_model_train = fit_dimred(X_tr_time, method=args.dimred, n_components=max(N,2), random_state=seed)
            Z_tr = transform_dimred(dim_model_train, X_tr_time, method=args.dimred)[:, :N]
            Z_va = transform_dimred(dim_model_train, X_val_p,   method=args.dimred)[:, :N]

            is_linear = (args.decoder == "Linear")
            X_tr, Y_tr = build_seq_with_cuts(Z_tr, Y_tr_time, K, cuts_train, STRIDE, is_linear)
            X_va, Y_va = build_seq_with_cuts(Z_va, Y_val_p,   K, cuts_val,   STRIDE, is_linear)
            if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                print(f"[WARN] seed={seed} fold={fold_idx}: empty after windowing; skipping")
                continue

            # Model
            n_in = (N*K) if is_linear else N
            n_out = Y_tr.shape[1]
            model = get_model(args.decoder, n_in, H, n_out)
            print(f"[INFO] seed={seed} fold={fold_idx+1}/{args.n_folds} - training {args.decoder} ...")
            t0=time.time()
            model = train_model(model, X_tr, Y_tr, epochs=E, lr=LR, batch_size=args.batch_size,
                                num_workers=WORKERS, use_amp=USE_AMP)
            print(f"[INFO] train time: {time.time()-t0:.1f}s")
            mVAF_val, vaf_ch_val = eval_decoder(model, X_va, Y_va, batch_size=args.batch_size, use_amp=USE_AMP)
            results.append({
                "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                "decoder": args.decoder, "dimred": args.dimred, "align": "internal_cv",
                "seed": seed, "fold": fold_idx, "split": [int(val_start), int(val_end)],
                "mean_VAF": float(mVAF_val), "per_channel_VAF": vaf_ch_val,
                "timestamp": datetime.datetime.now(),
                "K_LAG": K, "stride": STRIDE, "embargo_bins": EMB
            })
            print(f"[RESULT] internal CV fold={fold_idx} | VAF={mVAF_val:.4f}")

            # ── Cross-task off-diagonal on SAME day (strict preprocessing within cuts)
            df_same_day = df[df["date"]==day].copy() if sc.get("force_same_day", True) else df
            for tcfg in sc["tests"]:
                test_name = tcfg["name"]
                df_test = df_same_day[df_same_day.apply(tcfg["test_filter"], axis=1)].copy()
                if df_test.empty: 
                    print(f"[WARN] no test data for {test_name}"); 
                    continue

                # Build raw+cuts and preprocess per trial (no embargo needed for inference)
                ALL_UNITS_TEST = get_all_unit_names(df_test)
                X_raw_te, Y_raw_te, cuts_te = build_continuous_dataset_raw(df_test, bin_factor=args.bin_factor,
                                                                           all_units=ALL_UNITS_TEST, ref_emg_cols=emg_cols)
                if X_raw_te.size == 0: 
                    print(f"[WARN] empty X for {test_name}"); 
                    continue
                X_te_proc, Y_te_proc = preprocess_within_cuts(X_raw_te, Y_raw_te, cuts_te, bin_factor=args.bin_factor)

                # Direct projection via TRAIN manifold
                Zte_direct = transform_dimred(dim_model_train, X_te_proc, method=args.dimred)[:, :N]

                # Select alignment
                if args.align == 'none':
                    Zte = Zte_direct
                elif args.align == 'latent':
                    dim_model_test = fit_dimred(X_te_proc, method=args.dimred, n_components=max(N,2), random_state=seed)
                    Zte_test = transform_dimred(dim_model_test, X_te_proc, method=args.dimred)[:, :N]
                    Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
                elif args.align == 'pca_basis':
                    if args.dimred != 'pca':
                        # fall back to latent if UMAP
                        dim_model_test = fit_dimred(X_te_proc, method=args.dimred, n_components=max(N,2), random_state=seed)
                        Zte_test = transform_dimred(dim_model_test, X_te_proc, method=args.dimred)[:, :N]
                        Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
                    else:
                        pca_test = fit_dimred(X_te_proc, method='pca', n_components=max(N,2), random_state=seed)
                        Zte = align_pca_basis(dim_model_train, pca_test, X_te_proc, n_comp=N)
                else:
                    raise ValueError("unknown align option")

                # Window with cut respect
                X_te, Y_te = build_seq_with_cuts(Zte, Y_te_proc, K, cuts_te, STRIDE, is_linear)
                if X_te.shape[0] == 0:
                    print(f"[WARN] empty windows for {test_name} ({args.align})"); 
                    continue
                mVAF, vaf_ch = eval_decoder(model, X_te, Y_te, batch_size=args.batch_size, use_amp=USE_AMP)
                results.append({
                    "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                    "decoder": args.decoder, "dimred": args.dimred, "align": args.align,
                    "seed": seed, "fold": fold_idx,
                    "test_task": parse_test_task(test_name),
                    "mean_VAF": float(mVAF), "per_channel_VAF": vaf_ch,
                    "timestamp": datetime.datetime.now(),
                    "K_LAG": K, "stride": STRIDE, "embargo_bins": EMB
                })
                print(f"[RESULT] cross-task {test_name} | {args.dimred}+{args.align} | seed={seed} fold={fold_idx} | VAF={mVAF:.4f}")

    # save all
    out = os.path.join(
        args.output_dir,
        f"strict_{args.scenario}_day{args.train_day_idx}_{args.decoder}_{args.dimred}_{args.align}_kf{args.n_folds}_seeds{','.join(map(str,seeds))}.pkl"
    )
    pd.to_pickle(pd.DataFrame(results), out)
    print(f"[INFO] saved → {out}")

if __name__ == "__main__":
    main()
