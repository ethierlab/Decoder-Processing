#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full cross-task EMG decoder validation with PCA/UMAP, alignment, and perf tweaks.

What this script does (short):
  • Loads data (output.pkl), maps EMG channels consistently across days.
  • Defines scenarios (train task → test other task(s), same day by default).
  • Trains requested decoders on a chosen training day for the train task.
  • Tests cross-task on the same day (or across days if force_same_day=False),
    with multiple alignment modes and either PCA or UMAP dim. reduction.
  • Saves per-architecture mean VAF and per-channel VAFs to a pickle.

Key features added vs your baseline:
  • UMAP support (–-dimred umap).  
  • Alignment options:
      - none: direct projection via train manifold
      - latent: Moore–Penrose in latent space (works for PCA & UMAP)
      - pca_basis: pinv on PCA bases (PCA only)
  • Perf goodies: TF32 (optional), DataLoader num_workers/pin_memory, AMP, stride windowing.
  • Lower leakage risk on INTERNAL HOLDOUT: when split_ratio>0, the manifold is
    fit on the training split only; holdout never touches the fit.

CLI examples:
  python emg_cross_task_validation_umap_align_perf.py \
    --input output.pkl --scenario Jaco_mgpt --train_day_idx 0 \
    --decoder LSTM --dimred umap --align latent --perf_mode

  python emg_cross_task_validation_umap_align_perf.py \
    --input output.pkl --scenario Jango_wm --train_day_idx 0 \
    --decoder GRU --dimred pca --align pca_basis --stride_mul 1.0

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
SMOOTHING = 0.05  # 50 ms

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

# ───────────────────────── Preprocessing ─────────────────────────

def smooth_spike_data(X, bin_size=BIN_SIZE, smoothing=SMOOTHING):
    sigma=(smoothing/bin_size)/2.0
    return gaussian_filter1d(X.astype(np.float32), sigma=sigma, axis=0)

def smooth_emg(Y, fs=50, rect_notch=False):
    # by default: rectify + 5 Hz low-pass
    rect = np.abs(Y)
    if rect_notch:
        b_notch, a_notch = iirnotch(60, 30, fs)
        rect = filtfilt(b_notch, a_notch, rect, axis=0)
    b,a = butter(4, 5.0/(fs/2), 'low')
    return filtfilt(b,a, rect, axis=0)

def build_continuous_dataset(df, ref_cols=None):
    Xs, Ys = [], []
    expected=[f"neuron{i}" for i in range(1,97)]
    for _,r in df.iterrows():
        sp=r.get("spike_counts"); emg=r.get("EMG")
        if not isinstance(sp, pd.DataFrame) or sp.empty or emg is None: continue
        sp = sp.reindex(columns=expected, fill_value=0)
        Xs.append(smooth_spike_data(sp.values, BIN_SIZE, SMOOTHING))
        if isinstance(emg, pd.DataFrame):
            e = emg if ref_cols is None else emg.reindex(ref_cols, axis=1, fill_value=0)
            Ys.append(smooth_emg(e.values, fs=int(1/BIN_SIZE)))
        else:
            Ys.append(smooth_emg(np.asarray(emg), fs=int(1/BIN_SIZE)))
    if not Xs: return np.empty((0,)), np.empty((0,))
    return np.concatenate(Xs,0), np.concatenate(Ys,0)

# ───────────────────── Windowing (sequence builders) ─────────────────────

def create_rnn_dataset(Z, Y, k, stride=1):
    if Z.shape[0] <= k: return np.empty((0,k,Z.shape[1])), np.empty((0,Y.shape[1]))
    idx = range(k, Z.shape[0], stride)
    Xo = np.stack([Z[t-k:t] for t in idx]).astype(np.float32)
    Yo = np.stack([Y[t] for t in idx]).astype(np.float32)
    return Xo, Yo

def create_linear_dataset(Z, Y, k, stride=1):
    if Z.shape[0] <= k: return np.empty((0,k*Z.shape[1])), np.empty((0,Y.shape[1]))
    idx = range(k, Z.shape[0], stride)
    Xo = np.stack([Z[t-k:t].reshape(-1) for t in idx]).astype(np.float32)
    Yo = np.stack([Y[t] for t in idx]).astype(np.float32)
    return Xo, Yo

# ───────────────────── Decoders (independent classes) ─────────────────────
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
        # x: (B, T, D)  batch_first
        B, T, _ = x.size()
        # init h avec même dtype/device que x
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

def align_pca_basis(pca_train: PCA, pca_test: PCA, Z_test: np.ndarray, n_comp: int) -> np.ndarray:
    V0 = pca_train.components_[:n_comp, :].T
    Vx = pca_test.components_[:n_comp, :].T
    R  = pinv(Vx) @ V0
    return Z_test @ R

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

# ───────────────────────── Helpers ─────────────────────────

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
    ap.add_argument('--split_ratio', type=float, default=0.25, help='internal holdout ratio (0 disables)')
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--stride_mul', type=float, default=1.0, help='stride = max(1, int(stride_mul*K_LAG))')
    ap.add_argument('--num_workers', type=int, default=-1)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--perf_mode', action='store_true')
    ap.add_argument('--output_dir', default='results_emg_cross_task')
    args = ap.parse_args()

    # perf/repro
    global PERF_MODE
    if args.perf_mode: PERF_MODE = True
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    # load
    df = pd.read_pickle(args.input)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df, emg_cols = filter_and_map_emg(df)

    scen_map = {s["name"]:s for s in build_scenarios()}
    sc = scen_map[args.scenario]

    df_train_task = df[df.apply(sc["train_filter"], axis=1)].copy()
    days = sorted(df_train_task["date"].dropna().unique())
    if args.train_day_idx >= len(days):
        raise IndexError("train_day_idx out of range")
    day = days[args.train_day_idx]
    df_train_day = df_train_task[df_train_task["date"]==day].copy()

    cfg = DECODER_CONFIG[args.decoder]
    N = cfg["N_PCA"]; K = cfg["K_LAG"]; H = cfg["HIDDEN"]
    LR = args.lr if args.lr is not None else cfg["LR"]
    E  = args.epochs if args.epochs is not None else cfg["EPOCHS"]
    STRIDE = max(1, int(args.stride_mul * K))
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # === Build TRAIN raw ===
    X_raw_all, Y_raw_all = build_continuous_dataset(df_train_day, ref_cols=emg_cols)
    if X_raw_all.size == 0:
        print("[WARN] empty train"); sys.exit(0)

    # === Internal holdout without leakage: split first, fit manifold on TRAIN only ===
    split_ratio = max(0.0, min(0.99, args.split_ratio))
    if split_ratio > 0:
        nT = X_raw_all.shape[0]
        cut = int((1.0 - split_ratio) * nT)
        if cut <= max(K, 10):
            print("[WARN] too small train after split; disabling holdout")
            cut = nT; split_ratio = 0.0
        Xtr_raw, Ytr_raw = X_raw_all[:cut], Y_raw_all[:cut]
        Xho_raw, Yho_raw = X_raw_all[cut:], Y_raw_all[cut:]
    else:
        Xtr_raw, Ytr_raw = X_raw_all, Y_raw_all
        Xho_raw, Yho_raw = np.empty((0,)), np.empty((0,))

    dim_model_train = fit_dimred(Xtr_raw, method=args.dimred, n_components=max(N,2), random_state=SEED)
    Ztr_full = transform_dimred(dim_model_train, Xtr_raw, method=args.dimred)[:, :N]

    # train sequences
    if args.decoder == "Linear":
        Xtr, Ytr = create_linear_dataset(Ztr_full, Ytr_raw, K, stride=STRIDE); n_in = N*K
    else:
        Xtr, Ytr = create_rnn_dataset(Ztr_full, Ytr_raw, K, stride=STRIDE); n_in = N
    n_out = Ytr.shape[1]
    model = get_model(args.decoder, n_in, H, n_out)
    print(f"[INFO] Training {args.decoder} on {args.scenario} day={str(pd.to_datetime(day).date())} ...")
    t0=time.time()
    model = train_model(model, Xtr, Ytr, epochs=E, lr=LR, batch_size=args.batch_size, num_workers=WORKERS, use_amp=USE_AMP)
    print(f"[INFO] train time: {time.time()-t0:.1f}s")

    results=[]

    # Internal holdout eval (diagonal)
    if split_ratio > 0:
        Zho = transform_dimred(dim_model_train, Xho_raw, method=args.dimred)[:, :N]
        if args.decoder == "Linear":
            Xho, Yho = create_linear_dataset(Zho, Yho_raw, K, stride=STRIDE)
        else:
            Xho, Yho = create_rnn_dataset(Zho, Yho_raw, K, stride=STRIDE)
        if Xho.shape[0]:
            mVAF, vaf_ch = eval_decoder(model, Xho, Yho, batch_size=args.batch_size, use_amp=USE_AMP)
            results.append({
                "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                "decoder": args.decoder, "dimred": args.dimred, "align": "internal_holdout",
                "test_task": parse_test_task(args.scenario.split('_')[-1]),
                "mean_VAF": float(mVAF), "per_channel_VAF": vaf_ch,
                "timestamp": datetime.datetime.now(),
            })
            print(f"[RESULT] internal holdout | VAF={mVAF:.4f}")

    # === Cross-task (off-diagonal) ===
    df_same_day = df[df["date"]==day].copy() if sc.get("force_same_day", True) else df
    for tcfg in sc["tests"]:
        test_name = tcfg["name"]
        df_test = df_same_day[df_same_day.apply(tcfg["test_filter"], axis=1)].copy()
        if df_test.empty:
            print(f"[WARN] no test data for {test_name}"); continue
        Xte_raw, Yte_raw = build_continuous_dataset(df_test, ref_cols=emg_cols)
        if Xte_raw.size == 0:
            print(f"[WARN] empty X for {test_name}"); continue

        # direct projection via TRAIN manifold
        Zte_direct = transform_dimred(dim_model_train, Xte_raw, method=args.dimred)[:, :N]

        align_opts = [args.align] if args.align else ['none']
        for align_mode in align_opts:
            if align_mode == 'none':
                Zte = Zte_direct
            elif align_mode == 'latent':
                # fit a TEST manifold of the same method, then Moore–Penrose between latents
                dim_model_test = fit_dimred(Xte_raw, method=args.dimred, n_components=max(N,2), random_state=SEED)
                Zte_test = transform_dimred(dim_model_test, Xte_raw, method=args.dimred)[:, :N]
                Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
            elif align_mode == 'pca_basis':
                if args.dimred != 'pca':
                    print("[WARN] pca_basis requires --dimred pca; falling back to 'latent'.")
                    dim_model_test = fit_dimred(Xte_raw, method=args.dimred, n_components=max(N,2), random_state=SEED)
                    Zte_test = transform_dimred(dim_model_test, Xte_raw, method=args.dimred)[:, :N]
                    Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
                else:
                    pca_test = fit_dimred(Xte_raw, method='pca', n_components=max(N,2), random_state=SEED)
                    Zte = align_pca_basis(dim_model_train, pca_test, Zte_direct, n_comp=N)
            else:
                raise ValueError("unknown align option")

            # make windows and evaluate
            if args.decoder == "Linear":
                Xte, Yte = create_linear_dataset(Zte, Yte_raw, K, stride=STRIDE)
            else:
                Xte, Yte = create_rnn_dataset(Zte, Yte_raw, K, stride=STRIDE)
            if Xte.shape[0] == 0:
                print(f"[WARN] empty windows for {test_name} ({align_mode})"); continue
            mVAF, vaf_ch = eval_decoder(model, Xte, Yte, batch_size=args.batch_size, use_amp=USE_AMP)
            results.append({
                "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                "decoder": args.decoder, "dimred": args.dimred, "align": align_mode,
                "test_task": parse_test_task(test_name),
                "mean_VAF": float(mVAF), "per_channel_VAF": vaf_ch,
                "timestamp": datetime.datetime.now(),
            })
            print(f"[RESULT] cross-task {test_name} | {args.dimred}+{align_mode} | VAF={mVAF:.4f}")

    # save
    out = os.path.join(args.output_dir, f"cross_task_{args.scenario}_day{args.train_day_idx}_{args.decoder}_{args.dimred}_{args.align}.pkl")
    pd.to_pickle(pd.DataFrame(results), out)
    print(f"[INFO] saved → {out}")

if __name__ == "__main__":
    main()
