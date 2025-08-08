import os, sys, random, datetime, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ──────────────── UMAP import ────────────────
try:
    import umap
except ImportError:
    umap = None

# ─────────────────────────── globals ────────────────────────────
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_PICKLE_FILE = "output.pkl"
SAVE_RESULTS_PATH    = "df_results_emg_validation_hybrid_200.pkl" 
BIN_SIZE = 0.02 # 20 ms bin size
SMOOTHING_LENGTH = 0.05 # 50 ms smoothing length

DECODER_CONFIG = {
    "GRU":    {"N_PCA": 32, "HIDDEN_DIM": 96, "K_LAG": 25, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 200},
    "LSTM":   {"N_PCA": 24, "HIDDEN_DIM": 128, "K_LAG": 25, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 300},
    "LIN":    {"N_PCA": 32, "HIDDEN_DIM": 64, "K_LAG": 16, "LEARNING_RATE": 0.003, "NUM_EPOCHS": 100 },
    "LiGRU":  {"N_PCA": 32, "HIDDEN_DIM": 5,  "K_LAG": 16, "LEARNING_RATE": 0.001, "NUM_EPOCHS": 200 },
}

NUM_EPOCHS   = 300
BATCH_SIZE   = 64
LEARNING_RATE= 1e-3

# ─────────────────── reproducibility helpers ────────────────────
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ──────────────────────── preprocessing ─────────────────────────
def smooth_spike_data(x, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    sigma = (smoothing_length / bin_size) / 2
    out = np.zeros_like(x, dtype=float)
    for ch in range(x.shape[1]):
        out[:, ch] = gaussian_filter1d(x[:, ch], sigma=sigma)
    return out

def butter_lowpass(data, fs, fc=5, order=4):
    nyq = fs / 2
    if fc >= nyq:
        raise ValueError(f"fc ({fc} Hz) doit être < à la fréquence de Nyquist ({nyq} Hz) pour fs={fs}")
    b, a = butter(order, fc / nyq, "low")
    return filtfilt(b, a, data, axis=0)

def smooth_emg(a, bin_width):
    fs = 1.0 / bin_width     # exemple : 1/0.02 = 50Hz
    rect = np.abs(a)
    filt = butter_lowpass(rect, fs=fs, fc=5, order=4)
    return filt

def map_emg_labels(emg_df):
    TARGET = {"FCR","FDS","FDP","FCU","ECR","EDC","ECU"}
    MAP = {'ECR_1':'ECR','ECR_2':'ECR','EDC_1':'EDC','EDC_2':'EDC',
           'FCR_1':'FCR','FCU_1':'FCU','FDS_1':'FDS','FDS_2':'FDS',
           'FDP_1':'FDP','FDP_2':'FDP','ECU_1':'ECU'}
    out, cnt = {}, defaultdict(int)
    for col in emg_df.columns:
        raw,tmp = col.strip().upper(), col.strip().upper()
        while tmp and tmp not in MAP: tmp = tmp[:-1]
        base = MAP.get(tmp, None)
        if base and base in TARGET:
            cnt[base]+=1; out[f"{base}_{cnt[base]}"] = emg_df[col]
    return pd.DataFrame(out)

def filter_and_map_emg(df):
    rows, cols = [], set()
    for _,row in df.iterrows():
        emg = row.get("EMG")
        if isinstance(emg, pd.DataFrame) and not emg.empty:
            mapped = map_emg_labels(emg)
            row["EMG"] = mapped
            cols.update(mapped.columns)
        rows.append(row)
    df2 = pd.DataFrame(rows)
    cols_sorted = sorted(cols)
    for idx,row in df2.iterrows():
        emg = row.get("EMG")
        if isinstance(emg, pd.DataFrame):
            row["EMG"] = emg.reindex(cols_sorted, fill_value=0)
    return df2, cols_sorted

def build_continuous_dataset_preprocessed(df, reference_emg_cols=None):
    Xs, Ys = [], []
    expected = [f"neuron{i}" for i in range(1,97)]
    for _,row in df.iterrows():
        bin_width = row.get("bin_width", BIN_SIZE)
        sp, emg = row["spike_counts"], row["EMG"]
        if not isinstance(sp, pd.DataFrame) or sp.empty: continue
        sp = sp.reindex(expected, axis=1, fill_value=0)
        Xs.append(smooth_spike_data(sp.values))
        if isinstance(emg, pd.DataFrame):
            e = emg
            if reference_emg_cols is not None:
                e = e.reindex(reference_emg_cols, axis=1, fill_value=0)
            Ys.append(smooth_emg(e.values, bin_width))
        else:
            Ys.append(smooth_emg(np.asarray(emg), bin_width))
    if not Xs: return np.empty((0,)), np.empty((0,))
    return np.concatenate(Xs), np.concatenate(Ys)

# ────────────── manifold fitters: PCA or UMAP ──────────────
def fit_manifold(X, method='pca', n_components=10, random_state=SEED):
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
    elif method == 'umap':
        if umap is None:
            raise RuntimeError("umap-learn not installed. Run 'pip install umap-learn'")
        model = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=30, min_dist=0.1)
    else:
        raise ValueError(f"Unknown manifold method: {method}")
    model.fit(X)
    return model

def manifold_transform(model, X, method='pca'):
    if method == 'pca':
        return model.transform(X)
    elif method == 'umap':
        return model.transform(X)
    else:
        raise ValueError(f"Unknown manifold method: {method}")

# ───────────────────────── split helper ─────────────────────────
def hybrid_time_based_split(df_day: pd.DataFrame, split_ratio: float=0.5):
    train_rows, hold_rows = [], []
    for _,row in df_day.iterrows():
        sp   = row["spike_counts"];  emg = row["EMG"]
        cut  = max(1, int(len(sp)*split_ratio))
        r_tr = row.copy(); r_ho = row.copy()
        r_tr["spike_counts"] = sp.iloc[:cut ].reset_index(drop=True)
        r_ho["spike_counts"] = sp.iloc[cut:].reset_index(drop=True)
        if isinstance(emg, pd.DataFrame):
            r_tr["EMG"] = emg.iloc[:cut ].reset_index(drop=True)
            r_ho["EMG"] = emg.iloc[cut:].reset_index(drop=True)
        train_rows.append(r_tr); hold_rows.append(r_ho)
    return pd.DataFrame(train_rows), pd.DataFrame(hold_rows)

def blocked_kfold_indices(n_samples, n_splits=5):
    block_size = n_samples // n_splits
    splits = []
    for k in range(n_splits):
        val_start = k * block_size
        val_end = (k + 1) * block_size if k < n_splits - 1 else n_samples
        idx_val = np.arange(val_start, val_end)
        idx_train = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
        splits.append((idx_train, idx_val))
    return splits
# ───────────────────── sequence constructors ───────────────────
def create_rnn_dataset(X, Y, k):
    if X.shape[0]<=k: return np.empty((0,k,X.shape[1])), np.empty((0,Y.shape[1]))
    Xo,Yo=[],[]
    for t in range(k, X.shape[0]):
        Xo.append(X[t-k:t]); Yo.append(Y[t])
    return np.asarray(Xo,np.float32), np.asarray(Yo,np.float32)

def create_linear_dataset(X, Y, k):
    if X.shape[0]<=k: return np.empty((0,k*X.shape[1])), np.empty((0,Y.shape[1]))
    Xo,Yo=[],[]
    for t in range(k, X.shape[0]):
        Xo.append(X[t-k:t].reshape(-1)); Yo.append(Y[t])
    return np.asarray(Xo,np.float32), np.asarray(Yo,np.float32)

def build_day_decoder_data(df, manifold_model, n_manifold, k, is_linear, ref_cols, method='pca'):
    Xbig,Ybig = build_continuous_dataset_preprocessed(df, ref_cols)
    if Xbig.size==0: return np.empty((0,)), np.empty((0,))
    Z = manifold_transform(manifold_model, Xbig, method)[:,:n_manifold]
    if is_linear: return create_linear_dataset(Z,Ybig,k)
    return create_rnn_dataset(Z,Ybig,k)

def get_model_for_decoder(dec_name, n_pca, cfg, n_out):
    if dec_name == "GRU":
        return GRUDecoder(n_pca, cfg["HIDDEN_DIM"], n_out).to(DEVICE)
    elif dec_name == "LSTM":
        return LSTMDecoder(n_pca, cfg["HIDDEN_DIM"], n_out).to(DEVICE)
    elif dec_name == "LIN":
        return LinearLagDecoder(n_pca * cfg["K_LAG"], cfg["HIDDEN_DIM"], n_out).to(DEVICE)
    elif dec_name == "LiGRU":
        return LiGRUDecoder(n_pca, cfg["HIDDEN_DIM"], n_out).to(DEVICE)
    else:
        raise ValueError(f"Unknown decoder type: {dec_name}")
# ---------------- Model Definitions ----------------
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
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
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

# ---------------- Training Function ----------------
def train_model(model, X_train, Y_train, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(Y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{num_epochs}: Loss = {total_loss / len(loader):.4f}")
    return model

# ---------------- Evaluation Functions ----------------
def compute_vaf_1d(y_true, y_pred):
    var_true = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    var_resid = np.var(y_true - y_pred)
    return 1.0 - (var_resid / var_true)

def compute_multichannel_vaf(y_true, y_pred):
    n_ch = y_true.shape[1]
    return np.array([compute_vaf_1d(y_true[:, ch], y_pred[:, ch]) for ch in range(n_ch)])

def evaluate_decoder(model, X_val, Y_val, context=""):
    print(f"[DEBUG]{context} - Evaluating {model.__class__.__name__}: X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch_X = torch.tensor(X_val[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            out = model(batch_X)
            preds.append(out.cpu().numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
    else:
        preds = np.empty((0,))
    vaf_ch = compute_multichannel_vaf(Y_val, preds)
    mean_vaf = np.nanmean(vaf_ch)
    return preds, vaf_ch, mean_vaf

# ---------------- Main Pipeline ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifold', default='pca', choices=['pca', 'umap'], help='Manifold learning method')
    parser.add_argument('--output', default=SAVE_RESULTS_PATH)
    parser.add_argument('--input', default=COMBINED_PICKLE_FILE)
    parser.add_argument('--kfold', action='store_true', help='Active k-fold cross-validation (blocs temporels)')
    parser.add_argument('--n_splits', type=int, default=5, help='Nombre de folds pour k-fold')
    parser.add_argument('--decoder', default=None, choices=list(DECODER_CONFIG.keys()) + [None], help='Nom du décodeur à tester (ou tous)')
    args = parser.parse_args()

    set_seed()
    print("device:", DEVICE)
    

    # ---------- 1) Chargement & prétraitement -----------
    if not os.path.exists(args.input):
        print(f"[ERROR] cannot find {args.input}")
        sys.exit(1)
    df_raw = pd.read_pickle(args.input)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df, ref_cols = filter_and_map_emg(df_raw)
    all_results = []

    # ---------- 2) scenario definitions ---------------------------------------
    scenarios = [
        {   # Jango : iso / wm / spr
            "name"        : "Jango_all",
            "train_filter": lambda r: r["monkey"] == "Jango",
            "tests"       : [
                {"name": "iso", "test_filter": lambda r: r["task"].strip().lower() == "iso"},
                {"name": "wm" , "test_filter": lambda r: r["task"].strip().lower() == "wm"},
                {"name": "spr", "test_filter": lambda r: r["task"].strip().lower() == "spr"},
            ],
        },
        {   # JacB  : iso / wm / spr
            "name"        : "JacB_all",
            "train_filter": lambda r: r["monkey"] == "JacB",
            "tests"       : [
                {"name": "iso", "test_filter": lambda r: r["task"].strip().lower() == "iso"},
                {"name": "wm" , "test_filter": lambda r: r["task"].strip().lower() == "wm"},
                {"name": "spr", "test_filter": lambda r: r["task"].strip().lower() == "spr"},
            ],
        },
        {   # Jaco  : mg‑pt / ball
            "name"        : "Jaco_all",
            "train_filter": lambda r: r["monkey"] == "Jaco",
            "tests"       : [
                {"name": "mgpt", "test_filter": lambda r: r["task"].strip().lower() in ["mgpt", "mg-pt"]},
                {"name": "ball", "test_filter": lambda r: r["task"].strip().lower() == "ball"},
            ],
        },
        {   # Theo  : mg‑pt / ball
            "name"        : "Theo_all",
            "train_filter": lambda r: r["monkey"] == "Theo",
            "tests"       : [
                {"name": "mgpt", "test_filter": lambda r: r["task"].strip().lower() in ["mgpt", "mg-pt"]},
                {"name": "ball", "test_filter": lambda r: r["task"].strip().lower() == "ball"},
            ],
        },
    ]
    print(f"[INFO] {len(scenarios)} scenarios to process")

    decoders_to_run = [args.decoder] if args.decoder else list(DECODER_CONFIG.keys())

    for dec_name in decoders_to_run:
        cfg = DECODER_CONFIG[dec_name]
        print(f"\n============ {dec_name} ============")
        for sc in scenarios:
            df_scenario = df[df.apply(sc["train_filter"], axis=1)].copy()
            if df_scenario.empty:
                print(f"[WARNING] no trials for scenario {sc['name']} – skip")
                continue
            for day in sorted(df_scenario["date"].dropna().unique()):
                df_day = df_scenario[df_scenario["date"] == day].copy()
                if df_day.empty: continue
                dstr = day.strftime("%Y-%m-%d")
                print(f"\n=== {sc['name']} | {dstr} | {len(df_day)} trials ===")

                # --- Concatène X/Y pour toute la journée ---
                Xbig, Ybig = build_continuous_dataset_preprocessed(df_day, reference_emg_cols=ref_cols)
                if Xbig.size == 0:
                    print("  [WARNING] empty training slice – skip day")
                    continue

                max_n_pca = max(cfg2["N_PCA"] for cfg2 in DECODER_CONFIG.values())
                manifold_model = fit_manifold(Xbig, method=args.manifold, n_components=max_n_pca)
                Zbig = manifold_model.transform(Xbig)
                n_pca = cfg["N_PCA"]
                k_lag = cfg["K_LAG"]
                is_linear = (dec_name == "LIN")
                # Création du dataset séquentiel (X, Y pour tout le jour)
                if is_linear:
                    X_all, Y_all = create_linear_dataset(Zbig[:, :n_pca], Ybig, k_lag)
                else:
                    X_all, Y_all = create_rnn_dataset(Zbig[:, :n_pca], Ybig, k_lag)
                n_samples = X_all.shape[0]
                if n_samples == 0: continue

                if args.kfold:
                    splits = blocked_kfold_indices(n_samples, n_splits=args.n_splits)
                    vafs = []
                    for i_fold, (idx_train, idx_val) in enumerate(splits):
                        print(f"[Fold {i_fold+1}/{args.n_splits}]")
                        X_train, Y_train = X_all[idx_train], Y_all[idx_train]
                        X_val, Y_val = X_all[idx_val], Y_all[idx_val]
                        model = get_model_for_decoder(dec_name, n_pca, cfg, Y_all.shape[1])
                        model = train_model(model, X_train, Y_train, num_epochs=cfg["NUM_EPOCHS"], batch_size=BATCH_SIZE, lr=cfg["LEARNING_RATE"])
                        _, vaf_ch, mVAF = evaluate_decoder(model, X_val, Y_val, context=f"{dec_name}-fold{i_fold+1}")
                        vafs.append(mVAF)
                        # Si tu veux les vafs channels : stocke vaf_ch ici aussi
                    mean_vaf = np.mean(vafs)
                    print(f"== {dec_name} | {day} | Cross-val mean VAF: {mean_vaf:.4f} ==")
                    all_results.append({
                        "scenario_name": sc["name"],
                        "train_monkey": df_day.iloc[0]["monkey"],
                        "test_monkey": df_day.iloc[0]["monkey"],
                        "train_task": "crossval",
                        "test_task": "crossval",
                        "decoder_type": dec_name,
                        "fold_mean_VAF": mean_vaf,
                        "fold_VAFs": vafs,
                        "emg_labels": ref_cols,
                        "timestamp": datetime.datetime.now(),
                        "train_date": day,
                        "test_date": day,
                        "date": day,
                    })
                else:
                    split_point = int(n_samples * sc["split_ratio"])
                    X_train, Y_train = X_all[:split_point], Y_all[:split_point]
                    X_val, Y_val     = X_all[split_point:], Y_all[split_point:]
                    model = get_model_for_decoder(dec_name, n_pca, cfg, Y_all.shape[1])
                    model = train_model(model, X_train, Y_train, num_epochs=cfg["NUM_EPOCHS"], batch_size=BATCH_SIZE, lr=cfg["LEARNING_RATE"])
                    _, vaf_ch, mVAF = evaluate_decoder(model, X_val, Y_val, context=f"{dec_name}-holdout")
                    all_results.append({
                        "scenario_name": sc["name"],
                        "train_monkey": df_day.iloc[0]["monkey"],
                        "test_monkey": df_day.iloc[0]["monkey"],
                        "train_task": "hybrid",
                        "test_task": "hybrid",
                        "decoder_type": dec_name,
                        "mean_VAF": mVAF,
                        "per_channel_VAF": vaf_ch,
                        "emg_labels": ref_cols,
                        "timestamp": datetime.datetime.now(),
                        "train_date": day,
                        "test_date": day,
                        "date": day,
                    })
    # --- Sauvegarde ---
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_pickle(args.output)
        print(f"\n[INFO] saved results → {args.output}")
    else:
        print("\n[WARNING] nothing to save – no results collected")

if __name__ == "__main__":
    main()
