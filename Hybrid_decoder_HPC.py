import os, sys, random, datetime
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import umap
import argparse
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import iirnotch, filtfilt
# ────────────── GLOBALS ──────────────
DECODER_CONFIG = {
    "GRU":    {"N_PCA": 14, "HIDDEN_DIM": 5,  "K_LAG": 16, "LEARNING_RATE": 1e-3, "NUM_EPOCHS": 300, "BATCH_SIZE": 64},
    "LSTM":   {"N_PCA": 14, "HIDDEN_DIM": 16, "K_LAG": 16, "LEARNING_RATE": 1e-3, "NUM_EPOCHS": 300, "BATCH_SIZE": 64},
    "LIN":    {"N_PCA": 18, "HIDDEN_DIM": 64, "K_LAG": 16, "LEARNING_RATE": 1e-3, "NUM_EPOCHS": 300, "BATCH_SIZE": 64},
    "LiGRU":  {"N_PCA": 14, "HIDDEN_DIM": 5,  "K_LAG": 16, "LEARNING_RATE": 1e-3, "NUM_EPOCHS": 300, "BATCH_SIZE": 64},
}
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05
CV_HOLDOUT_RATIO = 0.33

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── Reproducibility ──────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ────────── Preprocessing ──────────
def smooth_spike_data(x, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    sigma = (smoothing_length / bin_size) / 2
    out = np.zeros_like(x, dtype=float)
    for ch in range(x.shape[1]):
        out[:, ch] = gaussian_filter1d(x[:, ch], sigma=sigma)
    return out

def notch_filter_emg(a, fs=1000, notch_freq=60, Q=30):
    b, c = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, c, a, axis=0)

def smooth_emg(a, window_size=5):
    rect = np.abs(a)
    filt = notch_filter_emg(rect)
    return np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window_size)/window_size, mode="same"),
        0, filt)

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
        sp, emg = row["spike_counts"], row["EMG"]
        if not isinstance(sp, pd.DataFrame) or sp.empty: continue
        sp = sp.reindex(expected, axis=1, fill_value=0)
        Xs.append(smooth_spike_data(sp.values))
        if isinstance(emg, pd.DataFrame):
            e = emg
            if reference_emg_cols is not None:
                e = e.reindex(reference_emg_cols, axis=1, fill_value=0)
            Ys.append(smooth_emg(e.values))
        else:
            Ys.append(smooth_emg(np.asarray(emg)))
    if not Xs: return np.empty((0,)), np.empty((0,))
    return np.concatenate(Xs), np.concatenate(Ys)

# ────────── CV Split: Random Sample Holdout ──────────
def random_holdout_split(X, Y, holdout_ratio, seed):
    n = X.shape[0]
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    rs.shuffle(idx)
    cut = int(n * (1 - holdout_ratio))
    train_idx, hold_idx = idx[:cut], idx[cut:]
    return X[train_idx], Y[train_idx], X[hold_idx], Y[hold_idx]

# ────────── Sequence constructors ──────────
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

# ────────── Dimensionality Reduction ──────────
def get_day_dimred(df, n_components, method="PCA"):
    X_day, _ = build_continuous_dataset_preprocessed(df)
    if X_day.shape[0] == 0:
        return None
    if method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reducer.fit(X_day)
        return reducer
    else:
        pca_model = PCA(n_components=n_components, random_state=42)
        pca_model.fit(X_day)
        return pca_model

def transform_dimred(model, X, method="PCA"):
    return model.transform(X)

# ────────── Model Definitions ──────────
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

# ────────── Training Function ──────────
def train_model(model, X_train, Y_train, num_epochs, batch_size, lr):
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

# ────────── Evaluation Functions ──────────
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
        for i in range(0, len(X_val), DECODER_CONFIG['GRU']["BATCH_SIZE"]):  # Use the largest batch size for safety
            batch_X = torch.tensor(X_val[i:i+DECODER_CONFIG['GRU']["BATCH_SIZE"]], dtype=torch.float32).to(DEVICE)
            out = model(batch_X)
            preds.append(out.cpu().numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
    else:
        preds = np.empty((0,))
    vaf_ch = compute_multichannel_vaf(Y_val, preds)
    mean_vaf = np.nanmean(vaf_ch)
    return preds, vaf_ch, mean_vaf

# ────────── Main Pipeline ──────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, choices=['GRU', 'LSTM', 'LIN', 'LiGRU'], required=True)
    parser.add_argument('--dim_red', type=str, choices=['PCA', 'UMAP'], required=True)
    parser.add_argument('--cv_fold', type=int, required=True)
    parser.add_argument('--n_cv_folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--combined_pickle_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    decoder_type = args.decoder
    dim_red = args.dim_red
    cv_fold = args.cv_fold
    N_CV_FOLDS = args.n_cv_folds
    seed = args.seed
    combined_pickle_file = args.combined_pickle_file
    output_dir = args.output_dir

    conf = DECODER_CONFIG[decoder_type]
    set_seed(seed + cv_fold)

    if not os.path.exists(combined_pickle_file):
        print(f"[ERROR] cannot find {combined_pickle_file}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    df_raw = pd.read_pickle(combined_pickle_file)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df, ref_cols = filter_and_map_emg(df_raw)

    all_results = []

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

    for sc in scenarios:
        df_scenario = df[df.apply(sc["train_filter"], axis=1)].copy()
        if df_scenario.empty:
            print(f"[WARNING] no trials for scenario {sc['name']} – skip")
            continue

        for day in sorted(df_scenario["date"].dropna().unique()):
            df_day = df_scenario[df_scenario["date"] == day].copy()
            if df_day.empty:
                continue

            dstr = day.strftime("%Y-%m-%d")
            print(f"\n=== {sc['name']} | {dstr} | {len(df_day)} trials ===", flush=True)

            # --- Preprocess whole day's data
            X_full, Y_full = build_continuous_dataset_preprocessed(df_day, reference_emg_cols=ref_cols)
            if X_full.shape[0] == 0:
                print("  [WARNING] empty data – skip day")
                continue

            print(f"  [CV] Fold {cv_fold+1}/{N_CV_FOLDS}", flush=True)
            X_train, Y_train, X_hold, Y_hold = random_holdout_split(
                X_full, Y_full, holdout_ratio=CV_HOLDOUT_RATIO, seed=seed + cv_fold
            )

            print(f"   [INFO] Dimensionality reduction: {dim_red}", flush=True)
            max_n_pca = conf["N_PCA"]
            model_dimred = get_day_dimred(
                pd.DataFrame([{"spike_counts": pd.DataFrame(X_train), "EMG": pd.DataFrame(Y_train)}]),
                n_components=max_n_pca, method=dim_red, seed=seed+cv_fold
            )
            Z_train = transform_dimred(model_dimred, X_train, method=dim_red)
            Z_hold  = transform_dimred(model_dimred, X_hold, method=dim_red)

            # Dataset for current decoder
            if decoder_type == "LIN":
                Xtr, Ytr = create_linear_dataset(Z_train[:,:conf["N_PCA"]], Y_train, conf["K_LAG"])
                Xte, Yte = create_linear_dataset(Z_hold[:,:conf["N_PCA"]], Y_hold, conf["K_LAG"])
                decoder = LinearLagDecoder(conf["N_PCA"]*conf["K_LAG"], conf["HIDDEN_DIM"], Ytr.shape[1]).to(DEVICE)
            elif decoder_type == "GRU":
                Xtr, Ytr = create_rnn_dataset(Z_train[:,:conf["N_PCA"]], Y_train, conf["K_LAG"])
                Xte, Yte = create_rnn_dataset(Z_hold[:,:conf["N_PCA"]], Y_hold, conf["K_LAG"])
                decoder = GRUDecoder(conf["N_PCA"], conf["HIDDEN_DIM"], Ytr.shape[1]).to(DEVICE)
            elif decoder_type == "LSTM":
                Xtr, Ytr = create_rnn_dataset(Z_train[:,:conf["N_PCA"]], Y_train, conf["K_LAG"])
                Xte, Yte = create_rnn_dataset(Z_hold[:,:conf["N_PCA"]], Y_hold, conf["K_LAG"])
                decoder = LSTMDecoder(conf["N_PCA"], conf["HIDDEN_DIM"], Ytr.shape[1]).to(DEVICE)
            elif decoder_type == "LiGRU":
                Xtr, Ytr = create_rnn_dataset(Z_train[:,:conf["N_PCA"]], Y_train, conf["K_LAG"])
                Xte, Yte = create_rnn_dataset(Z_hold[:,:conf["N_PCA"]], Y_hold, conf["K_LAG"])
                decoder = LiGRUDecoder(conf["N_PCA"], conf["HIDDEN_DIM"], Ytr.shape[1]).to(DEVICE)
            else:
                continue

            if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
                print("  [WARNING] skip empty train/test")
                continue

            print(f"      training {decoder_type}…", flush=True)
            decoder = train_model(
                decoder, Xtr, Ytr,
                num_epochs=conf["NUM_EPOCHS"],
                batch_size=conf["BATCH_SIZE"],
                lr=conf["LEARNING_RATE"]
            )
            preds, vaf_ch, mean_vaf = evaluate_decoder(
                decoder, Xte, Yte, batch_size=conf["BATCH_SIZE"]
            )

            # Save result for this fold/decoder/dim_red
            all_results.append({
                "scenario_name" : sc["name"],
                "monkey"        : df_day.iloc[0]["monkey"],
                "train_task"    : "hybrid",
                "test_task"     : "hybrid",
                "decoder_type"  : decoder_type,
                "dim_red"       : dim_red,
                "mean_VAF"      : mean_vaf,
                "VAF_channels"  : vaf_ch,
                "cv_fold"       : cv_fold,
                "seed"          : seed + cv_fold,
                "n_pca"         : conf["N_PCA"],
                "hidden_dim"    : conf["HIDDEN_DIM"],
                "k_lag"         : conf["K_LAG"],
                "learning_rate" : conf["LEARNING_RATE"],
                "epochs"        : conf["NUM_EPOCHS"],
                "batch_size"    : conf["BATCH_SIZE"],
                "timestamp"     : datetime.datetime.now(),
                "train_date"    : day,
                "test_date"     : day,
                "date"          : day,
            })

            # Save output after each day for robustness
            fname = f"{sc['name']}_{decoder_type}_{dim_red}_cv{cv_fold}_{day.strftime('%Y%m%d')}.pkl"
            path_out = os.path.join(output_dir, fname)
            pd.DataFrame(all_results).to_pickle(path_out)
            print(f"  [SAVED] {path_out}", flush=True)


if __name__ == "__main__":
    main()
