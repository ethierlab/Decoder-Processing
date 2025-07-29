import os, sys, random, datetime
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import iirnotch, filtfilt
import matplotlib.pyplot as plt
import umap

# ────────────── GLOBALS ──────────────
SEED                 = 42
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_PICKLE_FILE = "output.pkl"
SAVE_RESULTS_PATH    = "df_results_emg_validation_hybrid_200_CV.pkl"
BIN_SIZE             = 0.001
SMOOTHING_LENGTH     = 0.05

GRU_N_PCA   = 14 ; GRU_HIDDEN_DIM   = 5  ; GRU_K_LAG   = 16
LSTM_N_PCA  = 14 ; LSTM_HIDDEN_DIM  = 16 ; LSTM_K_LAG  = 16
LINEAR_N_PCA= 18 ; LINEAR_HIDDEN_DIM= 64 ; LINEAR_K_LAG= 16
LIGRU_N_PCA = 14 ; LIGRU_HIDDEN_DIM = 5  ; LIGRU_K_LAG = 16

NUM_EPOCHS   = 300
BATCH_SIZE   = 64
LEARNING_RATE= 1e-3

N_CV_FOLDS         = 10         
CV_HOLDOUT_RATIO   = 0.33       

# ────────── Reproducibility ──────────
def set_seed(seed=SEED):
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
        reducer = umap.UMAP(n_components=n_components, random_state=SEED)
        reducer.fit(X_day)
        return reducer
    else:
        pca_model = PCA(n_components=n_components, random_state=SEED)
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

# ────────── Main Pipeline ──────────
def main():
    set_seed()
    print("device:", DEVICE)

    if not os.path.exists(COMBINED_PICKLE_FILE):
        print(f"[ERROR] cannot find {COMBINED_PICKLE_FILE}")
        sys.exit(1)

    df_raw = pd.read_pickle(COMBINED_PICKLE_FILE)
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

            dstr = day.strftime("%Y‑%m‑%d")
            print(f"\n=== {sc['name']} | {dstr} | {len(df_day)} trials ===")

            # --- Preprocess whole day's data
            X_full, Y_full = build_continuous_dataset_preprocessed(df_day, reference_emg_cols=ref_cols)
            if X_full.shape[0] == 0:
                print("  [WARNING] empty data – skip day")
                continue

            # Cross-validation: N_CV_FOLDS random holdout splits
            for cv_fold in range(N_CV_FOLDS):
                print(f"  [CV] Fold {cv_fold+1}/{N_CV_FOLDS}")

                X_train, Y_train, X_hold, Y_hold = random_holdout_split(
                    X_full, Y_full, holdout_ratio=CV_HOLDOUT_RATIO, seed=SEED + cv_fold
                )

                for dim_red_method in ["PCA", "umap"]:
                    print(f"   [INFO] Dimensionality reduction: {dim_red_method}")

                    # Fit dim. reduction on train
                    if dim_red_method == "PCA":
                        max_n_pca = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
                    else:
                        max_n_pca = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)

                    # Fit on train set
                    model_dimred = get_day_dimred(
                        pd.DataFrame([{"spike_counts": pd.DataFrame(X_train), "EMG": pd.DataFrame(Y_train)}]),
                        n_components=max_n_pca, method=dim_red_method
                    )
                    # Project train and hold
                    Z_train = transform_dimred(model_dimred, X_train, method=dim_red_method)
                    Z_hold  = transform_dimred(model_dimred, X_hold, method=dim_red_method)

                    # Build datasets for each decoder
                    datasets = []
                    # GRU
                    Xg, Yg = create_rnn_dataset(Z_train[:,:GRU_N_PCA], Y_train, GRU_K_LAG)
                    Xgh, Ygh = create_rnn_dataset(Z_hold[:,:GRU_N_PCA], Y_hold, GRU_K_LAG)
                    datasets.append(("GRU", GRU_N_PCA, GRU_HIDDEN_DIM, GRU_K_LAG, False, Xg, Yg, Xgh, Ygh))
                    # LSTM
                    Xl, Yl = create_rnn_dataset(Z_train[:,:LSTM_N_PCA], Y_train, LSTM_K_LAG)
                    Xlh, Ylh = create_rnn_dataset(Z_hold[:,:LSTM_N_PCA], Y_hold, LSTM_K_LAG)
                    datasets.append(("LSTM", LSTM_N_PCA, LSTM_HIDDEN_DIM, LSTM_K_LAG, False, Xl, Yl, Xlh, Ylh))
                    # LIN
                    Xn, Yn = create_linear_dataset(Z_train[:,:LINEAR_N_PCA], Y_train, LINEAR_K_LAG)
                    Xnh, Ynh = create_linear_dataset(Z_hold[:,:LINEAR_N_PCA], Y_hold, LINEAR_K_LAG)
                    datasets.append(("LIN", LINEAR_N_PCA, LINEAR_HIDDEN_DIM, LINEAR_K_LAG, True, Xn, Yn, Xnh, Ynh))
                    # LiGRU
                    Xq, Yq = create_rnn_dataset(Z_train[:,:LIGRU_N_PCA], Y_train, LIGRU_K_LAG)
                    Xqh, Yqh = create_rnn_dataset(Z_hold[:,:LIGRU_N_PCA], Y_hold, LIGRU_K_LAG)
                    datasets.append(("LiGRU", LIGRU_N_PCA, LIGRU_HIDDEN_DIM, LIGRU_K_LAG, False, Xq, Yq, Xqh, Yqh))

                    for dec_name, nP, hid, lag, is_lin, Xtr, Ytr, Xte, Yte in datasets:
                        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
                            continue
                        # Build, train, eval
                        if dec_name == "GRU":
                            decoder = GRUDecoder(nP, hid, Ytr.shape[1]).to(DEVICE)
                        elif dec_name == "LSTM":
                            decoder = LSTMDecoder(nP, hid, Ytr.shape[1]).to(DEVICE)
                        elif dec_name == "LIN":
                            decoder = LinearLagDecoder(nP*lag, hid, Ytr.shape[1]).to(DEVICE)
                        elif dec_name == "LiGRU":
                            decoder = LiGRUDecoder(nP, hid, Ytr.shape[1]).to(DEVICE)
                        else:
                            continue

                        print(f"      training {dec_name}…")
                        decoder = train_model(decoder, Xtr, Ytr)
                        preds, vaf_ch, mean_vaf = evaluate_decoder(
                            decoder, Xte, Yte, context=f"{dec_name}-{dim_red_method}-CV{cv_fold+1}-{dstr}"
                        )

                        # Save result for this fold/decoder/dim_red
                        all_results.append({
                            "scenario_name" : sc["name"],
                            "monkey"        : df_day.iloc[0]["monkey"],
                            "train_task"    : "hybrid",
                            "test_task"     : "hybrid",
                            "decoder_type"  : dec_name,
                            "dim_red"       : dim_red_method,
                            "mean_VAF"      : mean_vaf,
                            "VAF_channels"  : vaf_ch,
                            "cv_fold"       : cv_fold,
                            "timestamp"     : datetime.datetime.now(),
                            "train_date"    : day,
                            "test_date"     : day,
                            "date"          : day,
                        })

    # Save all results
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_pickle(SAVE_RESULTS_PATH)
        print(f"\n[INFO] saved results → {SAVE_RESULTS_PATH}")
    else:
        print("\n[WARNING] nothing to save – no results collected")


if __name__ == "__main__":
    main()
