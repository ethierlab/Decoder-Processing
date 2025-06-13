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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─────────────────────────── globals ────────────────────────────
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_PICKLE_FILE = "output.pkl"
SAVE_RESULTS_PATH    = "df_results_emg_validation_hybrid_200.pkl"

BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05

GRU_N_PCA   = 14 ; GRU_HIDDEN_DIM   = 5  ; GRU_K_LAG   = 16
LSTM_N_PCA  = 14 ; LSTM_HIDDEN_DIM  = 16 ; LSTM_K_LAG  = 16
LINEAR_N_PCA= 18 ; LINEAR_HIDDEN_DIM= 64 ; LINEAR_K_LAG= 16
LIGRU_N_PCA = 14 ; LIGRU_HIDDEN_DIM = 5  ; LIGRU_K_LAG = 16

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

# ───────────────────────── split helper ─────────────────────────
### PATCH: brand‑new deterministic per‑trial splitter
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

def build_day_decoder_data(df, pca, n_pca, k, is_linear, ref_cols):
    Xbig,Ybig = build_continuous_dataset_preprocessed(df, ref_cols)
    if Xbig.size==0: return np.empty((0,)), np.empty((0,))
    Z = pca.transform(Xbig)[:,:n_pca] if pca is not None else Xbig[:,:n_pca]
    if is_linear: return create_linear_dataset(Z,Ybig,k)
    return create_rnn_dataset(Z,Ybig,k)

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # shuffle is OK here if trials are independent
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
# ---------------- Sequence Building Functions ----------------

def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T = X_arr.shape[0]
    for t in range(seq_len, T):
        X_out.append(X_arr[t - seq_len:t, :])
        Y_out.append(Y_arr[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len * X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T = X_arr.shape[0]
    for t in range(seq_len, T):
        window = X_arr[t - seq_len:t, :].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[t, :])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def build_dayX_decoder_data(df, day_pca_model, n_pca, seq_len, is_linear=False, 
                            reference_emg_cols=None, time_fraction=1.0, portion="first"):
    X_big, Y_big = build_continuous_dataset_preprocessed(
        df, 
        reference_emg_cols=reference_emg_cols, 
        time_fraction=time_fraction, 
        portion=portion
    )
    if X_big.shape[0] == 0:
        return np.empty((0,)), np.empty((0,))
    
    if day_pca_model is not None:
        z_full = day_pca_model.transform(X_big)
    else:
        z_full = X_big
    
    X_pca = z_full[:, :n_pca]
    
    if not is_linear:
        X_arr, Y_arr = create_rnn_dataset_continuous(X_pca, Y_big, seq_len)
    else:
        X_arr, Y_arr = create_linear_dataset_continuous(X_pca, Y_big, seq_len)
        
    return X_arr, Y_arr
# ---------------- PCA Caching ----------------

def get_day_pca(df, monkey, date, task, n_components):
    X_day, _ = build_continuous_dataset_preprocessed(df)
    if X_day.shape[0] == 0:
        return None
    pca_model = PCA(n_components=n_components, random_state=SEED)
    pca_model.fit(X_day)
    return pca_model
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

def validate_scenario(train_scenario, test_scenario, model, n_pca, seq_len, is_linear=False,
                      alignment_mode="bland", reference_emg_cols=None, force_external=False):
    # --- Internal Holdout Branch: if test DF equals train DF, use time-based split per trial ---
    if not force_external and train_scenario["df"].equals(test_scenario["df"]):
        X_train_raw, Y_train_raw = build_continuous_dataset_preprocessed(
            train_scenario["df"],
            reference_emg_cols=reference_emg_cols,
            time_fraction=0.5,
            portion="first"
        )
        print(f"[DEBUG] Training continuous data shape (first half): {X_train_raw.shape}")
        X_hold_raw, Y_hold_raw = build_continuous_dataset_preprocessed(
            train_scenario["df"],
            reference_emg_cols=reference_emg_cols,
            time_fraction=0.5,
            portion="second"
        )
        print(f"[DEBUG] Holdout continuous data shape (second half): {X_hold_raw.shape}")
        if X_train_raw.shape[0] == 0 or X_hold_raw.shape[0] == 0:
            print("One of the partitions is empty for internal holdout.")
            return None, None, None

        z_train = train_scenario["pca_model"].transform(X_train_raw)
        X_train_proj = z_train[:, :n_pca]
        z_hold = train_scenario["pca_model"].transform(X_hold_raw)
        X_hold_proj = z_hold[:, :n_pca]

        if not is_linear:
            X_train_seq, Y_train_seq = create_rnn_dataset_continuous(X_train_proj, Y_train_raw, seq_len)
            X_hold_seq, Y_hold_seq = create_rnn_dataset_continuous(X_hold_proj, Y_hold_raw, seq_len)
        else:
            X_train_seq, Y_train_seq = create_linear_dataset_continuous(X_train_proj, Y_train_raw, seq_len)
            X_hold_seq, Y_hold_seq = create_linear_dataset_continuous(X_hold_proj, Y_hold_raw, seq_len)
        
        preds, vaf_ch, mean_vaf = evaluate_decoder(model, X_hold_seq, Y_hold_seq)
        print("Internal Holdout Validation (time-based split):")
        print(f"Mean VAF: {mean_vaf:.4f}")
        return preds, vaf_ch, mean_vaf

    # --- External Evaluation Branch ---
    X_test_raw, Y_test_raw = build_continuous_dataset_preprocessed(test_scenario["df"],
                                                                    reference_emg_cols=reference_emg_cols)
    if X_test_raw.shape[0] == 0:
        print("Test data empty.")
        return None, None, None

    if alignment_mode == "bland":
        print("Alignment mode: bland")
        transform_matrix = train_scenario["pca_model"].components_.T
        X_test_proj = X_test_raw @ transform_matrix
        X_test_proj = X_test_proj[:, :n_pca]
    elif alignment_mode == "recalculated":
        print("Alignment mode: recalculated")
        test_date = test_scenario["df"].iloc[0]["date"] if "date" in test_scenario["df"].columns else None
        test_pca = get_day_pca(test_scenario["df"],
                               monkey=test_scenario["monkey"],
                               date=test_date,
                               task=test_scenario["task"],
                               n_components=n_pca)
        if test_pca is None:
            print("Test PCA failed; defaulting to bland.")
            transform_matrix = train_scenario["pca_model"].components_.T
            X_test_proj = X_test_raw @ transform_matrix
            X_test_proj = X_test_proj[:, :n_pca]
        else:
            X_test_proj = test_pca.transform(X_test_raw)[:, :n_pca]
    elif alignment_mode == "realign":
        print("Alignment mode: realign")
        test_date = test_scenario["df"].iloc[0]["date"] if "date" in test_scenario["df"].columns else None
        test_pca = get_day_pca(test_scenario["df"],
                               monkey=test_scenario["monkey"],
                               date=test_date,
                               task=test_scenario["task"],
                               n_components=n_pca)
        if test_pca is None:
            print("Test PCA failed; defaulting to bland.")
            transform_matrix = train_scenario["pca_model"].components_.T
            X_test_proj = X_test_raw @ transform_matrix
            X_test_proj = X_test_proj[:, :n_pca]
        else:
            V_train = train_scenario["pca_model"].components_.T[:, :n_pca]
            V_test  = test_pca.components_.T[:, :n_pca]
            R = pinv(V_test) @ V_train
            z_test_local = test_pca.transform(X_test_raw)
            X_test_proj = (z_test_local[:, :n_pca]) @ R
    else:
        print(f"Unknown alignment mode: {alignment_mode}; defaulting to bland.")
        transform_matrix = train_scenario["pca_model"].components_.T
        X_test_proj = X_test_raw @ transform_matrix
        X_test_proj = X_test_proj[:, :n_pca]
    
    if not is_linear:
        X_val, Y_val = create_rnn_dataset_continuous(X_test_proj, Y_test_raw, seq_len)
    else:
        X_val, Y_val = create_linear_dataset_continuous(X_test_proj, Y_test_raw, seq_len)
    
    if X_val.shape[0] == 0:
        print("Insufficient test samples after sequence construction.")
        return None, None, None
    preds, vaf_ch, mean_vaf = evaluate_decoder(model, X_val, Y_val)
    return preds, vaf_ch, mean_vaf

def hybrid_time_based_split(df_day: pd.DataFrame, split_ratio: float = 0.5):
    """
    For every trial (row) in df_day, keep the first `split_ratio`
    of samples as 'train' and the rest as 'holdout'.  
    Returned DataFrames have exactly the same columns as the input.

    Returns
    -------
    df_train : DataFrame
    df_hold  : DataFrame
    """
    train_rows, hold_rows = [], []

    for _, row in df_day.iterrows():
        row_train, row_hold = row.copy(), row.copy()

        # --- split spike counts ---
        spikes: pd.DataFrame = row["spike_counts"]
        n = len(spikes)
        cut = max(1, int(n * split_ratio))           # at least 1 sample
        row_train["spike_counts"] = spikes.iloc[:cut].reset_index(drop=True)
        row_hold ["spike_counts"] = spikes.iloc[cut:].reset_index(drop=True)

        # --- split EMG (if present) ---
        emg = row.get("EMG", None)
        if isinstance(emg, pd.DataFrame):
            row_train["EMG"] = emg.iloc[:cut].reset_index(drop=True)
            row_hold ["EMG"] = emg.iloc[cut:].reset_index(drop=True)

        train_rows.append(row_train)
        hold_rows.append(row_hold)

    return pd.DataFrame(train_rows), pd.DataFrame(hold_rows)
# ---------------- Main Pipeline ----------------
# ────────────────────────────── MAIN ──────────────────────────────
def main():
    set_seed()
    print("device:", DEVICE)

    # ---------- 1) load & basic preprocessing ---------------------------------
    if not os.path.exists(COMBINED_PICKLE_FILE):
        print(f"[ERROR] cannot find {COMBINED_PICKLE_FILE}")
        sys.exit(1)

    df_raw = pd.read_pickle(COMBINED_PICKLE_FILE)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    # map / filter EMG, returns cleaned df and a reference list of EMG columns
    df, ref_cols = filter_and_map_emg(df_raw)

    all_results = []

    # ---------- 2) scenario definitions ---------------------------------------
    scenarios = [
        {   # Jango : iso / wm / spr
            "name"        : "Jango_all",
            "train_filter": lambda r: r["monkey"] == "Jango",
            "split_ratio" : 0.3,
            "tests"       : [
                {"name": "iso", "test_filter": lambda r: r["task"].strip().lower() == "iso"},
                {"name": "wm" , "test_filter": lambda r: r["task"].strip().lower() == "wm"},
                {"name": "spr", "test_filter": lambda r: r["task"].strip().lower() == "spr"},
            ],
        },
        {   # JacB  : iso / wm / spr
            "name"        : "JacB_all",
            "train_filter": lambda r: r["monkey"] == "JacB",
            "split_ratio" : 0.3,
            "tests"       : [
                {"name": "iso", "test_filter": lambda r: r["task"].strip().lower() == "iso"},
                {"name": "wm" , "test_filter": lambda r: r["task"].strip().lower() == "wm"},
                {"name": "spr", "test_filter": lambda r: r["task"].strip().lower() == "spr"},
            ],
        },
        {   # Jaco  : mg‑pt / ball
            "name"        : "Jaco_all",
            "train_filter": lambda r: r["monkey"] == "Jaco",
            "split_ratio" : 0.3,
            "tests"       : [
                {"name": "mgpt", "test_filter": lambda r: r["task"].strip().lower() in ["mgpt", "mg-pt"]},
                {"name": "ball", "test_filter": lambda r: r["task"].strip().lower() == "ball"},
            ],
        },
        {   # Theo  : mg‑pt / ball
            "name"        : "Theo_all",
            "train_filter": lambda r: r["monkey"] == "Theo",
            "split_ratio" : 0.3,
            "tests"       : [
                {"name": "mgpt", "test_filter": lambda r: r["task"].strip().lower() in ["mgpt", "mg-pt"]},
                {"name": "ball", "test_filter": lambda r: r["task"].strip().lower() == "ball"},
            ],
        },
    ]

    print(f"[INFO] {len(scenarios)} scenarios to process")

    # ---------- 3) loop: scenario → day ---------------------------------------
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

            # ---------- 3a) hybrid split (per‑trial first half vs second half) --
            df_train_part, df_hold_part = hybrid_time_based_split(
                df_day, split_ratio=sc["split_ratio"]
            )

            # ---------- 3b) PCA fit on TRAIN slice -----------------------------
            Xtr_raw, _ = build_continuous_dataset_preprocessed(
                df_train_part, reference_emg_cols=ref_cols
            )
            if Xtr_raw.size == 0:
                print("  [WARNING] empty training slice – skip day")
                continue

            max_n_pca = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
            pca = PCA(n_components=max_n_pca, random_state=SEED).fit(Xtr_raw)

            # ---------- 3c) build TRAIN datasets -------------------------------
            Xg, Yg = build_day_decoder_data(df_train_part, pca,
                                            GRU_N_PCA, GRU_K_LAG, False,
                                            ref_cols)
            Xl, Yl = build_day_decoder_data(df_train_part, pca,
                                            LSTM_N_PCA, LSTM_K_LAG, False,
                                            ref_cols)
            Xn, Yn = build_day_decoder_data(df_train_part, pca,
                                            LINEAR_N_PCA, LINEAR_K_LAG, True,
                                            ref_cols)
            Xq, Yq = build_day_decoder_data(df_train_part, pca,
                                            LIGRU_N_PCA, LIGRU_K_LAG, False,
                                            ref_cols)

            n_out = Yg.shape[1]  # same for all

            # ---------- 3d) instantiate & train decoders -----------------------
            gru   = GRUDecoder   (GRU_N_PCA , GRU_HIDDEN_DIM , n_out).to(DEVICE)
            lstm  = LSTMDecoder  (LSTM_N_PCA, LSTM_HIDDEN_DIM, n_out).to(DEVICE)
            lin   = LinearLagDecoder(LINEAR_N_PCA * LINEAR_K_LAG,
                                     LINEAR_HIDDEN_DIM, n_out).to(DEVICE)
            ligru = LiGRUDecoder (LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_out).to(DEVICE)

            print("   training GRU  …");   gru   = train_model(gru , Xg, Yg)
            print("   training LSTM …");   lstm  = train_model(lstm, Xl, Yl)
            print("   training LIN  …");   lin   = train_model(lin , Xn, Yn)
            print("   training LiGRU…");   ligru = train_model(ligru, Xq, Yq)

            decoders = [
                ("GRU",   gru,   GRU_N_PCA,   GRU_K_LAG,   False),
                ("LSTM",  lstm,  LSTM_N_PCA,  LSTM_K_LAG,  False),
                ("LIN",   lin,   LINEAR_N_PCA,LINEAR_K_LAG, True ),
                ("LiGRU", ligru, LIGRU_N_PCA, LIGRU_K_LAG, False),
            ]

            # ---------- 3e) HOLD‑OUT evaluation per task -----------------------
            for dec_name, mdl, nP, lag, is_lin in decoders:

                for tst in sc["tests"]:   # iso / wm / spr / mgpt / ball …
                    df_tst = df_hold_part[df_hold_part
                                          .apply(tst["test_filter"], axis=1)].copy()
                    if df_tst.empty:
                        continue

                    Xh, Yh = build_day_decoder_data(
                                df_tst, pca, nP, lag, is_lin, ref_cols)

                    if Xh.size == 0:
                        continue

                    _, _, mVAF = evaluate_decoder(
                        mdl, Xh, Yh,
                        context=f"{dec_name}-{tst['name']}-{dstr}"
                    )

                    all_results.append({
                        "scenario_name" : sc["name"],
                        "train_monkey"  : df_day.iloc[0]["monkey"],
                        "test_monkey"   : df_day.iloc[0]["monkey"],
                        "train_task"    : "hybrid",
                        "test_task"     : tst["name"],
                        "decoder_type"  : dec_name,
                        "alignment_mode": "time‑split",
                        "mean_VAF"      : mVAF,
                        "timestamp"     : datetime.datetime.now(),
                        "train_date"    : day,   # date you fitted on
                        "test_date"     : day,   # same day, hold‑out half
                        "date"          : day,   # convenient alias
                    })
    # ---------- 4) save all results ------------------------------------------
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_pickle(SAVE_RESULTS_PATH)
        print(f"\n[INFO] saved results → {SAVE_RESULTS_PATH}")
    else:
        print("\n[WARNING] nothing to save – no results collected")



if __name__ == "__main__": 
    main()