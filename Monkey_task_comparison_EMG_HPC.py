#!/usr/bin/env python3
import os, sys, random, datetime, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import time

from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import iirnotch, filtfilt, butter
from sklearn.model_selection import train_test_split

# ---------- Default Hyperparameters ----------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_PICKLE_FILE = "output.pkl"
BIN_SIZE = 0.02
SMOOTHING_LENGTH = 0.05
LAG_BINS = 0
GRU_N_PCA = 14
LSTM_N_PCA = 14
LINEAR_N_PCA = 18
LIGRU_N_PCA = 14
GRU_HIDDEN_DIM = 5
GRU_K_LAG = 16
LSTM_HIDDEN_DIM = 16
LSTM_K_LAG = 16
LINEAR_HIDDEN_DIM = 64
LINEAR_K_LAG = 16
LIGRU_HIDDEN_DIM = 5
LIGRU_K_LAG = 16
NUM_EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001

TARGET_MUSCLES = {"FCR", "FDS", "FDP", "FCU", "ECR", "EDC", "ECU"}
GLOBAL_MUSCLE_MAP = {
    'ECR_1': 'ECR', 'ECR_2': 'ECR',
    'EDC_1': 'EDC', 'EDC_2': 'EDC',
    'FCR_1': 'FCR', 'FCU_1': 'FCU',
    'FDS_1': 'FDS', 'FDS_2': 'FDS',
    'FDP_1': 'FDP', 'FDP_2': 'FDP',
    'ECU_1': 'ECU',
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, help='Scenario name to run')
    parser.add_argument('--train_day_idx', type=int, required=True, help='Index of the training day in scenario')
    parser.add_argument('--decoder', type=str, required=True, choices=['GRU','LSTM','Linear','LiGRU'])
    parser.add_argument('--output_dir', type=str, default="results_emg", help="Directory to save result files")
    parser.add_argument('--combined_pickle', type=str, default=COMBINED_PICKLE_FILE, help="Pickle file for all data")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smooth_spike_data(x_2d, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    sigma = (smoothing_length / bin_size) / 2
    out = np.zeros_like(x_2d, dtype=float)
    for ch in range(x_2d.shape[1]):
        out[:, ch] = gaussian_filter1d(x_2d[:, ch].astype(float), sigma=sigma)
    return out

def smooth_emg(emg_array, fs=1000):
    rect = np.abs(emg_array)
    b_notch, a_notch = iirnotch(60, 30, fs)
    rect = filtfilt(b_notch, a_notch, rect, axis=0)
    b_lp, a_lp = butter(4, 20/(fs/2), 'low')
    return filtfilt(b_lp, a_lp, rect, axis=0)

def map_emg_labels(emg_df):
    new_data = {}
    count_for_muscle = defaultdict(int)
    for col in emg_df.columns:
        raw = col.strip().upper()
        tmp = raw
        while len(tmp) > 0 and (tmp[-1].isdigit() or tmp[-1].isalpha()):
            if tmp in GLOBAL_MUSCLE_MAP:
                break
            tmp = tmp[:-1]
        if tmp == "":
            tmp = raw
        base = GLOBAL_MUSCLE_MAP.get(tmp, None)
        if base is None or base not in TARGET_MUSCLES:
            continue
        count_for_muscle[base] += 1
        new_label = f"{base}_{count_for_muscle[base]}"
        new_data[new_label] = emg_df[col]
    return pd.DataFrame(new_data)

def filter_and_map_emg(df):
    new_rows = []
    all_cols = set()
    for _, row in df.iterrows():
        emg_df = row.get("EMG")
        if isinstance(emg_df, pd.DataFrame) and not emg_df.empty:
            mapped = map_emg_labels(emg_df)
            row["EMG"] = mapped
            all_cols.update(mapped.columns)
        new_rows.append(row)
    df_new = pd.DataFrame(new_rows)
    sorted_cols = sorted(list(all_cols))
    for idx, row in df_new.iterrows():
        emg_df = row.get("EMG")
        if isinstance(emg_df, pd.DataFrame):
            row["EMG"] = emg_df.reindex(columns=sorted_cols, fill_value=0)
    return df_new, sorted_cols

def build_continuous_dataset_preprocessed(df, reference_emg_cols=None):
    X_list, Y_list = [], []
    expected_neurons = [f"neuron{i}" for i in range(1, 97)]
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        spike_df = spike_df.reindex(columns=expected_neurons, fill_value=0)
        if emg_val is None:
            continue
        X_smoothed = smooth_spike_data(spike_df.values, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH)
        if isinstance(emg_val, pd.DataFrame):
            emg_df = emg_val
            if reference_emg_cols is not None:
                emg_df = emg_df.reindex(reference_emg_cols, axis=1, fill_value=0)
            emg_array = emg_df.values
        else:
            emg_array = np.asarray(emg_val)
        Y_smoothed = smooth_emg(emg_array, fs=1000)
        X_list.append(X_smoothed)
        Y_list.append(emg_array)
    if len(X_list) == 0:
        return np.empty((0,)), np.empty((0,))
    X_big = np.concatenate(X_list, axis=0)
    Y_big = np.concatenate(Y_list, axis=0)
    return X_big, Y_big

def build_scenarios(df):
    def f_jaco_mgpt(r):  return (r['monkey']=='Jaco') and (r['task'] in ['mg-pt','mgpt'])
    def f_jaco_ball(r):  return (r['monkey']=='Jaco') and (r['task']=='ball')
    def f_theo_mgpt(r):  return (r['monkey']=='Theo') and (r['task'] in ['mg-pt','mgpt'])
    def f_theo_ball(r):  return (r['monkey']=='Theo') and (r['task']=='ball')
    scenarios = []
    scenarios.append({
        'name': 'Jaco_mgpt',
        'train_filter': f_jaco_mgpt,
        'split_ratio': 0.25,
        'tests': [{'test_filter': f_jaco_ball, 'name': 'jaco_ball'}],
        'force_same_day': True
    })
    scenarios.append({
        'name': 'Jaco_ball',
        'train_filter': f_jaco_ball,
        'split_ratio': 0.25,
        'tests': [{'test_filter': f_jaco_mgpt, 'name': 'jaco_mgpt'}],
        'force_same_day': True
    })
    scenarios.append({
        'name': 'Theo_mgpt',
        'train_filter': f_theo_mgpt,
        'split_ratio': 0.25,
        'tests': [{'test_filter': f_theo_ball, 'name': 'theo_ball'}],
        'force_same_day': True
    })
    scenarios.append({
        'name': 'Theo_ball',
        'train_filter': f_theo_ball,
        'split_ratio': 0.25,
        'tests': [{'test_filter': f_theo_mgpt, 'name': 'theo_mgpt'}],
        'force_same_day': True
    })
    return scenarios

def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len + LAG_BINS:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T = X_arr.shape[0]
    for t in range(seq_len + LAG_BINS, T):
        X_out.append(X_arr[t-seq_len-LAG_BINS : t-LAG_BINS])
        Y_out.append(Y_arr[t, :])
    return np.asarray(X_out, np.float32), np.asarray(Y_out, np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len + LAG_BINS:
        return np.empty((0, seq_len*X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T = X_arr.shape[0]
    for t in range(seq_len + LAG_BINS, T):
        window = X_arr[t-seq_len-LAG_BINS : t-LAG_BINS].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[t, :])
    return np.asarray(X_out, np.float32), np.asarray(Y_out, np.float32)

# ---------------- Model Classes ----------------
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
            print(f"Epoch {ep}/{num_epochs}: Loss = {total_loss/len(loader):.4f}")
    return model

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
        preds = np.empty((0, ))
    vaf_ch = compute_multichannel_vaf(Y_val, preds)
    mean_vaf = np.nanmean(vaf_ch)
    return preds, vaf_ch, mean_vaf

def main():
    args = parse_args()
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    # --- Data loading and scenario selection ---
    df_raw = pd.read_pickle(args.combined_pickle)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_proc, train_emg_cols = filter_and_map_emg(df_raw)
    scenarios = build_scenarios(df_proc)
    scenario_dict = {sc["name"]: sc for sc in scenarios}
    sc = scenario_dict[args.scenario]
    scenario_name = sc["name"]
    df_train = df_proc[df_proc.apply(sc['train_filter'], axis=1)].copy()
    unique_train_days = sorted(df_train["date"].unique())
    if args.train_day_idx >= len(unique_train_days):
        raise IndexError("train_day_idx is out of range for scenario %s" % scenario_name)
    train_day = unique_train_days[args.train_day_idx]
    df_train_day = df_train[df_train["date"] == train_day].copy()
    sc["df"] = df_train_day
    print(f"[INFO] Scenario: {scenario_name} | Day: {train_day} | Decoder: {args.decoder} | Samples: {df_train_day.shape[0]}")
    # --- Model selection ---
    decoder_name = args.decoder
    if decoder_name == "GRU":
        N_PCA, HID, LAG, IS_LIN = GRU_N_PCA, GRU_HIDDEN_DIM, GRU_K_LAG, False
    elif decoder_name == "LSTM":
        N_PCA, HID, LAG, IS_LIN = LSTM_N_PCA, LSTM_HIDDEN_DIM, LSTM_K_LAG, False
    elif decoder_name == "Linear":
        N_PCA, HID, LAG, IS_LIN = LINEAR_N_PCA, LINEAR_HIDDEN_DIM, LINEAR_K_LAG, True
    elif decoder_name == "LiGRU":
        N_PCA, HID, LAG, IS_LIN = LIGRU_N_PCA, LIGRU_HIDDEN_DIM, LIGRU_K_LAG, False
    else:
        raise ValueError("Unknown decoder")
    # --- Data preprocessing ---
    X_train_raw, Y_train_raw = build_continuous_dataset_preprocessed(df_train_day, reference_emg_cols=train_emg_cols)
    if X_train_raw.shape[0] == 0:
        print(f"[WARNING] No training data for scenario {scenario_name} on day {train_day}")
        sys.exit(0)
    max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
    pca_train = PCA(n_components=max_dim, random_state=SEED)
    pca_train.fit(X_train_raw)
    if not IS_LIN:
        X_train, Y_train = create_rnn_dataset_continuous(pca_train.transform(X_train_raw)[:, :N_PCA], Y_train_raw, LAG)
    else:
        X_train, Y_train = create_linear_dataset_continuous(pca_train.transform(X_train_raw)[:, :N_PCA], Y_train_raw, LAG)
    n_emg_channels = Y_train.shape[1]
    # --- Model construction ---
    if decoder_name == "GRU":
        model = GRUDecoder(N_PCA, HID, n_emg_channels).to(DEVICE)
    elif decoder_name == "LSTM":
        model = LSTMDecoder(N_PCA, HID, n_emg_channels).to(DEVICE)
    elif decoder_name == "Linear":
        model = LinearLagDecoder(N_PCA*LAG, HID, n_emg_channels).to(DEVICE)
    elif decoder_name == "LiGRU":
        model = LiGRUDecoder(N_PCA, HID, n_emg_channels).to(DEVICE)
    # --- Train model ---
    print(f"[INFO] Training {decoder_name}...")
    model = train_model(model, X_train, Y_train, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    # --- Holdout evaluation (optionnel : tu peux ajouter external validation ici) ---
    preds, vaf_ch, mean_vaf = evaluate_decoder(model, X_train, Y_train, context="train")
    result = {
        "scenario_name": scenario_name,
        "train_day": train_day,
        "decoder_type": decoder_name,
        "mean_VAF": mean_vaf,
        "VAF_channels": vaf_ch,
        "timestamp": datetime.datetime.now(),
        "train_date": train_day
    }
    # --- Save results ---
    out_path = os.path.join(
        args.output_dir,
        f"results_{scenario_name}_day{args.train_day_idx}_{decoder_name}.pkl"
    )
    pd.DataFrame([result]).to_pickle(out_path)
    print(f"[INFO] Result saved to {out_path}")

if __name__ == "__main__":
    main()
