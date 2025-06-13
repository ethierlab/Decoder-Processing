#!/usr/bin/env python3
"""
Full validation pipeline for EMG decoders.
This pipeline:
  1. Loads raw data, maps EMG channels, and groups data into scenarios.
  2. For a selected training scenario, builds a continuous dataset.
     If a 'split_ratio' is provided, it splits the dataset into a training partition (for PCA & model training) and an internal holdout.
  3. Fits a PCA model on the training partition.
  4. Trains each of the decoders (GRU, LSTM, Linear, LiGRU) on the training partition.
  5. Runs external validation over all test scenarios using different alignment modes.
  6. Saves the validation metrics into a DataFrame for later plotting.
  
A PCA caching mechanism is provided.
"""

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
from scipy.signal import butter, filtfilt
# ---------------- Global Parameters ----------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
COMBINED_PICKLE_FILE = "output.pkl"  # Update this to your file location
SAVE_RESULTS_PATH = "df_results_emg_validation_GRU.pkl"

# Data are already binned:
BIN_SIZE = 0.02       # seconds per sample
BIN_FACTOR = 1         # no downsampling
SMOOTHING_LENGTH = 0.05  # for spikes
LAG_BINS = 0

# PCA and decoder parameters
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
DECODERS_TO_RUN = {"GRU"} 
# EMG mapping
TARGET_MUSCLES = {"FCR", "FDS", "FDP", "FCU", "ECR", "EDC", "ECU"}
GLOBAL_MUSCLE_MAP = {
    'ECR_1': 'ECR', 'ECR_2': 'ECR',
    'EDC_1': 'EDC', 'EDC_2': 'EDC',
    'FCR_1': 'FCR',
    'FCU_1': 'FCU',
    'FDS_1': 'FDS', 'FDS_2': 'FDS',
    'FDP_1': 'FDP', 'FDP_2': 'FDP',
    'ECU_1': 'ECU',
}

# Options for PCA and alignment
RECALC_PCA_EACH_DAY = True
APPLY_ZSCORE = False  
REALIGN_PCA_TO_DAY0 = True

# ---------------- Preprocessing Functions ----------------

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
    X_list = []
    Y_list = []
    expected_neurons = [f"neuron{i}" for i in range(1, 97)]  # Always expect neurons 1 to 96.
    
    for idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val = row["EMG"]
        # Ensure spike_df is a DataFrame and not empty.
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        
        # Check which expected columns are missing in the original data.
        original_missing = set(expected_neurons) - set(spike_df.columns)
        if original_missing:
            print(f"[DEBUG] Padding will be applied for row {idx}: missing neurons {sorted(original_missing)}")
        
        # Reindex spike_df so it always contains all expected neuron columns, filling missing with zeros.
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

def parse_test_task(tname):
    tasks = ['iso', 'iso8', 'wm', 'spr', 'mgpt', 'ball']
    tname = tname.lower()  # ensure lowercase matching
    for t in tasks:
        if t in tname:
            return t
    return "unknown"

def build_scenarios(df):
    """
    We'll define:
     1) Within monkey: Jango, JacB, Jaco, Theo
     2) Across monkey same tasks
     3) Cross monkey CROSS tasks => iso/wm/spr vs mg-pt/ball, etc.
    """
    def f_jango_iso(r):  return (r['monkey']=='Jango') and (r['task']=='iso')
    def f_jango_wm(r):   return (r['monkey']=='Jango') and (r['task']=='wm')
    def f_jango_spr(r):  return (r['monkey']=='Jango') and (r['task']=='spr')

    def f_jacb_iso(r):   return (r['monkey']=='JacB') and (r['task']=='iso')
    def f_jacb_wm(r):    return (r['monkey']=='JacB') and (r['task']=='wm')
    def f_jacb_spr(r):   return (r['monkey']=='JacB') and (r['task']=='spr')

    def f_jaco_mgpt(r):  return (r['monkey']=='Jaco') and (r['task'] in ['mg-pt','mgpt'])
    def f_jaco_ball(r):  return (r['monkey']=='Jaco') and (r['task']=='ball')

    def f_theo_mgpt(r):  return (r['monkey']=='Theo') and (r['task'] in ['mg-pt','mgpt'])
    def f_theo_ball(r):  return (r['monkey']=='Theo') and (r['task']=='ball')

    scenarios = []

    # -------------------------------
    # WITHIN-MONKEY
    # Jango
    # scenarios.append({
    # 'name': 'Jango_iso',
    # 'train_filter': f_jango_iso,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jango_wm, 'name': 'jango_wm'},
    #     {'test_filter': f_jango_spr, 'name': 'jango_spr'}
    # ],
    # 'force_same_day': True
    # })
    # scenarios.append({
    # 'name': 'Jango_wm',
    # 'train_filter': f_jango_wm,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jango_iso, 'name': 'jango_iso'},
    #     {'test_filter': f_jango_spr, 'name': 'jango_spr'}
    # ],
    # 'force_same_day': True
    # })
    # scenarios.append({
    # 'name': 'Jango_spr',
    # 'train_filter': f_jango_spr,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jango_iso, 'name': 'jango_iso'},
    #     {'test_filter': f_jango_wm, 'name': 'jango_wm'}
    # ],
    # 'force_same_day': True
    # })

    # # JacB
    # scenarios.append({
    # 'name': 'JacB_iso',
    # 'train_filter': f_jacb_iso,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jacb_wm, 'name': 'jacB_wm'},
    #     {'test_filter': f_jacb_spr, 'name': 'jacB_spr'}
    # ],
    # 'force_same_day': True
    # })
    # scenarios.append({
    # 'name': 'JacB_wm',
    # 'train_filter': f_jacb_wm,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jacb_iso, 'name': 'jacB_iso'},
    #     {'test_filter': f_jacb_spr, 'name': 'jacB_spr'}
    # ],
    # 'force_same_day': True
    # })
    # scenarios.append({
    # 'name': 'JacB_spr',
    # 'train_filter': f_jacb_spr,
    # 'split_ratio': 0.25,
    # 'tests': [
    #     {'test_filter': f_jacb_iso, 'name': 'jacB_iso'},
    #     {'test_filter': f_jacb_wm, 'name': 'jacB_wm'}
    # ],
    # 'force_same_day': True
    # })

    # Jaco
    scenarios.append({
    'name': 'Jaco_mgpt',
    'train_filter': f_jaco_mgpt,
    'split_ratio': 0.25,
    'tests': [
        {'test_filter': f_jaco_ball, 'name': 'jaco_ball'}
    ],
    'force_same_day': True
    })
    scenarios.append({
    'name': 'Jaco_ball',
    'train_filter': f_jaco_ball,
    'split_ratio': 0.25,
    'tests': [
        {'test_filter': f_jaco_mgpt, 'name': 'jaco_mgpt'}
    ],
    'force_same_day': True
    })

    # Theo
    scenarios.append({
    'name': 'Theo_mgpt',
    'train_filter': f_theo_mgpt,
    'split_ratio': 0.25,
    'tests': [
        {'test_filter': f_theo_ball, 'name': 'theo_ball'}
    ],
    'force_same_day': True
    })
    scenarios.append({
    'name': 'Theo_ball',
    'train_filter': f_theo_ball,
    'split_ratio': 0.25,
    'tests': [
        {'test_filter': f_theo_mgpt, 'name': 'theo_mgpt'}
    ],
    'force_same_day': True
    })

    # -------------------------------
    # ACROSS-MONKEY (SAME TASK)
    # Jango <-> JacB (these monkeys likely never share the same recording day)
    # scenarios.append({
    # 'name': 'MC_JangoIso2JacBIso',
    # 'train_filter': f_jango_iso,
    # 'tests': [
    #     {'test_filter': f_jacb_iso, 'name': 'jacB_iso'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JangoWm2JacBWm',
    # 'train_filter': f_jango_wm,
    # 'tests': [
    #     {'test_filter': f_jacb_wm, 'name': 'jacB_wm'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JangoSpr2JacBSpr',
    # 'train_filter': f_jango_spr,
    # 'tests': [
    #     {'test_filter': f_jacb_spr, 'name': 'jacB_spr'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JacBIso2JangoIso',
    # 'train_filter': f_jacb_iso,
    # 'tests': [
    #     {'test_filter': f_jango_iso, 'name': 'jango_iso'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JacBWm2JangoWm',
    # 'train_filter': f_jacb_wm,
    # 'tests': [
    #     {'test_filter': f_jango_wm, 'name': 'jango_wm'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JacBSpr2JangoSpr',
    # 'train_filter': f_jacb_spr,
    # 'tests': [
    #     {'test_filter': f_jango_spr, 'name': 'jango_spr'}
    # ],
    # 'force_same_day': False
    # })

    # # Jaco <-> Theo (same tasks)
    # scenarios.append({
    # 'name': 'MC_JacoMgpt2TheoMgpt',
    # 'train_filter': f_jaco_mgpt,
    # 'tests': [
    #     {'test_filter': f_theo_mgpt, 'name': 'theo_mgpt'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_JacoBall2TheoBall',
    # 'train_filter': f_jaco_ball,
    # 'tests': [
    #     {'test_filter': f_theo_ball, 'name': 'theo_ball'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_TheoMgpt2JacoMgpt',
    # 'train_filter': f_theo_mgpt,
    # 'tests': [
    #     {'test_filter': f_jaco_mgpt, 'name': 'jaco_mgpt'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'MC_TheoBall2JacoBall',
    # 'train_filter': f_theo_ball,
    # 'tests': [
    #     {'test_filter': f_jaco_ball, 'name': 'jaco_ball'}
    # ],
    # 'force_same_day': False
    # })

    # -------------------------------
    # ACROSS-MONKEY CROSS-TASK
    # (Comparisons where recordings never align by date)
    # scenarios.append({
    # 'name': 'Cross_JangoIso_2_JacoMgpt',
    # 'train_filter': f_jango_iso,
    # 'tests': [
    #     {'test_filter': f_jaco_mgpt, 'name': 'jaco_mgpt'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'Cross_JangoSpr_2_TheoBall',
    # 'train_filter': f_jango_spr,
    # 'tests': [
    #     {'test_filter': f_theo_ball, 'name': 'theo_ball'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'Cross_JacBWm_2_JacoBall',
    # 'train_filter': f_jacb_wm,
    # 'tests': [
    #     {'test_filter': f_jaco_ball, 'name': 'jaco_ball'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'Cross_JacBIso_2_TheoMgpt',
    # 'train_filter': f_jacb_iso,
    # 'tests': [
    #     {'test_filter': f_theo_mgpt, 'name': 'theo_mgpt'}
    # ],
    # 'force_same_day': False
    # })
    # scenarios.append({
    # 'name': 'Cross_JacoMgpt_2_JangoSpr',
    # 'train_filter': f_jaco_mgpt,
    # 'tests': [
    #     {'test_filter': f_jango_spr, 'name': 'jango_spr'}
    # ],
    # 'force_same_day': False
    # })


    return scenarios

# ---------------- Sequence Building Functions ----------------

def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len + LAG_BINS:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
    X_out, Y_out = [], []
    T = X_arr.shape[0]
    for t in range(seq_len + LAG_BINS, T):
        X_out.append(X_arr[t-seq_len-LAG_BINS : t-LAG_BINS])
        Y_out.append(Y_arr[t, :])      # ← keeps the channel dimension
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

def build_dayX_decoder_data(df, day_pca_model, n_pca, seq_len, is_linear=False, reference_emg_cols=None):
    X_big, Y_big = build_continuous_dataset_preprocessed(df, reference_emg_cols=reference_emg_cols)
    if X_big.shape[0] == 0:
        return np.empty((0, )), np.empty((0, ))
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

# ---------------- Validation Functions ----------------
def append_holdout_results(decoder_name, model, X_hold, Y_hold, scenario_name, df_train_day, sc):
    preds_hold, vaf_ch_hold, mean_vaf_hold = evaluate_decoder(model, X_hold, Y_hold)
    date_val = df_train_day.iloc[0]["date"] if "date" in df_train_day.columns else None
    train_task = sc.get("task", "Unknown")
    # List the external alignment modes for which you want a diagonal entry.
    align_modes = ["bland", "recalculated", "realign"]
    results = []
    for mode in align_modes:
        result = {
            "scenario_name": scenario_name,
            "train_monkey": sc.get("monkey", "Unknown"),
            "test_monkey": sc.get("monkey", "Unknown"),
            "train_task": train_task,
            "test_task": train_task,
            # Use the training task as the test name for the diagonal.
            "test_name": train_task,
            "decoder_type": decoder_name,
            "alignment_mode": mode,
            "mean_VAF": mean_vaf_hold,
            "VAF_ch": vaf_ch_hold,
            "timestamp": datetime.datetime.now(),
            "train_date": date_val,
            "test_date": date_val,
            "holdout_type": "internal",
            "date": date_val
        }
        results.append(result)
    return results

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
    print(f"[DEBUG]{context} - Evaluating decoder {model.__class__.__name__}: X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
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

def validate_scenario(train_scenario, test_scenario, model, n_pca, seq_len, is_linear=False,
                      alignment_mode="bland", reference_emg_cols=None, force_external=False):
    # If not forcing external, and the test DataFrame equals the training DataFrame,
    # run the internal holdout branch.
    if not force_external and train_scenario["df"].equals(test_scenario["df"]):
        split_ratio = train_scenario.get('split_ratio', None)
        if split_ratio is not None and 0 < split_ratio < 1:
            X_train_raw, Y_train_raw = build_continuous_dataset_preprocessed(train_scenario["df"],
                                                                             reference_emg_cols=reference_emg_cols)
            if X_train_raw.shape[0] == 0:
                print("Training data empty for internal holdout validation.")
                return None, None, None
            z_train = train_scenario["pca_model"].transform(X_train_raw)
            X_train_proj = z_train[:, :n_pca]
            if not is_linear:
                X_seq, Y_seq = create_rnn_dataset_continuous(X_train_proj, Y_train_raw, seq_len)
            else:
                X_seq, Y_seq = create_linear_dataset_continuous(X_train_proj, Y_train_raw, seq_len)
            X_tr, X_val, Y_tr, Y_val = train_test_split(X_seq, Y_seq, test_size=split_ratio, random_state=SEED, shuffle=False)
            preds, vaf_ch, mean_vaf = evaluate_decoder(model, X_val, Y_val)
            print("Internal Holdout Validation:")
            print(f"Mean VAF: {mean_vaf:.4f}")
            return preds, vaf_ch, mean_vaf

    # --- External Validation Branch ---
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
            print("Failed to compute test PCA; defaulting to bland.")
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
            print("Failed to compute test PCA; defaulting to bland.")
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
        print(f"Unknown alignment_mode: {alignment_mode}; defaulting to bland.")
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

# Example of how to loop through scenarios and decoders:
# def run_validation(scenarios, trained_models, reference_emg_cols):
#     results = []
#     for scenario in scenarios:
#         train_df = scenario["df"]
#         if "pca_model" not in scenario:
#             X_train_tmp, _ = build_continuous_dataset_preprocessed(train_df, reference_emg_cols=reference_emg_cols)
#             if X_train_tmp.shape[0] > 0:
#                 max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
#                 pca_model = PCA(n_components=max_dim, random_state=SEED)
#                 pca_model.fit(X_train_tmp)
#                 scenario["pca_model"] = pca_model
#             else:
#                 continue
#         if "tests" in scenario:
#             for test_def in scenario["tests"]:
#                 test_scenario = {
#                     "scenario_name": f"{scenario['monkey']}_{test_def['name']}",
#                     "monkey": scenario["monkey"],
#                     "task": test_def["name"],
#                     "df": scenario["df"]  # Modify if you want a different test df.
#                 }
#                 for decoder_name, model in trained_models.items():
#                     if decoder_name == "GRU":
#                         n_pca_val = GRU_N_PCA
#                         seq_len = GRU_K_LAG
#                     elif decoder_name == "LSTM":
#                         n_pca_val = LSTM_N_PCA
#                         seq_len = LSTM_K_LAG
#                     elif decoder_name == "Linear":
#                         n_pca_val = LINEAR_N_PCA
#                         seq_len = LINEAR_K_LAG
#                     elif decoder_name == "LiGRU":
#                         n_pca_val = LIGRU_N_PCA
#                         seq_len = LIGRU_K_LAG
#                     else:
#                         continue
#                     for align_mode in ["bland", "recalculated", "realign"]:
#                         preds, vaf_ch, mean_vaf = validate_scenario(
#                             train_scenario=scenario,
#                             test_scenario=test_scenario,
#                             model=model,
#                             n_pca=n_pca_val,
#                             seq_len=seq_len,
#                             is_linear=(decoder_name=="Linear"),
#                             alignment_mode=align_mode,
#                             reference_emg_cols=reference_emg_cols
#                         )
#                         result = {
#                             "scenario_name": scenario["scenario_name"],
#                             "train_monkey": scenario["monkey"],
#                             "train_task": scenario["task"],
#                             "test_name": test_def["name"],
#                             "decoder_type": decoder_name,
#                             "alignment_mode": align_mode,
#                             "mean_VAF": mean_vaf,
#                             "VAF_ch": vaf_ch,
#                             "timestamp": datetime.datetime.now(),
#                             "date": train_df.iloc[0]["date"] if "date" in train_df.columns else None
#                         }
#                         results.append(result)
#                         print(f"[RESULT] {result}")
#     df_results = pd.DataFrame(results)
#     df_results.to_pickle(SAVE_RESULTS_PATH)
#     return df_results


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
            print(f"Epoch {ep}/{num_epochs}: Loss = {total_loss/len(loader):.4f}")
    return model

# ---------------- Main Pipeline ----------------

def main():
    set_seed(SEED)
    print(f"[INFO] Using device: {DEVICE}")
    
    # 1. Load raw data and preprocess EMG.
    df_raw = pd.read_pickle(COMBINED_PICKLE_FILE)
    try:
        df_raw["date"] = pd.to_datetime(df_raw["date"])
    except Exception as e:
        print("Error converting date column:", e)
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_proc, train_emg_cols = filter_and_map_emg(df_raw)
    print(f"[INFO] Processed df shape: {df_proc.shape}")
    all_results = []
    training_times = {}
    
    # 2. Build scenarios.
    scenarios = build_scenarios(df_proc)
    print(f"[INFO] Found {len(scenarios)} scenarios.")
    
    # 3. Process each scenario.
    for sc in scenarios:
        scenario_name = sc.get("name", "UnnamedScenario")
        print(f"\n[INFO] Processing scenario: {scenario_name}")
        
        # Set default metadata if not present.
        if "monkey" not in sc:
            sc["monkey"] = scenario_name.split("_")[0]
        if "task" not in sc:
            sc["task"] = scenario_name.split("_")[-1]
        
        # Apply the train_filter to obtain training data.
        df_train = df_proc[df_proc.apply(sc['train_filter'], axis=1)].copy()
        if df_train.empty:
            print(f"[WARNING] Scenario {scenario_name} has no training data; skipping.")
            continue
        
        # Always split training data by day.
        if "date" in df_train.columns:
            unique_train_days = sorted(df_train["date"].unique())
        else:
            unique_train_days = [None]
        
        if scenario_name not in training_times:
            training_times[scenario_name] = {}
        
        # Loop over each training day.
        for train_day in unique_train_days:
            if train_day is not None:
                df_train_day = df_train[df_train["date"] == train_day].copy()
                day_train_str = train_day.strftime("%Y%m%d") if hasattr(train_day, "strftime") else str(train_day)
            else:
                df_train_day = df_train.copy()
                day_train_str = "all"
            
            if df_train_day.empty:
                print(f"[WARNING] Scenario {scenario_name} - Train Day {day_train_str}: no training data; skipping.")
                continue
            
            print(f"[INFO] Scenario {scenario_name} - Train Day {day_train_str}: {df_train_day.shape[0]} samples")
            
            # Update the scenario with this day-specific training DataFrame.
            sc["df"] = df_train_day
            
            # Build continuous training dataset.
            X_train_raw, Y_train_raw = build_continuous_dataset_preprocessed(df_train_day, reference_emg_cols=train_emg_cols)
            if X_train_raw.shape[0] == 0:
                print(f"[WARNING] Scenario {scenario_name} - Train Day {day_train_str}: no raw training data; skipping.")
                continue
            
            # Fit PCA on training data.
            max_dim = max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
            pca_train = PCA(n_components=max_dim, random_state=SEED)
            pca_train.fit(X_train_raw)
            sc["pca_model"] = pca_train
            print(f"[DEBUG] Scenario {scenario_name} - Train Day {day_train_str}: PCA components shape: {pca_train.components_.shape}")
            # Build training sequences.
            X_gru_train, Y_gru_train = build_dayX_decoder_data(df_train_day, pca_train, GRU_N_PCA, GRU_K_LAG, is_linear=False, reference_emg_cols=train_emg_cols)
            X_lstm_train, Y_lstm_train = build_dayX_decoder_data(df_train_day, pca_train, LSTM_N_PCA, LSTM_K_LAG, is_linear=False, reference_emg_cols=train_emg_cols)
            X_lin_train, Y_lin_train   = build_dayX_decoder_data(df_train_day, pca_train, LINEAR_N_PCA, LINEAR_K_LAG, is_linear=True,  reference_emg_cols=train_emg_cols)
            X_ligru_train, Y_ligru_train = build_dayX_decoder_data(df_train_day, pca_train, LIGRU_N_PCA, LIGRU_K_LAG, is_linear=False, reference_emg_cols=train_emg_cols)
            
            print("============= INPUT SHAPE =============")
            print(f" GRU    : X: {X_gru_train.shape} , Y: {Y_gru_train.shape}")
            print(f" LSTM   : X: {X_lstm_train.shape} , Y: {Y_lstm_train.shape}")
            print(f" Linear : X: {X_lin_train.shape} , Y: {Y_lin_train.shape}")
            print(f" LiGRU  : X: {X_ligru_train.shape} , Y: {Y_ligru_train.shape}")
            n_emg_channels = Y_train_raw.shape[1]
            
            # --- Internal Holdout Split ---
            split_ratio = sc.get("split_ratio", None)
            if split_ratio is not None and 0 < split_ratio < 1:
                print(f"[INFO] Scenario {scenario_name} - Train Day {day_train_str}: Applying internal holdout split (ratio={split_ratio}).")
                X_gru_tr, X_gru_hold, Y_gru_tr, Y_gru_hold = train_test_split(X_gru_train, Y_gru_train, test_size=split_ratio, random_state=SEED, shuffle=False)
                X_lstm_tr, X_lstm_hold, Y_lstm_tr, Y_lstm_hold = train_test_split(X_lstm_train, Y_lstm_train, test_size=split_ratio, random_state=SEED, shuffle=False)
                X_lin_tr, X_lin_hold, Y_lin_tr, Y_lin_hold = train_test_split(X_lin_train, Y_lin_train, test_size=split_ratio, random_state=SEED, shuffle=False)
                X_ligru_tr, X_ligru_hold, Y_ligru_tr, Y_ligru_hold = train_test_split(X_ligru_train, Y_ligru_train, test_size=split_ratio, random_state=SEED, shuffle=False)
            else:
                X_gru_tr, X_gru_hold, Y_gru_tr, Y_gru_hold = X_gru_train, None, Y_gru_train, None
                X_lstm_tr, X_lstm_hold, Y_lstm_tr, Y_lstm_hold = X_lstm_train, None, Y_lstm_train, None
                X_lin_tr, X_lin_hold, Y_lin_tr, Y_lin_hold = X_lin_train, None, Y_lin_train, None
                X_ligru_tr, X_ligru_hold, Y_ligru_tr, Y_ligru_hold = X_ligru_train, None, Y_ligru_train, None
            
            # --- Initialize Models ---
            trained_models = {}

            if "GRU" in DECODERS_TO_RUN:
                gru_model = GRUDecoder(GRU_N_PCA, GRU_HIDDEN_DIM, n_emg_channels).to(DEVICE)
                trained_models["GRU"] = gru_model

            if "LSTM" in DECODERS_TO_RUN:
                lstm_model = LSTMDecoder(LSTM_N_PCA, LSTM_HIDDEN_DIM, n_emg_channels).to(DEVICE)
                trained_models["LSTM"] = lstm_model

            if "Linear" in DECODERS_TO_RUN:
                linear_model = LinearLagDecoder(LINEAR_N_PCA * LINEAR_K_LAG,
                                                LINEAR_HIDDEN_DIM, n_emg_channels).to(DEVICE)
                trained_models["Linear"] = linear_model

            if "LiGRU" in DECODERS_TO_RUN:
                ligru_model = LiGRUDecoder(LIGRU_N_PCA, LIGRU_HIDDEN_DIM, n_emg_channels).to(DEVICE)
                trained_models["LiGRU"] = ligru_model
            
            # --- Train Each Model on the Training Partition ---
            print(f"[INFO] Scenario {scenario_name} - Train Day {day_train_str}: Training Models...")
            if "GRU" in DECODERS_TO_RUN:
                print("Training GRU...")
                start_time = time.time()
                gru_model = train_model(gru_model, X_gru_tr, Y_gru_tr,
                                        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
                elapse = time.time() - start_time
                print(f"GRU training completed in {elapse} s.")

            if "LSTM" in DECODERS_TO_RUN:
                print("Training LSTM...")
                start_time = time.time()
                lstm_model = train_model(lstm_model, X_lstm_tr, Y_lstm_tr,
                                        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
                elapse = time.time() - start_time
                print(f"LSTM training completed in {elapse} s.")

            if "Linear" in DECODERS_TO_RUN:
                print("Training Linear...")
                start_time = time.time()
                linear_model = train_model(linear_model, X_lin_tr, Y_lin_tr,
                                        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
                elapse = time.time() - start_time
                print(f"Linear training completed in {elapse} s.")

            if "LiGRU" in DECODERS_TO_RUN:
                print("Training LiGRU...")
                start_time = time.time()
                ligru_model = train_model(ligru_model, X_ligru_tr, Y_ligru_tr,
                                        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
                elapse = time.time() - start_time
                print(f"LiGRU training completed in {elapse} s.")

            has_holdout = (
                ("GRU"    in DECODERS_TO_RUN and X_gru_hold    is not None) or
                ("LSTM"   in DECODERS_TO_RUN and X_lstm_hold   is not None) or
                ("Linear" in DECODERS_TO_RUN and X_lin_hold    is not None) or
                ("LiGRU"  in DECODERS_TO_RUN and X_ligru_hold  is not None)
            )
            # --- Internal Holdout Evaluation ---
            if has_holdout:
                if "GRU" in DECODERS_TO_RUN:
                    all_results.extend(append_holdout_results("GRU", gru_model,
                                                            X_gru_hold, Y_gru_hold,
                                                            scenario_name, df_train_day, sc))
                if "LSTM" in DECODERS_TO_RUN:
                    all_results.extend(append_holdout_results("LSTM", lstm_model,
                                                            X_lstm_hold, Y_lstm_hold,
                                                            scenario_name, df_train_day, sc))
                if "Linear" in DECODERS_TO_RUN:
                    all_results.extend(append_holdout_results("Linear", linear_model,
                                                            X_lin_hold, Y_lin_hold,
                                                            scenario_name, df_train_day, sc))
                if "LiGRU" in DECODERS_TO_RUN:
                    all_results.extend(append_holdout_results("LiGRU", ligru_model,
                                                            X_ligru_hold, Y_ligru_hold,
                                                            scenario_name, df_train_day, sc))
            
            # --- External Validation ---
            if sc.get("force_same_day", True):
                rec_date = df_train_day.iloc[0]["date"]
                monkey_val = sc.get("monkey", "Unknown")
                # Use global data for this day & monkey.
                df_test_day = df_proc[(df_proc["date"] == rec_date) & (df_proc["monkey"] == monkey_val)].copy()
                day_test_str = rec_date.strftime("%Y%m%d") if hasattr(rec_date, "strftime") else str(rec_date)
                
                # print(f"[DEBUG] Global data for date {day_test_str} (monkey {monkey_val}):")
                if "task" in df_test_day.columns:
                    print(df_test_day["task"].value_counts())
                
                test_filters = sc.get("tests", [])
                for test_def in test_filters:
                    test_name = test_def["name"]
                    train_task = sc.get("task", "Unknown")
                    parsed_test_task = parse_test_task(test_name)
                    
                    if train_task.lower() != parsed_test_task.lower():
                        df_test_sub = df_test_day[df_test_day.apply(test_def["test_filter"], axis=1)].copy()
                        print(f"[DEBUG] Filtered test data for off-diagonal (test filter: {test_def['name']}): {df_test_sub.shape[0]} samples")
                        if df_test_sub.empty:
                            print(f"[WARNING] Scenario {scenario_name} - Test Filter {test_name}: no data after filtering; skipping external evaluation.")
                            continue
                    else:
                        print(f"[DEBUG] Diagonal entry detected for scenario '{scenario_name}' (train_task: '{train_task}' == parsed test task: '{parsed_test_task}').")
                        print("        Skipping external evaluation; using holdout result instead.")
                        continue
                    
                    test_scenario = {"df": df_test_sub, "monkey": monkey_val, "task": parsed_test_task}
                    
                    print(f"[DEBUG] Processing test_def='{test_def}' for scenario '{scenario_name}' on day {day_test_str}:")
                    print(f"        train_task = {train_task}, test_name = {test_name}, parsed_test_task = {parsed_test_task}")
                    
                    for decoder_name, model in trained_models.items():
                        if decoder_name == "GRU":
                            n_pca_val = GRU_N_PCA
                            seq_len_val = GRU_K_LAG
                            is_linear_val = False
                        elif decoder_name == "LSTM":
                            n_pca_val = LSTM_N_PCA
                            seq_len_val = LSTM_K_LAG
                            is_linear_val = False
                        elif decoder_name == "Linear":
                            n_pca_val = LINEAR_N_PCA
                            seq_len_val = LINEAR_K_LAG
                            is_linear_val = True
                        elif decoder_name == "LiGRU":
                            n_pca_val = LIGRU_N_PCA
                            seq_len_val = LIGRU_K_LAG
                            is_linear_val = False
                        else:
                            continue
                        for align_mode in ["bland", "recalculated", "realign"]:
                            preds, vaf_ch, mean_vaf = validate_scenario(
                                train_scenario=sc,
                                test_scenario=test_scenario,
                                model=model,
                                n_pca=n_pca_val,
                                seq_len=seq_len_val,
                                is_linear=is_linear_val,
                                alignment_mode=align_mode,
                                reference_emg_cols=train_emg_cols,
                                force_external=True
                            )
                            print(f"[DEBUG] External validation result for scenario '{scenario_name}':")
                            print(f"        Decoder {decoder_name}, Alignment {align_mode} -> mean_VAF = {mean_vaf:.4f}")
                            # If diagonal, skip (external branch should only produce off-diagonals)
                            if train_task.lower() == parsed_test_task.lower():
                                print(f"[DEBUG] Diagonal entry detected (train_task: '{train_task}' == parsed test task: '{parsed_test_task}').")
                                print("        Skipping external result since holdout covers the diagonal.")
                                continue
                            result = {
                                "scenario_name": sc.get("name", "UnnamedScenario"),
                                "train_monkey": monkey_val,
                                "test_monkey": monkey_val,
                                "train_task": train_task,
                                "test_name": test_name,
                                "test_task": parsed_test_task,
                                "decoder_type": decoder_name,
                                "alignment_mode": align_mode,
                                "mean_VAF": mean_vaf,
                                "VAF_ch": vaf_ch,
                                "timestamp": datetime.datetime.now(),
                                "train_date": df_train_day.iloc[0]["date"] if "date" in df_train_day.columns else None,
                                "test_date": df_test_sub.iloc[0]["date"] if "date" in df_test_sub.columns else None,
                                "date": df_train_day.iloc[0]["date"] if "date" in df_train_day.columns else None
                            }
                            all_results.append(result)


            else:
                # force_same_day is False: reapply test filters on the full dataset.
                for test_def in sc.get("tests", []):
                    test_name = test_def["name"]
                    test_task_val = parse_test_task(test_name)
                    df_test_all = df_proc[df_proc.apply(test_def["test_filter"], axis=1)].copy()
                    if df_test_all.empty:
                        print(f"[WARNING] Scenario {scenario_name} - Test Filter {test_name}: no test data; skipping.")
                        continue
                    if "date" in df_test_all.columns:
                        unique_test_days = sorted(df_test_all["date"].unique())
                    else:
                        unique_test_days = [None]
                    for test_day in unique_test_days:
                        if test_day is not None:
                            df_test_day = df_test_all[df_test_all["date"] == test_day].copy()
                            day_test_str = test_day.strftime("%Y%m%d") if hasattr(test_day, "strftime") else str(test_day)
                        else:
                            df_test_day = df_test_all
                            day_test_str = "all"
                        if df_test_day.empty:
                            print(f"[WARNING] Scenario {scenario_name} - Test Filter {test_name} - Test Day {day_test_str}: no test data; skipping.")
                            continue
                        test_scenario = {"df": df_test_day, "monkey": sc.get("monkey", None), "task": test_task_val}
                        for decoder_name, model in trained_models.items():
                            if decoder_name == "GRU":
                                n_pca_val = GRU_N_PCA
                                seq_len_val = GRU_K_LAG
                                is_linear_val = False
                            elif decoder_name == "LSTM":
                                n_pca_val = LSTM_N_PCA
                                seq_len_val = LSTM_K_LAG
                                is_linear_val = False
                            elif decoder_name == "Linear":
                                n_pca_val = LINEAR_N_PCA
                                seq_len_val = LINEAR_K_LAG
                                is_linear_val = True
                            elif decoder_name == "LiGRU":
                                n_pca_val = LIGRU_N_PCA
                                seq_len_val = LIGRU_K_LAG
                                is_linear_val = False
                            else:
                                continue
                            for align_mode in ["bland", "recalculated", "realign"]:
                                preds, vaf_ch, mean_vaf = validate_scenario(
                                    train_scenario=sc,
                                    test_scenario=test_scenario,
                                    model=model,
                                    n_pca=n_pca_val,
                                    seq_len=seq_len_val,
                                    is_linear=is_linear_val,
                                    alignment_mode=align_mode,
                                    reference_emg_cols=train_emg_cols
                                )
                                result = {
                                    "scenario_name": sc.get("name", "UnnamedScenario"),
                                    "train_monkey": sc.get("monkey", "Unknown"),
                                    "test_monkey": test_scenario.get("monkey", "Unknown"),
                                    "train_task": sc.get("task", "Unknown"),
                                    "test_name": test_name,
                                    "test_task": test_task_val,
                                    "decoder_type": decoder_name,
                                    "alignment_mode": align_mode,
                                    "mean_VAF": mean_vaf,
                                    "VAF_ch": vaf_ch,
                                    "timestamp": datetime.datetime.now(),
                                    "train_date": df_train_day.iloc[0]["date"] if "date" in df_train_day.columns else None,
                                    "test_date": df_test_day.iloc[0]["date"] if "date" in df_test_day.columns else None,
                                    "date": df_train_day.iloc[0]["date"] if "date" in df_train_day.columns else None
                                }
                                all_results.append(result)
    
    if all_results:
        df_final_results = pd.DataFrame(all_results)
        print("DEBUG: Validation Results Summary:")
        print(df_final_results.head())
        df_final_results.to_pickle(SAVE_RESULTS_PATH)
        print(f"[INFO] Validation complete. Results saved to {SAVE_RESULTS_PATH}")
    else:
        print("[WARNING] No validation results were generated.")

if __name__ == "__main__":
    main()