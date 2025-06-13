#!/usr/bin/env python3

import os
import sys
import random
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
###############################################################################
# GLOBAL SETTINGS
###############################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_VERBOSE = True
def debug_print(*args):
    if GLOBAL_VERBOSE:
        print(*args)

# Example binning/smoothing for spikes
BIN_FACTOR = 1
BIN_SIZE   = 0.02
SMOOTHING_LENGTH = 0.05

GRU_N_PCA    = 14
GRU_HIDDEN_DIM= 12
GRU_K_LAG     = 16

LSTM_N_PCA   = 14
LSTM_HIDDEN_DIM=16
LSTM_K_LAG   = 16

LINEAR_N_PCA = 18
LINEAR_HIDDEN_DIM=64
LINEAR_K_LAG = 16

LIGRU_N_PCA  = 14
LIGRU_HIDDEN_DIM=16
LIGRU_K_LAG  = 16

NUM_EPOCHS   = 100
BATCH_SIZE   = 64
LEARNING_RATE= 0.001
###############################################################################
# We want to decode only these 7 muscles
###############################################################################
TARGET_MUSCLES = {"FCR","FDS","FDP","FCU","ECR","EDC","ECU"}

# A mapping from raw EMG labels => canonical muscle. We keep stubs for underscores, digits, etc.
GLOBAL_MUSCLE_MAP = {
    # ECR with underscore
    'ECR_1': 'ECR',
    'ECR_2': 'ECR',
    # EDC
    'EDC_1': 'EDC',
    'EDC_2': 'EDC',
    # FCR
    'FCR_1': 'FCR',
    # FCU
    'FCU_1': 'FCU',
    # FDS
    'FDS_1': 'FDS',
    'FDS_2': 'FDS',
    # FDP
    'FDP_1': 'FDP',
    'FDP_2': 'FDP',
    # ECU
    'ECU_1': 'ECU',
}


###############################################################################
# unify_spike_headers => so all rows have same # of spike columns
###############################################################################
def unify_spike_headers(df, spike_col="spike_counts", verbose=True):
    if verbose:
        debug_print("[DEBUG] unify_spike_headers: Start")
    all_neurons = set()
    for i, row in df.iterrows():
        spk_df = row.get(spike_col)
        if isinstance(spk_df, pd.DataFrame) and not spk_df.empty:
            all_neurons.update(spk_df.columns)
    all_neuron_list = sorted(list(all_neurons))
    debug_print(f"[DEBUG] unify_spike_headers => total unique neurons={len(all_neuron_list)}")

    df_aligned = df.copy()
    for i, row in df_aligned.iterrows():
        spk_df = row.get(spike_col)
        if not isinstance(spk_df, pd.DataFrame) or spk_df.empty:
            df_aligned.at[i, spike_col] = pd.DataFrame(0,index=[],columns=all_neuron_list)
        else:
            spk_df = spk_df.reindex(columns=all_neuron_list, fill_value=0)
            df_aligned.at[i, spike_col] = spk_df
    return df_aligned, all_neuron_list

###############################################################################
# unify_emg_labels => map raw => canonical => keep only 7 => reindex union
###############################################################################
def drop_cols_if_all_zero_in_train_or_test(df_train, df_test, columns):
    """
    Given two DataFrames (train & test) that have EMG columns = columns,
    we check each column. If it's ALL zero in either df_train or df_test,
    we skip it.

    Returns:
      df_train_pruned,
      df_test_pruned,
      final_cols (list of columns that are not all-zero in both sets)
    """
    debug_print(f"[DROP ZERO] Checking {len(columns)} columns in train/test for all-zero.")
    columns_to_keep = []
    n_dropped = 0

    for col in columns:
        # Gather train data for this column
        train_vals = []
        for _, row in df_train.iterrows():
            emg_df = row.get('EMG')
            if emg_df is not None and not emg_df.empty:
                train_vals.extend(emg_df[col].values)

        # Gather test data
        test_vals = []
        for _, row in df_test.iterrows():
            emg_df = row.get('EMG')
            if emg_df is not None and not emg_df.empty:
                test_vals.extend(emg_df[col].values)

        # Check
        if (not np.allclose(train_vals, 0.0)) and (not np.allclose(test_vals, 0.0)):
            columns_to_keep.append(col)
        else:
            n_dropped += 1

    debug_print(f"[DROP ZERO] => keeping={len(columns_to_keep)}, dropped={n_dropped}")
    if not columns_to_keep:
        return pd.DataFrame(), pd.DataFrame(), []

    df_train_pruned = reindex_emg_in_df(df_train, columns_to_keep)
    df_test_pruned  = reindex_emg_in_df(df_test,  columns_to_keep)

    return df_train_pruned, df_test_pruned, columns_to_keep

def gather_emg_columns(df_subset):
    all_cols = set()
    for _, row in df_subset.iterrows():
        emg_df = row.get('EMG')
        if isinstance(emg_df, pd.DataFrame) and not emg_df.empty:
            all_cols.update(emg_df.columns)
    return all_cols

def reindex_emg_in_df(df_subset, columns):
    """
    Reindex each row's EMG DataFrame to the given list of columns.
    Missing columns become 0; extra columns are dropped.
    """
    debug_print(f"[REINDEX] #rows={len(df_subset)}, target columns={list(columns)}")

    rows = []
    for i, row in df_subset.iterrows():
        emg_df = row.get('EMG')
        if isinstance(emg_df, pd.DataFrame) and not emg_df.empty:
            old_cols = emg_df.columns
            row['EMG'] = emg_df.reindex(columns=columns, fill_value=0.0)
            new_cols = row['EMG'].columns
            debug_print(
                f"   Row {i} => old_cols={list(old_cols)}, "
                f"-> new_cols={list(new_cols)} shape={row['EMG'].shape}"
            )
        rows.append(row)
    return pd.DataFrame(rows)

def unify_emg_labels(emg_df):
    if not isinstance(emg_df, pd.DataFrame) or emg_df.empty:
        return pd.DataFrame()
    new_cols={}
    count_for_muscle=defaultdict(int)

    for col in emg_df.columns:
        raw = col.strip().upper()
        tmp = raw
        # strip trailing digits if needed:
        while len(tmp)>0 and (tmp[-1].isdigit() or tmp[-1].isalpha()):
            if tmp in GLOBAL_MUSCLE_MAP:
                break
            tmp=tmp[:-1]
        if tmp=='':
            tmp=raw
        base = GLOBAL_MUSCLE_MAP.get(tmp, None)
        if base is None:
            continue
        if base not in TARGET_MUSCLES:
            continue
        count_for_muscle[base]+=1
        new_name = f"{base}_{count_for_muscle[base]}"
        new_cols[new_name] = emg_df[col]
    return pd.DataFrame(new_cols)

def filter_to_7_muscles(df):
    rows=[]
    all_cols=set()
    for i, row in df.iterrows():
        emg_df=row.get('EMG')
        if not isinstance(emg_df, pd.DataFrame):
            row['EMG']=pd.DataFrame()
            rows.append(row)
            continue
        new_emg = unify_emg_labels(emg_df)
        row['EMG']= new_emg
        rows.append(row)

    df_new= pd.DataFrame(rows)
    # gather union of columns
    for i, row in df_new.iterrows():
        e=row['EMG']
        if isinstance(e,pd.DataFrame) and not e.empty:
            all_cols.update(e.columns)
    all_cols= sorted(list(all_cols))

    # reindex each row
    new_rows=[]
    for i, row in df_new.iterrows():
        e=row['EMG']
        if isinstance(e, pd.DataFrame):
            row['EMG'] = e.reindex(columns=all_cols, fill_value=0)
        new_rows.append(row)
    return pd.DataFrame(new_rows)
###############################################################################
# BINNING + SMOOTHING
###############################################################################
def smooth_emg(emg_array, window_size=20):
    """
    Rectifies and smooths an EMG signal along axis=0 (time axis) with a simple moving average.
    
    Parameters:
      emg_array: numpy array of shape (n_samples, n_channels)
      window_size: size of the moving average window
      
    Returns:
      smoothed_emg: numpy array with the same shape as emg_array
    """
    rectified_emg = np.abs(emg_array)
    normalized_emg = (rectified_emg - rectified_emg.min()) / (rectified_emg.max() - rectified_emg.min())
    smoothed_emg = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'),
                                        axis=0, arr=normalized_emg)
    return smoothed_emg

def downsample_spike_and_emg(spike_df, emg_df, bin_factor=1):
    """
    Downsample spike and EMG data by bin_factor.
    Spikes get summed; EMG gets averaged.
    Returns (spike_downsampled_df, emg_downsampled_df).
    """
    # If there's no data, just return as-is
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return pd.DataFrame(), pd.DataFrame()

    T_old, n_units = spike_df.shape
    # We'll only keep multiples of bin_factor
    T_new = T_old // bin_factor

    # -- Downsample spikes: sum across each bin
    spk_arr = spike_df.values[: T_new * bin_factor, :]          # shape => (T_new*bin_factor, n_units)
    spk_arr = spk_arr.reshape(T_new, bin_factor, n_units)       # => (T_new, bin_factor, n_units)
    spk_arr = spk_arr.sum(axis=1)                               # => (T_new, n_units)
    ds_spike_df = pd.DataFrame(spk_arr, columns=spike_df.columns)

    # -- Downsample EMG: average within each bin
    if isinstance(emg_df, pd.DataFrame):
        e_arr = emg_df.values
        col_names = emg_df.columns
    else:
        # If your code might store EMG as a numpy array, handle that.
        e_arr = np.array(emg_df)
        col_names = None

    if e_arr.shape[0] < bin_factor:
        return ds_spike_df, pd.DataFrame()

    e_arr = e_arr[: T_new * bin_factor, ...]          # shape => (T_new*bin_factor, n_emg)
    e_arr = e_arr.reshape(T_new, bin_factor, e_arr.shape[1])  # => (T_new, bin_factor, n_emg)
    e_arr = e_arr.mean(axis=1)                                 # => (T_new, n_emg)

    ds_emg_df = pd.DataFrame(e_arr, columns=col_names) if col_names is not None else pd.DataFrame(e_arr)
    return ds_spike_df, ds_emg_df

def smooth_spike_data(x_2d, bin_size=0.001, smoothing_length=0.05):
    """
    Applies a Gaussian filter channel-by-channel.
    x_2d shape: (T, #neurons)
    bin_size:   e.g. 0.01 if using 10-ms bins 
                (or 0.001 if still thinking "1 ms" but remember your data is binned now)
    smoothing_length: e.g. 0.05 => 50 ms

    sigma = (smoothing_length / bin_size) / 2
    """
    sigma = (smoothing_length / bin_size) / 2
    out = np.zeros_like(x_2d, dtype=float)
    for ch in range(x_2d.shape[1]):
        out[:, ch] = gaussian_filter1d(x_2d[:, ch], sigma=sigma)
    return out

###############################################################################
# MODELS
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru=nn.GRU(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        out,_=self.gru(x)
        out=out[:,-1,:]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        out,_=self.lstm(x)
        out=out[:,-1,:]
        return self.fc(out)

class LiGRUCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.x2z=nn.Linear(input_size,hidden_size)
        self.h2z=nn.Linear(hidden_size,hidden_size,bias=False)
        self.x2h=nn.Linear(input_size,hidden_size)
        self.h2h=nn.Linear(hidden_size,hidden_size,bias=False)
    def forward(self,x,h):
        z=torch.sigmoid(self.x2z(x)+self.h2z(h))
        h_cand=torch.relu(self.x2h(x)+self.h2h(h))
        h_next=(1-z)*h+z*h_cand
        return h_next

class LiGRUDecoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.cell=LiGRUCell(input_size,hidden_size)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        bsz=x.shape[0]
        h=torch.zeros(bsz,self.hidden_size,device=x.device)
        for t in range(x.shape[1]):
            h=self.cell(x[:,t,:],h)
        return self.fc(h)

class LinearLagDecoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.lin1=nn.Linear(input_dim,hidden_dim)
        self.act=nn.ReLU()
        self.lin2=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        x=self.lin1(x)
        x=self.act(x)
        return self.lin2(x)

def get_decoder_params(dec_name):
    if dec_name=='GRU':
        return (GRU_N_PCA, GRU_HIDDEN_DIM, GRU_K_LAG)
    elif dec_name=='LSTM':
        return (LSTM_N_PCA, LSTM_HIDDEN_DIM, LSTM_K_LAG)
    elif dec_name=='LiGRU':
        return (LIGRU_N_PCA, LIGRU_HIDDEN_DIM, LIGRU_K_LAG)
    else:
        return (LINEAR_N_PCA, LINEAR_HIDDEN_DIM, LINEAR_K_LAG)

def build_decoder_model(dec_name, input_dim, hidden_dim, output_dim):
    if dec_name=='GRU':
        return GRUDecoder(input_dim, hidden_dim, output_dim)
    elif dec_name=='LSTM':
        return LSTMDecoder(input_dim, hidden_dim, output_dim)
    elif dec_name=='LiGRU':
        return LiGRUDecoder(input_dim, hidden_dim, output_dim)
    else:
        return LinearLagDecoder(input_dim, hidden_dim, output_dim)

###############################################################################
# TRAIN/EVAL
###############################################################################
def train_model(model, X_train, Y_train, num_epochs, batch_size, lr):
    ds=TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                     torch.tensor(Y_train,dtype=torch.float32))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True)
    opt=optim.Adam(model.parameters(),lr=lr)
    crit=nn.MSELoss()

    for ep in range(num_epochs):
        model.train()
        total_loss=0.0
        for xb,yb in dl:
            xb=xb.to(DEVICE)
            yb=yb.to(DEVICE)
            opt.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        if ep%10==0:
            debug_print(f"[TRAIN] ep={ep}, avg_loss={total_loss/len(dl):.4f}")

def eval_model(model,X_test,batch_size=64):
    preds=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(X_test),batch_size):
            xb=torch.tensor(X_test[i:i+batch_size],dtype=torch.float32).to(DEVICE)
            out=model(xb)
            preds.append(out.cpu().numpy())
    if preds:
        return np.concatenate(preds,axis=0)
    else:
        print("AYO EMPTY PRED WAKE UP")
        return np.empty((0,))

def compute_vaf_multi(Y_true, Y_pred):
    eps=1e-12
    n_ch=Y_true.shape[1]
    vaf_ch=[]
    for c in range(n_ch):
        var_true=np.var(Y_true[:,c], ddof=1)
        if var_true<eps:
            vaf_ch.append(np.nan)
            continue
        var_resid=np.var(Y_true[:,c]-Y_pred[:,c], ddof=1)
        vaf_ch.append(1 - var_resid/var_true)
    vaf_ch=np.array(vaf_ch,dtype=np.float32)
    mean_vaf=np.nanmean(vaf_ch)
    return vaf_ch, mean_vaf

def compute_mse_multi(Y_true,Y_pred):
    n_ch=Y_true.shape[1]
    mse_ch=[]
    for c in range(n_ch):
        mse_val=np.mean((Y_true[:,c]-Y_pred[:,c])**2)
        mse_ch.append(mse_val)
    mse_ch=np.array(mse_ch,dtype=np.float32)
    mean_mse=np.nanmean(mse_ch)
    return mse_ch, mean_mse

###############################################################################
# BUILD SPIKES => X
###############################################################################
def build_continuous_dataset(df_subset,
                             bin_factor=1,
                             bin_size=0.02,
                             smoothing_length=0.05):
    """
    Builds a continuous dataset by:
      1) Downsampling (binning) spikes & EMG
      2) Smoothing the binned spikes with a Gaussian
      3) Concatenating across all rows of df_subset
    Returns (big_spike_arr, big_emg_arr)
    """
    debug_print(f"[DEBUG] build_continuous_dataset: #rows={len(df_subset)}")

    all_spike_list = []
    all_emg_list   = []

    for i, row in df_subset.iterrows():
        spk_df = row.get("spike_counts")
        emg_df = row.get("EMG")
        if not isinstance(spk_df, pd.DataFrame) or spk_df.empty:
            debug_print(f"   Row {i}: empty or invalid spk_df => skip")
            continue
        if not isinstance(emg_df, pd.DataFrame) or emg_df.empty:
            debug_print(f"   Row {i}: empty or invalid emg_df => skip")
            continue

        ds_spike_df, ds_emg_df = downsample_spike_and_emg(spk_df, emg_df, bin_factor)
        if ds_spike_df.empty or ds_emg_df.empty:
            debug_print(f"   Row {i}: after binning => spike={ds_spike_df.shape}, emg={ds_emg_df.shape}, skipping row.")
            continue

        # Print the shape right after binning
        debug_print(
            f"   Row {i}: binned spk => {ds_spike_df.shape}, "
            f"binned emg => {ds_emg_df.shape}"
        )

        spk_arr = ds_spike_df.values
        spk_smoothed = smooth_spike_data(spk_arr, bin_size=bin_size, smoothing_length=smoothing_length)
        
        # Print shape after smoothing
        debug_print(f"   Row {i}: smoothed spk => {spk_smoothed.shape}")
        emg_values = ds_emg_df.values
        print("[DEBUG] Before rectification: min =", emg_values.min(), "max =", emg_values.max())
        smoothed_emg = smooth_emg(emg_values, window_size=5)  # adjust window_size as needed
        
        all_spike_list.append(spk_smoothed)
        all_emg_list.append(smoothed_emg)

    if len(all_spike_list) == 0:
        debug_print(f"[DEBUG] => No valid rows => returning empty arrays.")
        return np.empty((0, 0)), np.empty((0, 0))

    X_all = np.concatenate(all_spike_list, axis=0)
    Y_all = np.concatenate(all_emg_list,   axis=0)

    debug_print(f"[DEBUG] build_continuous_dataset => final X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all

###############################################################################
# CREATE RNN OR LINEAR DATASET => merges X, Y
###############################################################################
def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    """
    For RNN, we shape X => (samples, seq_len, #spikes)
    Y => (samples, #emg_ch)
    We use the last frame in the seq for Y.
    """
    if X_arr.shape[0]<=seq_len:
        return np.empty((0,seq_len,X_arr.shape[1])), np.empty((0,Y_arr.shape[1]))
    X_out, Y_out=[], []
    for i in range(seq_len, X_arr.shape[0]):
        window=X_arr[i-seq_len:i,:]
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out,dtype=np.float32), np.array(Y_out,dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    """
    Flatten the past seq_len bins => shape => (samples, seq_len*#spikes)
    Y => (samples, #emg_ch)
    """
    if X_arr.shape[0]<=seq_len:
        return np.empty((0, seq_len*X_arr.shape[1])), np.empty((0,Y_arr.shape[1]))
    X_out, Y_out=[], []
    for i in range(seq_len, X_arr.shape[0]):
        window= X_arr[i-seq_len:i,:].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out,dtype=np.float32), np.array(Y_out,dtype=np.float32)

###############################################################################
# REALIGNMENT / PCA logic (domain adaptation stubs)
###############################################################################
def gather_day_level_spikes(df_subset):
    spike_arr, _ = build_continuous_dataset(df_subset, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
    return spike_arr

def prepare_realignment_pca(df_multi,monkeys,n_components=20):
    pca_objs={
        'realignment':{},
        'ref':{}
    }
    for M in monkeys:
        df_m=df_multi[df_multi['monkey']==M]
        days=sorted(df_m['date'].unique())
        if not days: continue
        ref_day=days[0]
        df_ref=df_m[df_m['date']==ref_day]
        X_ref=gather_day_level_spikes(df_ref)
        if X_ref.shape[0]<2: continue
        ref_pca=PCA(n_components=n_components,random_state=SEED)
        ref_pca.fit(X_ref)
        pca_objs['ref'][M]=ref_pca
        for d_i in days:
            df_day=df_m[df_m['date']==d_i]
            X_day=gather_day_level_spikes(df_day)
            if X_day.shape[0]<2: continue
            local_pca=PCA(n_components=n_components,random_state=SEED)
            local_pca.fit(X_day)
            R= local_pca.components_ @ ref_pca.components_.T
            pca_objs['realignment'][(M,d_i)] = {
                'local_pca': local_pca,
                'R': R
            }
    return pca_objs

def prepare_monkey_level_global_pca(df_multi, monkeys, n_components=20):
    pca_objs = { 'monkey_global': {} }

    for M in monkeys:
        df_m = df_multi[df_multi['monkey'] == M]
        # Destructure the build_continuous_dataset tuple
        spike_arr, _ = build_continuous_dataset(df_m, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)

        if spike_arr.shape[0] < 2:
            continue

        pca_m = PCA(n_components=n_components, random_state=SEED)
        pca_m.fit(spike_arr)
        pca_objs['monkey_global'][M] = pca_m

    return pca_objs

def compute_alignment_matrix(V_dayD,V_day0):
    return pinv(V_dayD)@V_day0

def apply_alignment_mode(x_2d, monkey_current, day_current,
                         alignment_mode, pca_objs,
                         n_components=20,
                         monkey_train=None):
    debug_print(f"[ALIGN] mode={alignment_mode}, monkey={monkey_current}, day={day_current}, shape_in={x_2d.shape}")
    if alignment_mode=='none':
        return x_2d
    if alignment_mode=='monkey_level':
        mg_pca= pca_objs['monkey_global'].get(monkey_current, None)
        if mg_pca is None:
            debug_print("[WARNING] no monkey_global => skip => raw dimension")
            return x_2d
        x_tr= mg_pca.transform(x_2d)
        debug_print(f"[ALIGN] => shape_out={x_tr.shape}, keeping first {n_components}")
        return x_tr[:, :n_components]

    if alignment_mode=='realignment':
        loc_info= pca_objs['realignment'].get((monkey_current,day_current),None)
        ref_pca= pca_objs['ref'].get(monkey_current,None)
        if (loc_info is None) or (ref_pca is None):
            debug_print("[WARNING] realignment missing => skip => raw dimension")
            return x_2d
        loc_pca= loc_info['local_pca']
        R= loc_info['R']
        x_loc= loc_pca.transform(x_2d)
        x_al = x_loc @ R
        debug_print(f"[ALIGN] => shape_loc={x_loc.shape}, shape_out={x_al.shape}")
        return x_al
    debug_print("[WARNING] unknown alignment => skip => raw dimension")
    return x_2d

###############################################################################
# SCENARIOS
###############################################################################
def build_scenarios():
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
    scenarios.append({
    'name': 'Jango_iso',
    'train_filter': f_jango_iso,
    'split_ratio': 0.25,
    'tests': [
        {'test_filter': f_jango_wm, 'name': 'jango_wm'},
        {'test_filter': f_jango_spr, 'name': 'jango_spr'}
    ],
    'force_same_day': True
    })


    return scenarios

###############################################################################
# RUN EXPERIMENTS => merges spikes => X, EMG => Y, plus alignment modes
###############################################################################
def run_experiments(df_multi, alignment_modes, pca_objs, decoders, results_file="df_results_emg.pkl",):
    if os.path.exists(results_file):
        df_existing = pd.read_pickle(results_file)
        debug_print(f"[INFO] Loaded existing results from {results_file}, shape={df_existing.shape}")
    else:
        df_existing = pd.DataFrame()
        debug_print(f"[INFO] No existing file found. Will create {results_file} when saving.")

    
    scenarios = build_scenarios()
    print(scenarios)
    for sc in scenarios:
        sname = sc['name']

        # Skip scenario if we already have it
        if not df_existing.empty and sname in df_existing['scenario_name'].unique():
            debug_print(f"[INFO] Scenario '{sname}' is already in {results_file}. Skipping.")
            continue

        debug_print(f"\n=== SCENARIO: {sname} ===")
        scenario_results = []

        # Train subset
        df_train = df_multi[df_multi.apply(sc['train_filter'], axis=1)]
        if df_train.empty:
            debug_print(f"[WARNING] scenario={sname} => empty train => skip")
            continue

        force_same_day = sc.get("force_same_day", True)
        if force_same_day and 'date' in df_train.columns:
            train_days = sorted(df_train['date'].unique())
        else:
            train_days = [None]

        for train_day in train_days:
            if train_day is not None:
                df_train_day = df_train[df_train['date'] == train_day]
            else:
                df_train_day = df_train

            if df_train_day.empty:
                debug_print(f"[WARNING] scenario={sname}, day={train_day} => no train data => skip")
                continue

            # Gather EMG columns from train
            train_cols = gather_emg_columns(df_train_day)

            # Prepare test sets
            test_defs = sc.get('tests', [])
            test_cols_all = set()
            test_dfs = {}
            for tdef in test_defs:
                df_test = df_multi[df_multi.apply(tdef['test_filter'], axis=1)]
                if force_same_day and train_day is not None and 'date' in df_test.columns:
                    df_test = df_test[df_test['date'] == train_day]
                if df_test.empty:
                    debug_print(f"[DEBUG] scenario={sname}, test={tdef['name']} => empty test => skipping")
                    test_dfs[tdef['name']] = None
                    continue
                test_dfs[tdef['name']] = df_test
                test_cols = gather_emg_columns(df_test)
                test_cols_all.update(test_cols)

            # Intersection or just train_cols if no tests
            if test_defs:
                final_emg_cols = train_cols.intersection(test_cols_all)
            else:
                final_emg_cols = train_cols

            if not final_emg_cols:
                debug_print(f"[WARNING] scenario={sname}, day={train_day} => no intersection => skip")
                continue

            # Reindex train to final_emg_cols
            df_train_day_reidx = reindex_emg_in_df(df_train_day, final_emg_cols)

            # We might also reindex each test set to final_emg_cols, 
            # BUT we haven't built them yet, so let's do that once we confirm columns.
            # Actually, let's do it after we optionally drop zero columns below:

            # We'll gather all test DataFrames in a single combined "df_test_all" so 
            # we can check zero columns across them. 
            df_test_all = pd.DataFrame()
            # We'll keep a dict of reindexed test sets so we only do binning once per test set
            reindexed_test_dfs = {}
            for tdef in test_defs:
                df_t = test_dfs[tdef['name']]
                if df_t is None or df_t.empty:
                    reindexed_test_dfs[tdef['name']] = None
                else:
                    df_test_reidx = reindex_emg_in_df(df_t, final_emg_cols)
                    # We'll temporarily store them
                    reindexed_test_dfs[tdef['name']] = df_test_reidx
                    # We can combine them row-wise into a single df so we check zeros across them all
                    df_test_all = pd.concat([df_test_all, df_test_reidx], ignore_index=True)

            if not df_test_all.empty:
                df_train_day_reidx, _, final_cols_kept = drop_cols_if_all_zero_in_train_or_test(
                    df_train_day_reidx, df_test_all, final_emg_cols
                )
                if not final_cols_kept:
                    debug_print(f"[WARNING] after dropping zero cols => no columns left => skip day={train_day}")
                    continue
                # Now we must re-split df_test_all_final back into separate test sets by scenario name
                # We'll do it by monkeypatching a "test_name" column or by re-filtering. 
                # Simpler approach: just reindex each test set again:
                for tdef in test_defs:
                    df_t = reindexed_test_dfs[tdef['name']]
                    if df_t is None or df_t.empty:
                        continue
                    # Now drop zero columns in that set too
                    # Actually easiest is to reindex again to final_cols_kept:
                    df_reidx2 = reindex_emg_in_df(df_t, final_cols_kept)
                    reindexed_test_dfs[tdef['name']] = df_reidx2

                final_emg_cols = final_cols_kept
            else:
                # No test data => just skip dropping zero columns
                pass

            # Now build the actual train arrays from df_train_day_reidx
            X_train_raw, Y_train_raw = build_continuous_dataset(
                df_train_day_reidx, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
            )
            min_len = min(X_train_raw.shape[0], Y_train_raw.shape[0])
            if min_len < 10:
                debug_print("[WARNING] Not enough train data => skip this day/scenario")
                continue

            X_train_raw = X_train_raw[:min_len, :]
            Y_train_raw = Y_train_raw[:min_len, :]

            debug_print(f"[SCENARIO {sname}, day={train_day}] => final_emg_cols={sorted(final_emg_cols)}")
            debug_print(f"Train EMG stats => mean={Y_train_raw.mean():.4f}, std={Y_train_raw.std():.4f}")

            split_ratio = sc.get('split_ratio', None)
            if split_ratio and 0 < split_ratio < 1.0:
                X_tr_raw, X_val_raw, Y_tr_raw, Y_val_raw = train_test_split(
                    X_train_raw, Y_train_raw, test_size=split_ratio, random_state=SEED
                )
            else:
                X_tr_raw, Y_tr_raw = X_train_raw, Y_train_raw
                X_val_raw, Y_val_raw = np.empty((0,0)), np.empty((0,0))

            train_monkey = df_train_day_reidx.iloc[0]['monkey'] if not df_train_day_reidx.empty else '?'
            day_current  = df_train_day_reidx.iloc[0]['date'] if ('date' in df_train_day_reidx.columns and not df_train_day_reidx.empty) else None

            for mode in alignment_modes:
                for dec_name in decoders:
                    n_comp, hidden_dim, k_lag = get_decoder_params(dec_name)

                    # Apply alignment
                    X_tr_aligned = apply_alignment_mode(
                        X_tr_raw, monkey_current=train_monkey, day_current=day_current,
                        alignment_mode=mode, pca_objs=pca_objs,
                        n_components=n_comp, monkey_train=train_monkey
                    )

                    # Build RNN or linear input
                    if dec_name in ['GRU','LSTM','LiGRU']:
                        X_tr_f, Y_tr_f = create_rnn_dataset_continuous(X_tr_aligned, Y_tr_raw, k_lag)
                        if X_tr_f.shape[0] < 50:
                            continue
                        input_dim = X_tr_f.shape[2]
                        output_dim = Y_tr_f.shape[1]
                    else:
                        X_tr_f, Y_tr_f = create_linear_dataset_continuous(X_tr_aligned, Y_tr_raw, k_lag)
                        if X_tr_f.shape[0] < 50:
                            continue
                        input_dim = X_tr_f.shape[1]
                        output_dim = Y_tr_f.shape[1]

                    # Train model
                    print(f'======= TRAINNING MODEL : {dec_name} =======')
                    print(f"input dimension: { input_dim}, output dimension { output_dim}")
                    model = build_decoder_model(dec_name, input_dim, hidden_dim, output_dim).to(DEVICE)
                    train_model(model, X_tr_f, Y_tr_f, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

                    # Evaluate on validation if we have it
                    if split_ratio and X_val_raw.size > 0 and Y_val_raw.size > 0:
                        print(f'Shape of Validation sets X: {X_val_raw.shape[0]}, Y: {Y_val_raw.shape[0]}')
                        min_len_val = min(X_val_raw.shape[0], Y_val_raw.shape[0])
                        if min_len_val >= 50:
                            X_val_cut = X_val_raw[:min_len_val, :]
                            Y_val_cut = Y_val_raw[:min_len_val, :]
                            X_val_aligned = apply_alignment_mode(
                                X_val_cut, monkey_current=train_monkey, day_current=day_current,
                                alignment_mode=mode, pca_objs=pca_objs,
                                n_components=n_comp, monkey_train=train_monkey
                            )
                            if dec_name in ['GRU','LSTM','LiGRU']:
                                X_val_f, Y_val_f = create_rnn_dataset_continuous(X_val_aligned, Y_val_cut, k_lag)
                            else:
                                X_val_f, Y_val_f = create_linear_dataset_continuous(X_val_aligned, Y_val_cut, k_lag)

                            if X_val_f.shape[0] >= 50:
                                preds_val = eval_model(model, X_val_f)
                                vaf_ch, vaf_avg = compute_vaf_multi(Y_val_f, preds_val)
                                mse_ch, mse_avg = compute_mse_multi(Y_val_f, preds_val)
                                print(f"Validation VAF for decoder {dec_name} on day {train_day}:")
                                print(f"Mean VAF: {vaf_avg:.4f}")
                                for i, vaf in enumerate(vaf_ch):
                                    print(f"  Channel {i}: VAF = {vaf:.4f}")
                                    # Create a time axis based on the number of validation samples
                                    t = np.arange(Y_val_f.shape[0])
                                    plt.figure(figsize=(10, 4))
                                    plt.plot(t, Y_val_f[:, i], label="True")
                                    plt.plot(t, preds_val[:, i], label="Predicted", alpha=0.7)
                                    plt.title(f"Validation: Channel {i} Output Comparison")
                                    plt.xlabel("Time")
                                    plt.ylabel("Output")
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.show()
                                scenario_results.append({
                                    'scenario_name': sname,
                                    'decoder_type': dec_name,
                                    'alignment_mode': mode,
                                    'train_monkey': train_monkey,
                                    'test_monkey': train_monkey,
                                    'test_name': 'val_25pct',
                                    'VAF': vaf_avg,
                                    'MSE': mse_avg,
                                    'VAF_ch': vaf_ch,
                                    'MSE_ch': mse_ch,
                                    'n_emg_ch': output_dim,
                                    'timestamp': datetime.datetime.now(),
                                    'date': train_day
                                })

                    # Evaluate on actual test sets
                    for tdef in test_defs:
                        df_test_reidx = reindexed_test_dfs[tdef['name']]
                        if df_test_reidx is None or df_test_reidx.empty:
                            continue
                        X_test_raw, Y_test_raw = build_continuous_dataset(
                            df_test_reidx, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH
                        )
                        min_len_test = min(X_test_raw.shape[0], Y_test_raw.shape[0])
                        if min_len_test < 50:
                            continue

                        X_test_cut = X_test_raw[:min_len_test, :]
                        Y_test_cut = Y_test_raw[:min_len_test, :]
                        test_monkey = df_test_reidx.iloc[0]['monkey'] if not df_test_reidx.empty else '?'
                        day_test = df_test_reidx.iloc[0]['date'] if ('date' in df_test_reidx.columns and not df_test_reidx.empty) else None

                        X_test_aligned = apply_alignment_mode(
                            X_test_cut, monkey_current=test_monkey, day_current=day_test,
                            alignment_mode=mode, pca_objs=pca_objs,
                            n_components=n_comp, monkey_train=train_monkey
                        )

                        if dec_name in ['GRU','LSTM','LiGRU']:
                            X_te_f, Y_te_f = create_rnn_dataset_continuous(X_test_aligned, Y_test_cut, k_lag)
                        else:
                            X_te_f, Y_te_f = create_linear_dataset_continuous(X_test_aligned, Y_test_cut, k_lag)

                        if X_te_f.shape[0] < 50:
                            continue
                        
                                    
                        preds_test = eval_model(model, X_te_f)
                        debug_print("Pred stats:", preds_test.mean(), preds_test.std())
                        print("Predictions mean:", preds_test.mean(), "std:", preds_test.std())
                        print("True Y_test mean:", Y_te_f.mean(), "std:", Y_te_f.std())
                        vaf_ch, vaf_avg = compute_vaf_multi(Y_te_f, preds_test)
                        mse_ch, mse_avg = compute_mse_multi(Y_te_f, preds_test)
                        print(f"VAF for decoder {dec_name} on day {train_day} for test '{tdef['name']}':")
                        print(f"Mean VAF: {vaf_avg}")
                        for i, vaf in enumerate(vaf_ch):
                            print(f"Channel {i}: VAF = {vaf:.4f}")
                            # Create a time axis of the same length as the number of samples.
                            t = np.arange(Y_te_f.shape[0])
                            plt.figure(figsize=(10, 4))
                            plt.plot(t, Y_te_f[:, i], label="True")
                            plt.plot(t, preds_test[:, i], label="Predicted", alpha=0.7)
                            plt.title(f"Channel {i} Output Comparison")
                            plt.xlabel("Time")
                            plt.ylabel("Output")
                            plt.legend()
                            plt.tight_layout()
                            plt.show()
                        scenario_results.append({
                            'scenario_name': sname,
                            'decoder_type': dec_name,
                            'alignment_mode': mode,
                            'train_monkey': train_monkey,
                            'test_monkey': test_monkey,
                            'test_name': tdef['name'],
                            'VAF': vaf_avg,
                            'MSE': mse_avg,
                            'VAF_ch': vaf_ch,
                            'MSE_ch': mse_ch,
                            'n_emg_ch': output_dim,
                            'timestamp': datetime.datetime.now(),
                            'date': train_day
                        })

        # End of train_days
        if scenario_results:
            df_scenario = pd.DataFrame(scenario_results)
            df_existing = pd.concat([df_existing, df_scenario], ignore_index=True)
            df_existing = add_task_columns(df_existing)
            df_existing.to_pickle(results_file)
            debug_print(f"[INFO] Finished scenario '{sname}'. Appended {len(df_scenario)} rows -> {results_file}")
        else:
            debug_print(f"[INFO] Scenario '{sname}' produced no new results.")

    return df_existing



###############################################################################
# parse tasks => confusion
###############################################################################
def parse_train_task(row):
    name=row.get('scenario_name','').lower()
    tasks=['iso','iso8','wm','spr','mg-pt','mgpt','ball']
    for t in tasks:
        if t in name:
            return t
    return None

def parse_test_task(row):
    tname=row.get('test_name','').lower()
    if 'val_25pct' in tname:
        return row.get('train_task','val')
    tasks=['iso','iso8','wm','spr','mg-pt','mgpt','ball']
    for t in tasks:
        if t in tname:
            return t
    return None

def add_task_columns(df):
    if 'train_task' not in df.columns:
        df['train_task']= df.apply(parse_train_task, axis=1)
    if 'test_task' not in df.columns:
        df['test_task']= df.apply(parse_test_task, axis=1)
    return df

###############################################################################
# MAIN
###############################################################################
def main():
    # Hard-coded paths
    df_in = "output.pkl" 
    df_out= "df_results_emg_godsentpleasehelp.pkl"

    if not os.path.exists(df_in):
        print(f"[ERROR] file not found => {df_in}")
        sys.exit(1)

    # 1) Load
    df_multi= pd.read_pickle(df_in)
    debug_print(f"[INFO] loaded df_multi => shape={df_multi.shape}")

    # 2) unify spike => same #neurons
    df_multi_aligned, neuron_list= unify_spike_headers(df_multi, "spike_counts")
    debug_print("[DEBUG] after unify_spike_headers => shape=", df_multi_aligned.shape)

    # 3) unify EMG => only 7 target muscles
    df_filtered= filter_to_7_muscles(df_multi_aligned)
    debug_print("[INFO] after filter_to_7_muscles => shape=", df_filtered.shape)

    # 4) build PCA objects if realignment or monkey_level
    monkeys= df_filtered['monkey'].unique()
    max_pca= max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
    pca_real= prepare_realignment_pca(df_filtered, monkeys, n_components=max_pca)
    pca_monk= prepare_monkey_level_global_pca(df_filtered, monkeys, n_components=max_pca)
    pca_objs={}
    pca_objs.update(pca_real)
    pca_objs.update(pca_monk)

    # 5) run with alignment_modes => e.g. ['none','realignment','monkey_level']
    alignment_modes= ["realignment"]
    decoders = ["Linear"] # ["GRU", "LSTM","LiGRU", "Linear"]
    # alignment_modes=['none','realignment','monkey_level']
    df_results = run_experiments(df_filtered, alignment_modes, pca_objs,decoders, results_file=df_out)

    # 6) parse tasks => confusion
    # df_results= add_task_columns(df_results)

    # 7) save
    # df_results.to_pickle(df_out)
    # print(f"[INFO] done. saved => {df_out}, shape={df_results.shape}")

if __name__=="__main__":
    main()
