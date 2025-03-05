import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import datetime
import sys

# Enable or disable global verbosity here:
GLOBAL_VERBOSE = True

def debug_print(*msg):
    """Helper to print debug messages if GLOBAL_VERBOSE is True."""
    if GLOBAL_VERBOSE:
        print(*msg)

###############################################################################
#                           DEVICE SETUP
###############################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_print(f"[DEBUG] Using device: {DEVICE}")

###############################################################################
#                           A) BASIC UTILITIES
###############################################################################
def compute_vaf(y_true, y_pred):
    """
    Variance Accounted For (VAF).
    VAF = 1 - Var(residuals)/Var(y_true).
    """
    var_true = np.var(y_true, ddof=1)
    var_resid= np.var(y_true - y_pred, ddof=1)
    if var_true < 1e-12:
        return np.nan
    return 1 - (var_resid / var_true)

###############################################################################
#                           B) ALIGN HEADERS
###############################################################################
def unify_spike_headers(df, spike_col="spike_counts", verbose=True):
    """
    Ensures that each row in df[spike_col] is a DataFrame with the same columns,
    in the same order, adding missing columns (zeros) and dropping extras.
    """
    if verbose:
        debug_print("[DEBUG] unify_spike_headers: Start")
        debug_print(f"[DEBUG] DataFrame shape: {df.shape}, spike_col='{spike_col}'")

    all_neurons = set()
    n_empty_spike_counts = 0
    for idx, row in df.iterrows():
        sc = row.get(spike_col)
        if isinstance(sc, pd.DataFrame):
            if sc.empty:
                n_empty_spike_counts += 1
            else:
                all_neurons.update(sc.columns)
        else:
            n_empty_spike_counts += 1

    all_neuron_list = sorted(all_neurons)
    if verbose:
        debug_print(f"[DEBUG] Found {len(all_neuron_list)} unique columns total.")
        debug_print(f"[DEBUG] Found {n_empty_spike_counts} empty/invalid spike_counts rows.")
        if len(all_neuron_list) > 10:
            debug_print(f"[DEBUG] Example of final neuron list (first 10): {all_neuron_list[:10]} ...")
        else:
            debug_print(f"[DEBUG] Neuron list: {all_neuron_list}")

    df_aligned = df.copy()
    n_rows_modified = 0
    for idx, row in df_aligned.iterrows():
        sc = row.get(spike_col)
        if not isinstance(sc, pd.DataFrame) or sc.empty:
            new_df = pd.DataFrame(0, index=[], columns=all_neuron_list)
            if verbose:
                debug_print(f"[DEBUG] Row index={idx} => empty/invalid spike_counts -> created 0-filled shape={new_df.shape}")
        else:
            missing_cols = [col for col in all_neuron_list if col not in sc.columns]
            if missing_cols and verbose:
                debug_print(f"[DEBUG] Row index={idx} => adding {len(missing_cols)} missing columns.")
            for col in missing_cols:
                sc[col] = 0
            
            extra_cols = [col for col in sc.columns if col not in all_neuron_list]
            if extra_cols and verbose:
                debug_print(f"[DEBUG] Row index={idx} => dropping {len(extra_cols)} extra columns.")
            if extra_cols:
                sc = sc.drop(columns=extra_cols)

            sc = sc[all_neuron_list]
            new_df = sc
        
        df_aligned.at[idx, spike_col] = new_df
        n_rows_modified += 1

    if verbose:
        debug_print(f"[DEBUG] unify_spike_headers: Finished on {n_rows_modified} rows.")
    return df_aligned, all_neuron_list

###############################################################################
#                           C) DATA HANDLING
###############################################################################
def build_continuous_dataset(df_subset, bin_factor, bin_size, smoothing_length):
    """
    Merges all rows in df_subset => continuous X (spikes) & Y (force).
    This example does minimal actual smoothing, just merges row by row.
    """
    debug_print(f"[DEBUG] build_continuous_dataset: #rows={len(df_subset)}, bin_factor={bin_factor}, bin_size={bin_size}, smooth_len={smoothing_length}")
    all_spike_list = []
    all_force_list = []
    for idx, row in df_subset.iterrows():
        spk_df = row.get("spike_counts")
        frc_df = row.get("force")
        if not isinstance(spk_df, pd.DataFrame) or spk_df.empty:
            debug_print(f"[DEBUG] row={idx} => invalid spike_counts, skipping.")
            continue
        if frc_df is None or len(frc_df) == 0:
            debug_print(f"[DEBUG] row={idx} => invalid force, skipping.")
            continue
        # We'll assume shape (T, n_units) for spk, shape (T,2) for force => take X=force[:,0]
        spk_arr = spk_df.values
        if hasattr(frc_df, 'values'):
            frc_arr = frc_df.values
        else:
            frc_arr = np.array(frc_df)
        force_x = frc_arr[:,0]

        all_spike_list.append(spk_arr)
        all_force_list.append(force_x)
    
    if not all_spike_list:
        debug_print("[DEBUG] build_continuous_dataset => no valid data, returning empty arrays.")
        return np.empty((0,)), np.empty((0,))
    X_all = np.concatenate(all_spike_list, axis=0)
    Y_all = np.concatenate(all_force_list, axis=0)
    debug_print(f"[DEBUG] build_continuous_dataset => final shape X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all

def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0,))
    X_out, Y_out = [], []
    for i in range(seq_len, X_arr.shape[0]):
        window = X_arr[i-seq_len:i, :]
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len * X_arr.shape[1])), np.empty((0,))
    X_out, Y_out = [], []
    for i in range(seq_len, X_arr.shape[0]):
        window = X_arr[i-seq_len:i, :].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.float32)

###############################################################################
#                           D) MODEL DEFINITIONS
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

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
        h_next = (1 - z)*h + z*h_candidate
        return h_next

class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            h = self.cell(x[:,t,:], h)
        return self.fc(h)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        return self.lin2(x)

###############################################################################
#                           E) TRAINING/EVAL UTILS
###############################################################################
def train_model(model, X_train, Y_train, num_epochs=30, batch_size=64, lr=0.001):
    debug_print(f"[DEBUG] train_model: X_train={X_train.shape}, Y_train={Y_train.shape}, epochs={num_epochs}")
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train).unsqueeze(-1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit= nn.MSELoss()

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss= crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if ep % 10 == 0 and GLOBAL_VERBOSE:
            debug_print(f"[DEBUG] Epoch={ep}/{num_epochs}, avg_loss={total_loss/len(dl):.4f}")

def eval_model(model, X_test, batch_size=64):
    debug_print(f"[DEBUG] eval_model: X_test={X_test.shape}, batch_size={batch_size}")
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.tensor(X_test[i:i+batch_size]).float().to(DEVICE)
            out= model(xb)
            preds.append(out.cpu().numpy().flatten())
    if preds:
        all_preds = np.concatenate(preds)
        debug_print(f"[DEBUG] eval_model => final preds shape={all_preds.shape}")
        return all_preds
    else:
        debug_print("[DEBUG] eval_model => no preds, returning empty array.")
        return np.array([])

###############################################################################
#                           F) PCA PREPARATION
###############################################################################
def gather_day_level_spikes(df_subset, bin_factor=10, bin_size=0.001, smoothing_length=0.05):
    """
    Merge all tasks in a single day => day-level X array
    """
    X_all, _ = build_continuous_dataset(df_subset, bin_factor, bin_size, smoothing_length)
    return X_all

def prepare_local_pca_daylevel(df_multi, monkeys, n_components=20):
    debug_print("[DEBUG] prepare_local_pca_daylevel: start")
    pca_objects = {
        'day_level': {},  # local PCA per (M, day)
        'ref': {}         # reference PCA per monkey
    }
    for M in monkeys:
        debug_print(f"[DEBUG]   => monkey={M}")
        df_m = df_multi[df_multi['monkey'] == M]
        days = sorted(df_m['date'].unique())
        if not days:
            debug_print(f"[DEBUG]     => no days found for monkey={M}, skipping.")
            continue
        ref_day = days[0]
        debug_print(f"[DEBUG]     => reference day={ref_day}")
        df_ref = df_m[df_m['date'] == ref_day]
        X_ref  = gather_day_level_spikes(df_ref)
        if X_ref.shape[0] < 2:
            debug_print(f"[DEBUG]     => reference day data too small, shape={X_ref.shape}, skipping.")
            continue

        ref_pca= PCA(n_components=n_components, random_state=42)
        ref_pca.fit(X_ref)
        pca_objects['ref'][M] = ref_pca
        debug_print(f"[DEBUG]     => ref PCA fitted. shape={ref_pca.components_.shape}")

        for d_i in days:
            df_day = df_m[df_m['date'] == d_i]
            X_day  = gather_day_level_spikes(df_day)
            if X_day.shape[0] < 2:
                debug_print(f"[DEBUG]       => day={d_i} => data too small, shape={X_day.shape}, skip.")
                continue
            local_pca = PCA(n_components=n_components, random_state=42)
            local_pca.fit(X_day)
            # alignment R => shape [n_components, n_components]
            # The # of original features should match between local & ref for matmul to succeed
            R = local_pca.components_ @ ref_pca.components_.T

            pca_objects['day_level'][(M, d_i)] = {
                'local_pca': local_pca,
                'R': R
            }
            debug_print(f"[DEBUG]       => day={d_i}, local PCA shape={local_pca.components_.shape}, R shape={R.shape}")
    debug_print("[DEBUG] prepare_local_pca_daylevel: done\n")
    return pca_objects

def prepare_monkey_level_global_pca(df_multi, monkeys, n_components=20):
    debug_print("[DEBUG] prepare_monkey_level_global_pca: start")
    pca_objects = {
        'monkey_global': {}
    }
    for M in monkeys:
        df_m = df_multi[df_multi['monkey'] == M]
        X_all, _ = build_continuous_dataset(df_m, 10, 0.001, 0.05)
        debug_print(f"[DEBUG]   => monkey={M}, combined shape={X_all.shape}")
        if X_all.shape[0] < 2:
            debug_print(f"[DEBUG]     => skipping monkey={M}, not enough data.")
            continue
        pca_m = PCA(n_components=n_components, random_state=42)
        pca_m.fit(X_all)
        pca_objects['monkey_global'][M] = pca_m
        debug_print(f"[DEBUG]     => monkey={M}, PCA shape={pca_m.components_.shape}")
    debug_print("[DEBUG] prepare_monkey_level_global_pca: done\n")
    return pca_objects

###############################################################################
#                           G) APPLY ALIGNMENT MODE
###############################################################################
def apply_alignment_mode(x_2d, monkey, day, alignment_mode, pca_objs, n_components=20):
    """
    x_2d shape => [T, n_units]
    alignment_mode => 'none', 'day_level', 'monkey_level'
    """
    if alignment_mode == 'none':
        return x_2d

    if alignment_mode == 'monkey_level':
        global_pca = pca_objs['monkey_global'].get(monkey, None)
        if global_pca is None:
            debug_print(f"[DEBUG] apply_alignment_mode => monkey={monkey} not in monkey_global, returning x_2d.")
            return x_2d
        x_trans = global_pca.transform(x_2d)
        return x_trans[:, :n_components]

    if alignment_mode == 'day_level':
        local_info = pca_objs['day_level'].get((monkey, day), None)
        ref_pca    = pca_objs['ref'].get(monkey, None)
        if (local_info is None) or (ref_pca is None):
            debug_print(f"[DEBUG] apply_alignment_mode => no local_info/ref_pca for monkey={monkey}, day={day}")
            return x_2d
        local_pca = local_info['local_pca']
        R         = local_info['R']

        # transform + align
        x_local   = local_pca.transform(x_2d)        # [T, n_components]
        x_aligned = x_local @ R                      # [T, n_components]
        return x_aligned

    debug_print(f"[DEBUG] apply_alignment_mode => unknown mode={alignment_mode}, returning x_2d.")
    return x_2d

###############################################################################
#                           H) MAIN TRAIN/TEST LOOP
###############################################################################
def run_experiments(df_multi, alignment_modes, decoders, pca_objs, n_components=20):
    debug_print("[DEBUG] run_experiments: start")
    scenarios = [
        {
          'train': {'monkey': 'Jango', 'date': '20140725', 'task': 'iso'},
          'test':  {'monkey': 'Jango', 'date': '20141010', 'task': 'iso'}
        },
        {
          'train': {'monkey': 'Jango', 'date': '20140725', 'task': 'iso'},
          'test':  {'monkey': 'Jango', 'date': '20140725', 'task': 'wm'}
        },
    ]
    seq_len = 16
    results_list = []
    for sc in scenarios:
        debug_print(f"[DEBUG] SCENARIO: {sc}")
        M_train = sc['train']['monkey']
        D_train = sc['train']['date']
        T_train = sc['train']['task']
        M_test  = sc['test']['monkey']
        D_test  = sc['test']['date']
        T_test  = sc['test']['task']

        df_train = df_multi[(df_multi['monkey'] == M_train) &
                            (df_multi['date'] == D_train) &
                            (df_multi['task'] == T_train) &
                            (df_multi['force'].notnull())]
        df_test  = df_multi[(df_multi['monkey'] == M_test) &
                            (df_multi['date'] == D_test) &
                            (df_multi['task'] == T_test) &
                            (df_multi['force'].notnull())]
        if df_train.empty or df_test.empty:
            debug_print("[DEBUG]   => skipping scenario: empty train or test.")
            continue

        X_train_raw, Y_train_raw = build_continuous_dataset(df_train, 10, 0.001, 0.05)
        X_test_raw,  Y_test_raw  = build_continuous_dataset(df_test,  10, 0.001, 0.05)
        if len(X_train_raw) < seq_len*2 or len(X_test_raw) < seq_len*2:
            debug_print("[DEBUG]   => skipping scenario: not enough train/test samples.")
            continue

        for mode in alignment_modes:
            debug_print(f"[DEBUG]   => alignment_mode={mode}")
            X_train_red = apply_alignment_mode(X_train_raw, M_train, D_train, mode, pca_objs, n_components)
            X_test_red  = apply_alignment_mode(X_test_raw,  M_test,  D_test,  mode, pca_objs, n_components)

            for dec_name in decoders:
                debug_print(f"[DEBUG]     => decoder={dec_name}")
                if dec_name in ['GRU','LSTM','LiGRU']:
                    X_tr_f, Y_tr_f = create_rnn_dataset_continuous(X_train_red, Y_train_raw, seq_len)
                    X_te_f, Y_te_f = create_rnn_dataset_continuous(X_test_red,  Y_test_raw,  seq_len)
                    input_dim = X_tr_f.shape[2] if X_tr_f.ndim == 3 else 0
                else:
                    X_tr_f, Y_tr_f = create_linear_dataset_continuous(X_train_red, Y_train_raw, seq_len)
                    X_te_f, Y_te_f = create_linear_dataset_continuous(X_test_red,  Y_test_raw,  seq_len)
                    input_dim = X_tr_f.shape[1] if X_tr_f.ndim == 2 else 0

                if X_tr_f.shape[0] < 50 or X_te_f.shape[0] < 50:
                    debug_print("       => skipping: not enough final windows.")
                    continue

                # Build model
                if dec_name == 'GRU':
                    model = GRUDecoder(input_size=input_dim, hidden_size=16).to(DEVICE)
                elif dec_name == 'LSTM':
                    model = LSTMDecoder(input_size=input_dim, hidden_size=16).to(DEVICE)
                elif dec_name == 'LiGRU':
                    model = LiGRUDecoder(input_size=input_dim, hidden_size=16).to(DEVICE)
                else:
                    model = LinearLagDecoder(input_dim=input_dim, hidden_dim=64).to(DEVICE)

                train_model(model, X_tr_f, Y_tr_f, num_epochs=300, batch_size=64, lr=0.001)
                preds = eval_model(model, X_te_f)
                vaf_val = compute_vaf(Y_te_f, preds)
                mse_val = np.mean((Y_te_f - preds)**2) if len(preds) else np.nan

                res_d = {
                    'train_monkey': M_train,
                    'train_date':   D_train,
                    'train_task':   T_train,
                    'test_monkey':  M_test,
                    'test_date':    D_test,
                    'test_task':    T_test,
                    'decoder_type': dec_name,
                    'alignment_mode': mode,
                    'VAF':          vaf_val,
                    'MSE':          mse_val,
                    'train_size':   len(X_train_red),
                    'test_size':    len(X_test_red),
                    'timestamp':    datetime.datetime.now()
                }
                results_list.append(res_d)

    df_results = pd.DataFrame(results_list)
    debug_print("[DEBUG] run_experiments: done\n")
    return df_results

###############################################################################
#                           I) MAIN
###############################################################################
def main():
    debug_print("[DEBUG] === Starting main() ===")
    # 1) Load your big DataFrame
    df_path = "output.pkl"  # adapt path
    debug_print(f"[DEBUG] Loading DataFrame from {df_path} ...")
    df_multi = pd.read_pickle(df_path)
    debug_print(f"[DEBUG] df_multi shape={df_multi.shape}, columns={df_multi.columns.tolist()}")

    # 2) Unify spike headers
    df_multi_aligned, channel_list = unify_spike_headers(df_multi, spike_col="spike_counts", verbose=True)
    debug_print(f"[DEBUG] => unify_spike_headers => final channel_list length={len(channel_list)}")

    # 3) Identify monkeys
    monkeys = df_multi_aligned['monkey'].unique()
    debug_print(f"[DEBUG] Found {len(monkeys)} monkeys => {monkeys}")

    # 4) Prepare day-level PCA
    debug_print("[DEBUG] prepare_local_pca_daylevel => running")
    pca_day = prepare_local_pca_daylevel(df_multi_aligned, monkeys, n_components=20)

    # 5) Prepare monkey-level PCA
    debug_print("[DEBUG] prepare_monkey_level_global_pca => running")
    pca_monk= prepare_monkey_level_global_pca(df_multi_aligned, monkeys, n_components=20)

    # 6) Merge PCA objects
    pca_objs = {}
    pca_objs.update(pca_day)    # has 'day_level' + 'ref'
    pca_objs.update(pca_monk)   # has 'monkey_global'
    debug_print("[DEBUG] Merged PCA objects => keys:", pca_objs.keys())

    # 7) Define alignment modes & decoders
    alignment_modes = ['none','day_level','monkey_level']
    decoders = ['GRU','LSTM','LiGRU','Linear']

    # 8) Run experiments
    df_results = run_experiments(df_multi_aligned, alignment_modes, decoders, pca_objs, n_components=20)

    # 9) Save results
    out_results = "my_alignment_results_debug.pkl"
    df_results.to_pickle(out_results)
    debug_print(f"[DEBUG] Saved results => {out_results}. shape={df_results.shape}")

    # 10) Example: Print head
    print("=== RESULTS SAMPLE ===")
    print(df_results.head())

if __name__ == "__main__":
    main()
