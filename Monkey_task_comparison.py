import numpy as np
import pandas as pd
import datetime
import random
import torch
import sys
from numpy.linalg import pinv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

###############################################################################
# Global seeds & device
###############################################################################
SEED = 18
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_VERBOSE = True
def debug_print(*msg):
    if GLOBAL_VERBOSE:
        print(*msg)

###############################################################################
# Global params
###############################################################################
BIN_FACTOR = 10
BIN_SIZE   = 0.001
SMOOTHING_LENGTH = 0.05

GRU_N_PCA    = 14
GRU_HIDDEN_DIM= 5
GRU_K_LAG     = 16

LSTM_N_PCA   = 14
LSTM_HIDDEN_DIM=16
LSTM_K_LAG   = 16

LINEAR_N_PCA = 18
LINEAR_HIDDEN_DIM=64
LINEAR_K_LAG = 16

LIGRU_N_PCA  = 14
LIGRU_HIDDEN_DIM=5
LIGRU_K_LAG  = 16

NUM_EPOCHS   = 1
BATCH_SIZE   = 64
LEARNING_RATE= 0.001

###############################################################################
# unify_spike_headers
###############################################################################
def unify_spike_headers(df, spike_col="spike_counts", verbose=True):
    if verbose:
        debug_print("[DEBUG] unify_spike_headers: Start")

    all_neurons = set()
    for idx,row in df.iterrows():
        sc = row.get(spike_col)
        if isinstance(sc, pd.DataFrame) and not sc.empty:
            all_neurons.update(sc.columns)
    all_neuron_list= sorted(all_neurons)

    df_aligned= df.copy()
    for idx,row in df_aligned.iterrows():
        sc=row.get(spike_col)
        if not isinstance(sc,pd.DataFrame) or sc.empty:
            df_aligned.at[idx, spike_col]= pd.DataFrame(0,index=[],columns=all_neuron_list)
        else:
            sc2= sc.reindex(columns=all_neuron_list, fill_value=0)
            df_aligned.at[idx, spike_col]= sc2
    return df_aligned, all_neuron_list

###############################################################################
# build_continuous_dataset
###############################################################################
def build_continuous_dataset(df_subset,
                             bin_factor=BIN_FACTOR,
                             bin_size=BIN_SIZE,
                             smoothing_length=SMOOTHING_LENGTH):
    debug_print(f"[DEBUG] build_continuous_dataset: #rows={len(df_subset)}")
    all_spike_list=[]
    all_force_list=[]
    for idx,row in df_subset.iterrows():
        spk_df=row.get("spike_counts")
        frc_df=row.get("force")
        if not isinstance(spk_df,pd.DataFrame) or spk_df.empty:
            continue
        if frc_df is None or len(frc_df)==0:
            continue
        spk_arr= spk_df.values
        frc_arr= frc_df.values if hasattr(frc_df,'values') else np.array(frc_df)
        force_x= frc_arr[:,0]
        all_spike_list.append(spk_arr)
        all_force_list.append(force_x)
    if not all_spike_list:
        return np.empty((0,)), np.empty((0,))
    X_all= np.concatenate(all_spike_list, axis=0)
    Y_all= np.concatenate(all_force_list, axis=0)
    debug_print(f"[DEBUG] build_continuous_dataset => X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all

def create_rnn_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0,))
    X_out, Y_out=[], []
    for i in range(seq_len, X_arr.shape[0]):
        window=X_arr[i-seq_len:i,:]
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out,dtype=np.float32), np.array(Y_out,dtype=np.float32)

def create_linear_dataset_continuous(X_arr, Y_arr, seq_len):
    if X_arr.shape[0] <= seq_len:
        return np.empty((0, seq_len*X_arr.shape[1])), np.empty((0,))
    X_out, Y_out=[], []
    for i in range(seq_len, X_arr.shape[0]):
        window= X_arr[i-seq_len:i,:].reshape(-1)
        X_out.append(window)
        Y_out.append(Y_arr[i])
    return np.array(X_out,dtype=np.float32), np.array(Y_out,dtype=np.float32)

###############################################################################
# Models
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru=nn.GRU(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_=self.gru(x)
        out= out[:,-1,:]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)
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
        h_candidate=torch.relu(self.x2h(x)+self.h2h(h))
        h_next=(1-z)*h+z*h_candidate
        return h_next

class LiGRUDecoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.cell=LiGRUCell(input_size,hidden_size)
        self.fc=nn.Linear(hidden_size,1)
    def forward(self,x):
        batch_size=x.shape[0]
        h=torch.zeros(batch_size,self.hidden_size,device=x.device)
        for t in range(x.shape[1]):
            h=self.cell(x[:,t,:],h)
        return self.fc(h)

class LinearLagDecoder(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.lin1=nn.Linear(input_dim,hidden_dim)
        self.act=nn.ReLU()
        self.lin2=nn.Linear(hidden_dim,1)
    def forward(self,x):
        x=self.lin1(x)
        x=self.act(x)
        return self.lin2(x)

def train_model(model,X_train,Y_train,num_epochs,batch_size,lr):
    ds=TensorDataset(torch.tensor(X_train), torch.tensor(Y_train).unsqueeze(-1))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True)
    opt=optim.Adam(model.parameters(),lr=lr)
    crit=nn.MSELoss()
    for ep in range(num_epochs):
        model.train()
        total_loss=0
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        if ep%10==0 and GLOBAL_VERBOSE:
            # print(len(dl))
            debug_print(f"[DEBUG-train] Epoch={ep}/{num_epochs}, avg_loss={total_loss/len(dl):.4f}")

def eval_model(model,X_test,batch_size=64):
    preds=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(X_test),batch_size):
            xb=torch.tensor(X_test[i:i+batch_size]).float().to(DEVICE)
            out=model(xb)
            preds.append(out.cpu().numpy().flatten())
    if preds:
        return np.concatenate(preds)
    else:
        return np.array([])

def compute_vaf(y_true,y_pred):
    var_true=np.var(y_true,ddof=1)
    var_resid=np.var(y_true-y_pred,ddof=1)
    if var_true<1e-12:
        return np.nan
    return 1-(var_resid/var_true)

###############################################################################
# PCA realignment + monkey_level
###############################################################################
def gather_day_level_spikes(df_subset):
    X_all,_=build_continuous_dataset(df_subset,BIN_FACTOR,BIN_SIZE,SMOOTHING_LENGTH)
    return X_all

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

def prepare_monkey_level_global_pca(df_multi,monkeys,n_components=20):
    pca_objs={
        'monkey_global':{}
    }
    for M in monkeys:
        df_m=df_multi[df_multi['monkey']==M]
        X_all,_= build_continuous_dataset(df_m,BIN_FACTOR,BIN_SIZE,SMOOTHING_LENGTH)
        if X_all.shape[0]<2: continue
        pca_m=PCA(n_components=n_components,random_state=SEED)
        pca_m.fit(X_all)
        pca_objs['monkey_global'][M]=pca_m
    return pca_objs

def compute_alignment_matrix(V_dayD,V_day0):
    return pinv(V_dayD)@V_day0

def apply_alignment_mode(x_2d, monkey_current, day_current,
                         alignment_mode, pca_objs,
                         n_components=20,
                         monkey_train=None):
    if alignment_mode=='none':
        return x_2d
    if alignment_mode=='monkey_level':
        mg_pca= pca_objs['monkey_global'].get(monkey_current, None)
        if mg_pca is None:
            debug_print("[WARNING] no monkey_global => skip => raw dimension")
            return x_2d
        x_tr= mg_pca.transform(x_2d)
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
        return x_al
    debug_print("[WARNING] unknown alignment => skip => raw dimension")
    return x_2d

###############################################################################
# SCENARIOS
###############################################################################
def get_decoder_params(dec_name):
    if dec_name=='GRU':
        return (GRU_N_PCA, GRU_HIDDEN_DIM, GRU_K_LAG)
    elif dec_name=='LSTM':
        return (LSTM_N_PCA, LSTM_HIDDEN_DIM, LSTM_K_LAG)
    elif dec_name=='LiGRU':
        return (LIGRU_N_PCA, LIGRU_HIDDEN_DIM, LIGRU_K_LAG)
    else:
        return (LINEAR_N_PCA, LINEAR_HIDDEN_DIM, LINEAR_K_LAG)

def build_decoder_model(dec_name,input_dim,hidden_dim):
    if dec_name=='GRU':
        return GRUDecoder(input_dim,hidden_dim).to(DEVICE)
    elif dec_name=='LSTM':
        return LSTMDecoder(input_dim,hidden_dim).to(DEVICE)
    elif dec_name=='LiGRU':
        return LiGRUDecoder(input_dim,hidden_dim).to(DEVICE)
    else:
        return LinearLagDecoder(input_dim,hidden_dim).to(DEVICE)

def build_scenarios():
    """
    Example scenario list:
      - Some have 'split_ratio'=0.25 => 75/25 internal
      - Others omit it => train on 100%
    For each scenario, 'tests' => additional test sets
    """
    
    def filter_jango_iso(row):
        return (row['monkey']=='Jango') and (row['task']=='iso')
    def filter_jango_wm(row):
        return (row['monkey']=='Jango') and (row['task']=='wm')
    def filter_jango_spr(row):
        return (row['monkey']=='Jango') and (row['task']=='spr')

    def filter_jacb_iso(row):
        return (row['monkey']=='JacB') and (row['task']=='iso')
    def filter_jacb_wm(row):
        return (row['monkey']=='JacB') and (row['task']=='wm')
    def filter_jacb_spr(row):
        return (row['monkey']=='JacB') and (row['task']=='spr')

    SCENARIOS = [

        # -----------------
        # TASK COMPARISON (Jango)
        # -----------------
        {
            'name': 'TaskComp_Jango_iso',
            'train_filter': filter_jango_iso,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_spr, 'name':'jango_spr' },
            { 'test_filter': filter_jango_wm,  'name':'jango_wm' }
            ]
        },
        {
            'name': 'TaskComp_Jango_spr',
            'train_filter': filter_jango_spr,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_iso, 'name':'jango_iso' },
            { 'test_filter': filter_jango_wm,  'name':'jango_wm' }
            ]
        },
        {
            'name': 'TaskComp_Jango_wm',
            'train_filter': filter_jango_wm,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_spr, 'name':'jango_spr' },
            { 'test_filter': filter_jango_iso, 'name':'jango_iso' }
            ]
        },

        # -----------------
        # TASK COMPARISON (JacB)
        # -----------------
        {
            'name': 'TaskComp_JacB_iso',
            'train_filter': filter_jacb_iso,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_spr, 'name':'jacB_spr' },
            { 'test_filter': filter_jacb_wm,  'name':'jacB_wm' }
            ]
        },
        {
            'name': 'TaskComp_JacB_spr',
            'train_filter': filter_jacb_spr,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_iso, 'name':'jacB_iso' },
            { 'test_filter': filter_jacb_wm,  'name':'jacB_wm' }
            ]
        },
        {
            'name': 'TaskComp_JacB_wm',
            'train_filter': filter_jacb_wm,
            'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_spr, 'name':'jacB_spr' },
            { 'test_filter': filter_jacb_iso, 'name':'jacB_iso' }
            ]
        },

        # -----------------
        # MONKEY COMPARISON (Jango -> JacB)
        # -----------------
        {
            'name': 'MonkeyComp_iso_Jango2JacB',
            'train_filter': filter_jango_iso,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_iso, 'name': 'jacB_iso' }
            ]
        },
        {
            'name': 'MonkeyComp_wm_Jango2JacB',
            'train_filter': filter_jango_wm,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_wm, 'name': 'jacB_wm' }
            ]
        },
        {
            'name': 'MonkeyComp_spr_Jango2JacB',
            'train_filter': filter_jango_spr,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jacb_spr, 'name': 'jacB_spr' }
            ]
        },

        # -----------------
        # MONKEY COMPARISON (JacB -> Jango)
        # -----------------
        {
            'name': 'MonkeyComp_iso_JacB2Jango',
            'train_filter': filter_jacb_iso,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_iso, 'name': 'jango_iso' }
            ]
        },
        {
            'name': 'MonkeyComp_wm_JacB2Jango',
            'train_filter': filter_jacb_wm,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_wm, 'name': 'jango_wm' }
            ]
        },
        {
            'name': 'MonkeyComp_spr_JacB2Jango',
            'train_filter': filter_jacb_spr,
            # 'split_ratio': 0.25,
            'tests': [
            { 'test_filter': filter_jango_spr, 'name': 'jango_spr' }
            ]
        }
    ]

    return SCENARIOS

###############################################################################
# run_experiments => handle split_ratio => val => plus test sets
###############################################################################
def run_experiments(df_multi, alignment_modes, pca_objs):
    decoders=['GRU','LSTM','LiGRU','Linear']
    scenarios= build_scenarios()

    results_list=[]
    for sc in scenarios:
        scenario_name= sc['name']
        debug_print(f"\n=== SCENARIO: {scenario_name} ===")

        df_train= df_multi[df_multi.apply(sc['train_filter'], axis=1)]
        if df_train.empty:
            debug_print(f"[WARNING] scenario={scenario_name} => empty train => skip")
            continue

        X_train_raw, Y_train_raw= build_continuous_dataset(df_train)
        if X_train_raw.shape[0]<10:
            debug_print("[WARNING] not enough train data => skip scenario")
            continue
        # plt.plot(Y_train_raw[:1000])
        # plt.title("First 1000 samples of force (Train)")
        # plt.show()
        train_first= df_train.iloc[0]
        M_train= train_first['monkey']
        D_train= train_first['date']

        # internal split
        split_ratio= sc.get('split_ratio', None)
        if split_ratio is not None:
            X_tr_raw, X_val_raw, Y_tr_raw, Y_val_raw= train_test_split(
                X_train_raw, Y_train_raw, 
                test_size=split_ratio, 
                random_state=SEED
            )
            debug_print(f"[DEBUG-split] train size={X_tr_raw.shape}, val size={X_val_raw.shape}")
        else:
            X_tr_raw, Y_tr_raw= X_train_raw, Y_train_raw
            X_val_raw, Y_val_raw= np.empty((0,)), np.empty((0,))

        for mode in alignment_modes:
            for dec_name in decoders:
                n_comp, hidden_dim, k_lag= get_decoder_params(dec_name)

                # align train
                X_tr_aligned= apply_alignment_mode(X_tr_raw, M_train, D_train, mode, pca_objs, n_comp, monkey_train=M_train)
                # window
                if dec_name in ['GRU','LSTM','LiGRU']:
                    X_tr_f, Y_tr_f= create_rnn_dataset_continuous(X_tr_aligned,Y_tr_raw,k_lag)
                    # print("Windowed train dataset shape:", X_tr_f.shape, Y_tr_f.shape)
                    # print("Train force stats:", Y_tr_raw.min(), Y_tr_raw.max(), np.var(Y_tr_raw))
                    # print("Val force stats:",   Y_val_raw.min(), Y_val_raw.max(), np.var(Y_val_raw))
                    
                    if X_tr_f.shape[0]<50: 
                        continue
                    input_dim= X_tr_f.shape[2]
                else:
                    X_tr_f, Y_tr_f= create_linear_dataset_continuous(X_tr_aligned,Y_tr_raw,k_lag)
                    if X_tr_f.shape[0]<50: 
                        continue
                    input_dim= X_tr_f.shape[1]

                # build & train
                model= build_decoder_model(dec_name,input_dim,hidden_dim)
                train_model(model,X_tr_f,Y_tr_f,NUM_EPOCHS,BATCH_SIZE,LEARNING_RATE)

                # (1) Evaluate the 25% val if split_ratio is present
                if split_ratio is not None and X_val_raw.shape[0]>0:
                    X_val_aligned= apply_alignment_mode(X_val_raw, M_train, D_train, mode, pca_objs, n_comp, monkey_train=M_train)
                    if dec_name in ['GRU','LSTM','LiGRU']:
                        X_val_f, Y_val_f= create_rnn_dataset_continuous(X_val_aligned, Y_val_raw, k_lag)
                    else:
                        X_val_f, Y_val_f= create_linear_dataset_continuous(X_val_aligned, Y_val_raw, k_lag)

                    if X_val_f.shape[0]>=50:
                        preds= eval_model(model,X_val_f)
                        vaf_val= compute_vaf(Y_val_f,preds)
                        mse_val= np.mean((Y_val_f-preds)**2) if len(preds) else np.nan

                        # new line => printing right after computation
                        debug_print(f"[DEBUG-val] scenario={scenario_name}, mode={mode}, dec={dec_name}, val VAF={vaf_val:.3f}, MSE={mse_val:.3f}")

                        res_d={
                            'scenario_name': scenario_name,
                            'decoder_type': dec_name,
                            'alignment_mode': mode,
                            'train_monkey': M_train,
                            'test_monkey': M_train,  # same monkey => diagonal
                            'train_size': X_tr_raw.shape[0],
                            'test_size':  X_val_raw.shape[0], # val size
                            'test_name': 'val_25pct',
                            'VAF': vaf_val,
                            'MSE': mse_val,
                            'timestamp': datetime.datetime.now()
                        }
                        results_list.append(res_d)

                # (2) Evaluate each test in sc['tests']
                test_defs= sc.get('tests', [])
                for tdef in test_defs:
                    df_test= df_multi[df_multi.apply(tdef['test_filter'], axis=1)]
                    if df_test.empty:
                        continue
                    X_test_raw,Y_test_raw= build_continuous_dataset(df_test)
                    if X_test_raw.shape[0]<10: 
                        continue

                    test_first= df_test.iloc[0]
                    M_test= test_first['monkey']
                    D_test= test_first['date']

                    X_test_aligned= apply_alignment_mode(X_test_raw, M_test, D_test, mode, pca_objs, n_comp, monkey_train=M_train)
                    if dec_name in ['GRU','LSTM','LiGRU']:
                        X_te_f, Y_te_f= create_rnn_dataset_continuous(X_test_aligned,Y_test_raw,k_lag)
                    else:
                        X_te_f, Y_te_f= create_linear_dataset_continuous(X_test_aligned,Y_test_raw,k_lag)
                    if X_te_f.shape[0]<50:
                        continue

                    preds= eval_model(model,X_te_f)
                    vaf_val= compute_vaf(Y_te_f,preds)
                    mse_val= np.mean((Y_te_f-preds)**2) if len(preds) else np.nan

                    # Print test results too
                    debug_print(f"[DEBUG-test] scenario={scenario_name}, mode={mode}, dec={dec_name}, test={tdef['name']}, VAF={vaf_val:.3f}, MSE={mse_val:.3f}")

                    res_d={
                        'scenario_name': scenario_name,
                        'decoder_type': dec_name,
                        'alignment_mode': mode,
                        'train_monkey': M_train,
                        'test_monkey': M_test,
                        'train_size': X_tr_raw.shape[0],
                        'test_size':  X_test_raw.shape[0],
                        'test_name':  tdef['name'],
                        'VAF': vaf_val,
                        'MSE': mse_val,
                        'timestamp': datetime.datetime.now()
                    }
                    results_list.append(res_d)

    return pd.DataFrame(results_list)

###############################################################################
# MAIN
###############################################################################
def main():
    debug_print("[DEBUG] === Starting main() ===")

    # 1) Load your DataFrame
    df_path="output.pkl"
    debug_print("[DEBUG] Loading DataFrame from", df_path)
    df_multi=pd.read_pickle(df_path)
    debug_print(f"[DEBUG] df_multi shape={df_multi.shape}")

    # 2) unify_spike_headers
    df_multi_aligned, channel_list= unify_spike_headers(df_multi, "spike_counts", verbose=True)
    monkeys= df_multi_aligned['monkey'].unique()
    debug_print("[DEBUG] monkeys =>",monkeys)

    # 3) build PCA objects
    max_pca= max(GRU_N_PCA, LSTM_N_PCA, LINEAR_N_PCA, LIGRU_N_PCA)
    pca_real= prepare_realignment_pca(df_multi_aligned, monkeys, n_components=max_pca)
    pca_monk= prepare_monkey_level_global_pca(df_multi_aligned, monkeys, n_components=max_pca)
    pca_objs={}
    pca_objs.update(pca_real)
    pca_objs.update(pca_monk)

    alignment_modes=['none','realignment','monkey_level']

    # 4) run
    df_results= run_experiments(df_multi_aligned, alignment_modes, pca_objs)
    out_file="train_val_3squares_results.pkl"
    df_results.to_pickle(out_file)
    debug_print("[DEBUG] saved =>",out_file)
    print("=== RESULTS SAMPLE ===")
    print(df_results.head())

if __name__=="__main__":
    main()
