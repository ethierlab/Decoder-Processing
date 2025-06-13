import argparse, gc, warnings, os, multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast
import optuna

# ───────────────────────────────────────────────────────────────
# Imports from your big helper script
# ───────────────────────────────────────────────────────────────
from Day_stability_EMG_CV import (
    set_seed, build_dayX_decoder_data,
    GRUDecoder, LSTMDecoder, LinearLagDecoder, LiGRUDecoder,
    train_decoder, evaluate_on_split,
    DEVICE, COMBINED_PICKLE_FILE
)

# ──────────────────────────── GLOBALS ──────────────────────────
BATCH_SIZE      = 1024            
MAX_EPOCHS      = 400
N_SPLITS        = 10
EARLY_PRUNE_EPS = (25, 50, 100)
RAW_CACHE = None        
DTYPE_TORCH     = torch.float16
scaler          = GradScaler("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

SEARCH = {
    "gru":    dict(hidden_dim=(8,128)),
    "lstm":   dict(hidden_dim=(8,128)),
    "ligru":  dict(hidden_dim=(8,128)),
    "linear": dict(hidden_dim=(32,256)),
}

# ───────────────────────── LOAD DAY-0 DATA ─────────────────────
combined_df = pd.read_pickle(COMBINED_PICKLE_FILE)
if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
    combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y/%m/%d")

day0   = sorted(combined_df["date"].unique())[0]
train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

n_emg_channels = next(
    emg.shape[1] for emg in combined_df["EMG"]
    if isinstance(emg, (pd.DataFrame, np.ndarray))
)

# ───────────────────────── dataset helper ──────────────────────

def get_raw_latents():
    global RAW_CACHE
    if RAW_CACHE is None:
        X_raw, Y_raw = build_dayX_decoder_data(
            train_df, None, 32, 1, False)
        RAW_CACHE = (
            X_raw.astype(np.float32, copy=False),
            Y_raw.astype(np.float32, copy=False),
        )
    return RAW_CACHE

def make_trial_dataset(n_pca, k_lag, is_linear):
    X_raw, Y_raw = get_raw_latents()
    X_feat = X_raw[:, :n_pca]

    if is_linear:
        X_out = np.stack([
            X_feat[t-k_lag:t].reshape(-1)
            for t in range(k_lag, len(X_feat))
        ], axis=0)
    else:
        X_out = np.stack([
            X_feat[t-k_lag:t]
            for t in range(k_lag, len(X_feat))
        ], axis=0)

    Y_out = Y_raw[k_lag:]
    return X_out, Y_out

# ───────────────────── build Optuna objective ──────────────────
def build_objective(decoder: str):
    hid_min, hid_max = SEARCH[decoder]["hidden_dim"]

    def objective(trial: optuna.Trial) -> float:
        n_pca      = trial.suggest_int("n_pca", 8, 32)
        k_lag      = trial.suggest_int("k_lag", 5, 25)
        hidden_dim = trial.suggest_int("hidden_dim", hid_min, hid_max)
        num_epochs = trial.suggest_int("num_epochs", 50, MAX_EPOCHS, step=50)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # full day-0 tensors (once per trial)
        is_lin = decoder == "linear"
        X_full, Y_full = make_trial_dataset(n_pca, k_lag, is_lin)
        if X_full.size == 0:
            raise optuna.TrialPruned("empty dataset")
        X_full = X_full.astype(np.float32, copy=False)
        Y_full = Y_full.astype(np.float32, copy=False)

        X_full_t = torch.from_numpy(X_full)          # float32 CPU tensor
        Y_full_t = torch.from_numpy(Y_full)

        # del X_full, Y_full
        gc.collect()
        rng   = np.random.default_rng(trial.number)
        idx   = rng.permutation(len(X_full))
        folds = np.array_split(idx, N_SPLITS)

        vafs, param_cnt = [], None

        for i_split in range(N_SPLITS):
            val_idx = folds[i_split]
            tr_idx  = np.hstack([folds[j] for j in range(N_SPLITS) if j != i_split])

            X_tr, Y_tr = X_full_t[tr_idx], Y_full_t[tr_idx]
            X_val, Y_val = X_full[val_idx], Y_full[val_idx]

            # new model
            set_seed(42 + trial.number + i_split)
            if decoder == "gru":
                model = GRUDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
            elif decoder == "lstm":
                model = LSTMDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
            elif decoder == "ligru":
                model = LiGRUDecoder(n_pca, hidden_dim, n_emg_channels).to(DEVICE)
            else:
                model = LinearLagDecoder(k_lag * n_pca, hidden_dim,
                                         n_emg_channels).to(DEVICE)

            if param_cnt is None:
                param_cnt = sum(p.numel() for p in model.parameters()
                                if p.requires_grad)

            opt  = optim.Adam(model.parameters(), lr=lr)
            crit = nn.MSELoss()
            dl_tr = DataLoader(
                TensorDataset(X_tr, Y_tr),     # still CPU tensors
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=8,
                pin_memory=True,               # safe again
                persistent_workers=True, prefetch_factor=4
            )

            # training
            for ep in range(num_epochs):
                model.train()
                for Xb, Yb in dl_tr:
                    Xb = Xb.to(DEVICE, non_blocking=True)
                    Yb = Yb.to(DEVICE, non_blocking=True)
                    with autocast("cuda"):
                        loss = crit(model(Xb), Yb)

                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()

                trial.report(float(loss), step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # validation
            with torch.no_grad():
                vaf = evaluate_on_split(model, X_val, Y_val,
                                        seq_len=k_lag, is_linear=is_lin)
            vafs.append(vaf if not np.isnan(vaf) else -1.0)

            del model; torch.cuda.empty_cache(); gc.collect()

        trial.set_user_attr("param_count", param_cnt)
        trial.set_user_attr("fold_vafs",   vafs)
        trial.set_user_attr("std_vaf",     float(np.std(vafs)))

        return float(np.mean(vafs))

    return objective

# ───────────────────── optimise one decoder ────────────────────
def optimise_decoder(decoder: str, n_trials: int):
    study = optuna.create_study(
        study_name=f"{decoder}_hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner =optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(build_objective(decoder),
                   n_trials=n_trials, n_jobs=1,   # single-process, no .db
                   show_progress_bar=True)

    print(f"\n[{decoder.upper()}] BEST VAF = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    study.trials_dataframe().to_pickle(f"{decoder}_hpo.pkl")
    print(f"Saved trials → {decoder}_hpo.pkl\n")

# ─────────────────────────────────── CLI ───────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150,
                        help="number of trials per decoder")
    parser.add_argument("--decoders", nargs="*",
                        default=["gru", "lstm", "linear", "ligru"])
    args = parser.parse_args()

    if mp.current_process().name == "MainProcess":
        print(f"[INFO] device: {DEVICE}  |  AMP: float16 | "
              f"folds: {N_SPLITS} | batch: {BATCH_SIZE}")

    for dec in args.decoders:
        print(f"\n=========== {dec.upper()} ({args.trials} trials) ===========")
        optimise_decoder(dec, args.trials)
