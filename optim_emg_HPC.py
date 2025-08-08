#!/usr/bin/env python3
import argparse, gc, os, pickle, warnings, itertools
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast
import pandas as pd

# ------------------- project imports (mets à jour si besoin) -------------------
from Day_stability_EMG_CV import (
    set_seed, build_dayX_decoder_data,
    GRUDecoder, LSTMDecoder, LinearLagDecoder, LiGRUDecoder,
    evaluate_on_split,
    DEVICE, COMBINED_PICKLE_FILE
)

BATCH_SIZE = 512
EARLY_STOP_PATIENCE = None

GRID: Dict[str, Dict[str, List[Any]]] = {
    "gru": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "lstm": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "ligru": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "linear": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [32, 64, 128, 192, 256],
        num_epochs = [50, 100, 150],
        lr         = [1e-3, 1e-2],
    ),
}

combined_df = pd.read_pickle(COMBINED_PICKLE_FILE)
if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
    combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y/%m/%d")
day0        = sorted(combined_df["date"].unique())[0]
train_df    = combined_df[combined_df["date"] == day0].reset_index(drop=True)
n_emg_channels = next(emg.shape[1] for emg in combined_df["EMG"]
                      if isinstance(emg, (pd.DataFrame, np.ndarray)))

def make_dataset(n_pca: int, k_lag: int, is_linear: bool):
    X, Y = build_dayX_decoder_data(
        train_df, day_pca_model=None,
        n_pca=n_pca, seq_len=k_lag, is_linear=is_linear
    )
    return X, Y

def run_kfold_training(decoder: str, cfg: Dict[str, Any],
                       n_folds: int, seed: int
)-> tuple[float, int, list[float], list[float]]:
    set_seed(seed)
    is_linear = decoder == "linear"
    X_full, Y_full = make_dataset(cfg["n_pca"], cfg["k_lag"], is_linear)
    X_full = torch.as_tensor(X_full, dtype=torch.float32, device=DEVICE)
    Y_full = torch.as_tensor(Y_full, dtype=torch.float32, device=DEVICE)
    if X_full.numel() == 0:
        raise RuntimeError("Empty dataset for config", cfg)

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_full))
    folds = np.array_split(idx, n_folds)

    vafs, fold_times = [], []
    param_count = None
    for i_split in range(n_folds):
        val_idx = folds[i_split]
        tr_idx  = np.hstack([folds[j] for j in range(n_folds) if j != i_split])
        X_tr, Y_tr = X_full[tr_idx],  Y_full[tr_idx]
        X_val, Y_val = X_full[val_idx], Y_full[val_idx]

        if decoder == "gru":
            model = GRUDecoder(cfg["n_pca"], cfg["hidden_dim"], n_emg_channels).to(DEVICE)
        elif decoder == "lstm":
            model = LSTMDecoder(cfg["n_pca"], cfg["hidden_dim"], n_emg_channels).to(DEVICE)
        elif decoder == "ligru":
            model = LiGRUDecoder(cfg["n_pca"], cfg["hidden_dim"], n_emg_channels).to(DEVICE)
        else:
            inp   = cfg["k_lag"] * cfg["n_pca"]
            model = LinearLagDecoder(inp, cfg["hidden_dim"], n_emg_channels).to(DEVICE)

        if param_count is None:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        opt  = optim.Adam(model.parameters(), lr=cfg["lr"])
        crit = nn.MSELoss()
        dl_tr = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=False
        )
        scaler = GradScaler()
        best_loss, bad_epochs = np.inf, 0
        start = time.perf_counter()
        for ep in range(cfg["num_epochs"]):
            model.train()
            for xb, yb in dl_tr:
                opt.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

            if EARLY_STOP_PATIENCE is not None:
                if loss.item() + 1e-6 < best_loss:
                    best_loss, bad_epochs = loss.item(), 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= EARLY_STOP_PATIENCE:
                        break
        dur = time.perf_counter() - start
        fold_times.append(dur)
        vaf = evaluate_on_split(model,
                                X_val.detach().cpu().numpy(),
                                Y_val.detach().cpu().numpy(),
                                seq_len=cfg["k_lag"],
                                is_linear=is_linear)
        vafs.append(vaf if not np.isnan(vaf) else -1.0)
        del model; torch.cuda.empty_cache(); gc.collect()

    return float(np.mean(vafs)), param_count, vafs, fold_times

def cartesian_product(param_dict: Dict[str, List[Any]]):
    keys, vals = zip(*param_dict.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--decoders", nargs="+", default=["gru", "lstm", "ligru", "linear"])
    p.add_argument("--seeds",    type=int, default=5)
    p.add_argument("--folds",    type=int, default=10)
    p.add_argument("--progress", type=int, default=50,
                   help="print a heartbeat every N runs")
    # CLUSTER PARTITIONNING:
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num_per_job", type=int, default=1)
    args = p.parse_args()

    # Générer toutes les combinaisons
    combos = []
    for dec in args.decoders:
        grid = list(cartesian_product(GRID[dec]))
        for cfg in grid:
            for seed in range(args.seeds):
                combos.append((dec, cfg, seed))

    start = args.start_idx
    end   = min(len(combos), start + args.num_per_job)
    my_combos = combos[start:end]
    print(f"Ce job traite {start} à {end-1} sur {len(combos)} combos.")

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    out_path = Path(f"gridsearch_results_{job_id}.pkl")
    if out_path.exists():
        results: List[Dict[str, Any]] = pickle.load(open(out_path, "rb"))
    else:
        results = []

    done_keys = {(r["decoder"], r["n_pca"], r["k_lag"],
                  r["hidden_dim"], r["num_epochs"], r["lr"], r["seed"])
                 for r in results}
    total = 0
    for dec, cfg, seed in my_combos:
        key = (dec, cfg["n_pca"], cfg["k_lag"],
               cfg["hidden_dim"], cfg["num_epochs"], cfg["lr"], seed)
        if key in done_keys:
            continue
        total += 1
        if total % args.progress == 0:
            print(f"  …{total} runs done in this job")

        try:
            mean_vaf, npar, folds_vaf, folds_sec = run_kfold_training(
                dec, cfg, args.folds, seed)
            results.append(dict(decoder=dec, seed=seed,
                        num_params=npar,
                        mean_vaf=mean_vaf,
                        fold_vafs=folds_vaf,
                        fold_times=folds_sec,
                        mean_time=float(np.mean(folds_sec)),
                        **cfg))
            done_keys.add(key)
            pickle.dump(results, open(out_path, "wb"))
        except RuntimeError as e:
            warnings.warn(f"{key} failed: {e}")
            continue

    print(f"\nJob terminé. {len(my_combos)} combos traités, sauvegardés dans {out_path.resolve()}")

if __name__ == "__main__":
    main()
