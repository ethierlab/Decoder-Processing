#!/usr/bin/env python3
# day0_from_gridsearch_stats.py
# Build Day-0 units from ALL_gridsearch_results.pkl, then paired Wilcoxon + Holm, + violin.

import os, json, pickle, argparse
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

DECODER_COLORS = {"gru":"#d62728", "lstm":"#1f77b4", "ligru":"#ff7f0e", "linear":"#2ca02c"}

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m, float); prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]; val = max(val, prev)
        adj[idx] = min(val, 1.0); prev = adj[idx]
    return adj

def mean_from_folds(v):
    try:
        arr = np.array(v, dtype=float); arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan
    except Exception: return np.nan

def load_grid_pkl(pkl_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = pickle.load(open(pkl_path, "rb"))
    df = pd.DataFrame(rows)
    # expected / sanitize
    for c in ["decoder","hidden_dim","k_lag","n_pca","num_epochs","lr","seed","fold_vafs","mean_vaf"]:
        if c not in df.columns: df[c] = np.nan
    for c in ["hidden_dim","k_lag","n_pca","num_epochs","lr","seed","mean_vaf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # recompute mean_vaf from folds when possible
    recalc = df["fold_vafs"].apply(mean_from_folds)
    df.loc[recalc.notna(), "mean_vaf"] = recalc[recalc.notna()]
    df = df.dropna(subset=["decoder","hidden_dim","k_lag","n_pca","num_epochs","lr","mean_vaf"])
    df["decoder"] = df["decoder"].str.lower()
    return df

def select_best_per_decoder(df: pd.DataFrame) -> pd.DataFrame:
    # average over folds×seeds (score agrégé) par configuration (sans seed dans la clé)
    key = ["decoder","hidden_dim","k_lag","n_pca","num_epochs","lr"]
    flat = []
    for _, r in df.iterrows():
        vafs = r["fold_vafs"] if isinstance(r["fold_vafs"], (list, tuple, np.ndarray)) else []
        for v in vafs:
            try: v = float(v)
            except: continue
            if np.isfinite(v):
                flat.append({k: r[k] for k in key} | {"vaf": v})
    flatdf = pd.DataFrame(flat)
    agg = flatdf.groupby(key, dropna=False)["vaf"].mean().reset_index().rename(columns={"vaf":"agg_score"})
    # tri: score↓, hidden_dim↑, k_lag↑, n_pca↑ (ta règle)
    agg_sorted = agg.sort_values(by=["agg_score","hidden_dim","k_lag","n_pca"],
                                 ascending=[False, True, True, True], kind="mergesort")
    best = (agg_sorted.sort_values(["decoder","agg_score"], ascending=[True, False], kind="mergesort")
                     .drop_duplicates(subset=["decoder"], keep="first"))
    return best

def build_day0_units_from_best(df: pd.DataFrame, best_cfgs: pd.DataFrame, use_all_seeds=True) -> pd.DataFrame:
    """Retourne un DF 'units' avec colonnes: decoder, day_int(=0), pair_id, fold, seed, VAF_unit"""
    recs = []
    for _, b in best_cfgs.iterrows():
        mask = (df["decoder"].eq(b["decoder"]) &
                (df["hidden_dim"]==b["hidden_dim"]) &
                (df["k_lag"]==b["k_lag"]) &
                (df["n_pca"]==b["n_pca"]) &
                (df["num_epochs"]==b["num_epochs"]) &
                (df["lr"]==b["lr"]))
        df_cfg = df[mask].copy()
        if df_cfg.empty: continue
        if not use_all_seeds:
            df_cfg["fold_mean"] = df_cfg["fold_vafs"].apply(mean_from_folds)
            df_cfg = df_cfg.sort_values("fold_mean", ascending=False).head(1)
        for _, r in df_cfg.iterrows():
            vafs = r["fold_vafs"] if isinstance(r["fold_vafs"], (list, tuple, np.ndarray)) else []
            for fold, v in enumerate(vafs):
                try: v = float(v)
                except: continue
                if not np.isfinite(v): continue
                seed = int(r["seed"]) if np.isfinite(r["seed"]) else 0
                pair_id = f"{fold}_{seed}"
                recs.append(dict(decoder=r["decoder"], day_int=0, fold=fold, seed=seed,
                                 pair_id=pair_id, VAF_unit=v))
    units = pd.DataFrame(recs)
    return units

def plot_violin(units: pd.DataFrame, out_png: Path):
    order = ["gru","lstm","ligru","linear"]
    present = [d for d in order if d in units["decoder"].unique().tolist()]
    data = [units.loc[units.decoder==d, "VAF_unit"].values for d in present]
    meds = [np.median(x) if len(x) else np.nan for x in data]
    plt.figure(figsize=(12,6))
    parts = plt.violinplot(data, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_alpha(0.35); b.set_facecolor(DECODER_COLORS.get(present[i], "gray"))
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals)==0: continue
        x = np.full_like(vals, i, dtype=float) + rng.uniform(-0.07,0.07,size=len(vals))
        plt.scatter(x, vals, s=18, alpha=0.9, c=DECODER_COLORS.get(present[i-1], "gray"))
    plt.scatter(np.arange(1,len(meds)+1), meds, s=30, c="black", zorder=10, label="Median")
    plt.xticks(np.arange(1,len(present)+1), [d.upper() if d!="ligru" else "LiGRU" for d in present])
    plt.ylabel("VAF (day 0, mean over muscles)"); plt.title("Day-0 validation • avg over muscles (CV)")
    plt.legend(loc="center left", bbox_to_anchor=(1,0.5), frameon=False)
    plt.grid(True, axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close()
    print(f"[save] {out_png}")

def wilcoxon_paired(units: pd.DataFrame, out_csv: Path):
    decs = sorted(units["decoder"].unique())
    rows = []
    for A, B in combinations(decs, 2):
        Ua = units[units.decoder==A].set_index("pair_id")["VAF_unit"]
        Ub = units[units.decoder==B].set_index("pair_id")["VAF_unit"]
        common = Ua.index.intersection(Ub.index)
        if len(common) < 2:
            rows.append([A,B,"wilcoxon_paired",np.nan,np.nan,len(common),np.nan,np.nan,np.nan])
            continue
        x = Ua.loc[common].values; y = Ub.loc[common].values
        diffs = x - y  # A minus B
        method = "exact" if len(diffs) <= 25 else "approx"
        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method=method)
        rows.append([A,B,"wilcoxon_paired",float(stat),float(p),int(len(common)),
                     float(np.median(x)), float(np.median(y)), float(np.median(diffs))])
    res = pd.DataFrame(rows, columns=[
        "decoder_A","decoder_B","test","W","p_value","n_pairs",
        "median_A","median_B","median_diff_AminusB"
    ])
    if not res["p_value"].isna().all():
        res["p_holm"] = holm_bonferroni(res["p_value"].values)
    res.to_csv(out_csv, index=False); print(f"[save] {out_csv}")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="ALL_gridsearch_results.pkl")
    ap.add_argument("--out_dir", type=str, default="figs_day0")
    ap.add_argument("--use_all_seeds", action="store_true", help="stack folds from all seeds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_grid_pkl(Path(args.pkl))
    best = select_best_per_decoder(df)
    units = build_day0_units_from_best(df, best, use_all_seeds=args.use_all_seeds)
    if units.empty: raise RuntimeError("No units to analyze (check fold_vafs / seeds).")

    print("\n[summary] points per decoder:", units.groupby("decoder")["VAF_unit"].count().to_dict())
    plot_violin(units, out_dir / "day0_validation_violin.png")
    res = wilcoxon_paired(units, out_dir / "day0_validation_stats.csv")
    print("\n", res.to_string(index=False))

if __name__ == "__main__":
    main()
