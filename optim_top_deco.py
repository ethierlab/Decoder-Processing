#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sélection de la meilleure configuration à partir d'un gridsearch sauvegardé
dans ALL_gridsearch_results.pkl (liste de dicts).

Fonctions :
- Nettoyage robuste des types (évite les colonnes 'object' qui cassent les plots/tri).
- Agrégation par config (decoder, hidden_dim, k_lag, n_pca, num_epochs, lr) en moyennant
  tous les VAFs de folds×seeds disponibles.
- Classement configurable:
    * "user_rule"    : score↓, hidden_dim↑, k_lag↑, n_pca↑
    * "params_first" : score↓, num_params↑, hidden_dim↑, k_lag↑, n_pca↑
    * "decoder_aware": score↓, hidden_dim↑, (RNN: n_pca↑ puis k_lag↑; Linear: k_lag↑ puis n_pca↑)
- Export JSON pour ré-entraînement final.

Usage:
    python select_best_config.py
"""

import json, pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
INPUT_PKL   = Path("ALL_gridsearch_results_early.pkl")
RANKING_MODE = "user_rule"      # "user_rule" | "params_first" | "decoder_aware"
OUT_BEST_OVERALL = Path("best_overall_config.json")
OUT_BEST_PER_DEC = Path("best_per_decoder_configs.json")

# -------------- I/O de base --------------------
if not INPUT_PKL.exists():
    raise FileNotFoundError(f"Fichier introuvable: {INPUT_PKL.resolve()}")

with open(INPUT_PKL, "rb") as f:
    rows: List[Dict[str, Any]] = pickle.load(f)

df = pd.DataFrame(rows)

# -------------- Normalisation colonnes ----------
# On s'assure que toutes les colonnes attendues existent
for c in ["decoder", "mean_vaf", "num_params", "hidden_dim", "k_lag",
          "n_pca", "num_epochs", "lr", "seed", "fold_vafs"]:
    if c not in df.columns:
        df[c] = np.nan

# Cast numériques robustes
num_cols = ["mean_vaf", "num_params", "hidden_dim", "k_lag", "n_pca", "num_epochs", "lr", "seed"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Filtre runs valides
df = df.dropna(subset=["decoder", "hidden_dim", "k_lag", "n_pca", "num_epochs", "lr"])
df = df[df["decoder"].astype(str).str.len() > 0]
df = df[df["num_params"].fillna(0) >= 0]  # certains jobs peuvent ne pas l'avoir; on fusionnera plus bas

# Ré-évalue mean_vaf à partir de fold_vafs si possible (source de vérité)
def mean_from_folds(v):
    try:
        arr = np.array(v, dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(arr.mean())
    except Exception:
        return np.nan

recalc = df["fold_vafs"].apply(mean_from_folds)
df.loc[recalc.notna(), "mean_vaf"] = recalc[recalc.notna()]
df = df[df["mean_vaf"].notna() & np.isfinite(df["mean_vaf"])]

# -------------- Aplatissement folds×seeds -------
flat = []
for _, row in df.iterrows():
    vafs = row["fold_vafs"] if isinstance(row["fold_vafs"], (list, tuple, np.ndarray)) else []
    for v in vafs:
        try:
            v = float(v)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        flat.append({
            "decoder":    str(row["decoder"]),
            "hidden_dim": int(row["hidden_dim"]),
            "k_lag":      int(row["k_lag"]),
            "n_pca":      int(row["n_pca"]),
            "num_epochs": int(row["num_epochs"]),
            "lr":         float(row["lr"]),
            "seed":       int(row["seed"]) if np.isfinite(row["seed"]) else None,
            "vaf":        v,
        })

flatdf = pd.DataFrame(flat)
if flatdf.empty:
    raise RuntimeError("Aucun VAF exploitable après nettoyage (folds×seeds).")

# -------------- Agrégation par config -----------
key_cols = ["decoder", "hidden_dim", "k_lag", "n_pca", "num_epochs", "lr"]
agg = (
    flatdf.groupby(key_cols, dropna=False)
          .agg(agg_score=("vaf", "mean"), score_std=("vaf", "std"), n_points=("vaf", "size"))
          .reset_index()
)

# Récupérer un num_params pour chaque config (identique sur seeds/folds pour une config donnée)
param_ref = (
    df.groupby(key_cols, dropna=False)
      .agg(min_params=("num_params", "min"))
      .reset_index()
)
agg = agg.merge(param_ref, on=key_cols, how="left")

# -------------- Classement ----------------------
def rank_configs(agg: pd.DataFrame, mode: str) -> pd.DataFrame:
    a = agg.copy()
    # Valeurs de secours si num_params est manquant
    if "min_params" not in a.columns:
        a["min_params"] = np.nan
    a["min_params"] = pd.to_numeric(a["min_params"], errors="coerce")

    if mode == "params_first":
        sort_by = ["agg_score", "min_params", "hidden_dim", "k_lag", "n_pca"]
        ascending = [False, True, True, True, True]
        return a.sort_values(sort_by, ascending=ascending, kind="mergesort")

    if mode == "decoder_aware":
        is_linear = a["decoder"].str.lower().eq("linear")
        tie2 = np.where(is_linear, a["k_lag"], a["n_pca"])
        tie3 = np.where(is_linear, a["n_pca"], a["k_lag"])
        a = a.assign(tie2=tie2, tie3=tie3)
        return a.sort_values(
            by=["agg_score", "hidden_dim", "tie2", "tie3"],
            ascending=[False, True, True, True],
            kind="mergesort"
        )

    # Par défaut: "user_rule" — ton protocole exact
    return a.sort_values(
        by=["agg_score", "hidden_dim", "k_lag", "n_pca"],
        ascending=[False, True, True, True],
        kind="mergesort"
    )

agg_sorted = rank_configs(agg, RANKING_MODE)

# Meilleure globale
best_overall = agg_sorted.iloc[0].to_dict()

# Meilleure par décodeur (1ère ligne par décodeur après tri)
best_per_decoder = (
    agg_sorted.drop_duplicates(subset=["decoder"], keep="first")
              .sort_values("decoder")
)

# -------------- Sorties & impression ------------
print("\n# ==== MEILLEURE CONFIG GLOBALE ====")
print(json.dumps(best_overall, indent=2, ensure_ascii=False))

print("\n# ==== MEILLEURE CONFIG PAR DÉCODEUR ====")
for _, r in best_per_decoder.iterrows():
    print(f"{r['decoder'].upper():<8} | score={r['agg_score']:.4f} ± {0 if pd.isna(r['score_std']) else r['score_std']:.4f} "
          f"| hid={int(r['hidden_dim'])} | k={int(r['k_lag'])} | n_pca={int(r['n_pca'])} "
          f"| lr={r['lr']:.4g} | epochs={int(r['num_epochs'])} | n={int(r['n_points'])} "
          f"| params={int(r['min_params']) if pd.notna(r['min_params']) else -1}")

# JSON
with open(OUT_BEST_OVERALL, "w") as f:
    json.dump(best_overall, f, indent=2, ensure_ascii=False)

recs = best_per_decoder.to_dict(orient="records")
with open(OUT_BEST_PER_DEC, "w") as f:
    json.dump(recs, f, indent=2, ensure_ascii=False)

print(f"\nÉcrit: {OUT_BEST_OVERALL.resolve()}")
print(f"       {OUT_BEST_PER_DEC.resolve()}")
