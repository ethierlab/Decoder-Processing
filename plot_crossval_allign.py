# -*- coding: utf-8 -*-
"""
Plot cross‑validation results for a *single* decoder across several
realignment conditions (each stored in its own pickle file).

Each pickle must contain the same keys that your original script expects:
    test_days (1‑D array of datetime‑like objects)
    gru_vafs, lstm_vafs, lin_vafs, ligru_vafs (shape = n_cv_runs × n_days)

Usage (inside this file, edit the two lists):
    pkl_paths = ["align_A.pkl", "align_B.pkl", "align_C.pkl"]
    labels    = ["Align‑A", "Align‑B", "Align‑C"]
    decoder   = "GRU"  # one of: "GRU", "LSTM", "Linear", "LiGRU"
    plot_crossval_results_multi(pkl_paths, labels, decoder)

All four plots from the original script are reproduced, but series / boxes
are now coloured by *realignment* instead of *decoder*.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# 0) Helper: outlier removal (unchanged)
# -----------------------------------------------------------------------------

def remove_outliers(group: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Return rows whose ``vaf`` lies within ±``threshold`` · SD of the mean."""
    if len(group) < 2:
        return group
    mean_vaf = group["vaf"].mean()
    std_vaf = group["vaf"].std()
    if std_vaf == 0:
        return group
    lower, upper = mean_vaf - threshold * std_vaf, mean_vaf + threshold * std_vaf
    return group[(group["vaf"] >= lower) & (group["vaf"] <= upper)]


# -----------------------------------------------------------------------------
# 1) Core loader – returns a tidy, *cleaned* DataFrame for the chosen decoder
# -----------------------------------------------------------------------------

def _load_single_file(
    pkl_path: os.PathLike | str,
    decoder: str,
    realign_label: str,
    outlier_thresh: float = 2.0,
) -> pd.DataFrame:
    """Read *one* pickle file and return a cleaned, tidy DataFrame for *one* decoder."""

    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(pkl_path)

    res = pd.read_pickle(pkl_path)

    # ---------------------------------------------------------------------
    # Unpack arrays
    # ---------------------------------------------------------------------
    decoder_map = {
        "GRU": res["gru_vafs"],
        "LSTM": res["lstm_vafs"],
        "Linear": res["lin_vafs"],
        "LiGRU": res["ligru_vafs"],
    }
    if decoder not in decoder_map:
        raise KeyError(
            f"Decoder '{decoder}' not present. Choose from {list(decoder_map)}.")

    arr = np.asarray(decoder_map[decoder])  # (n_cv_runs, n_days)
    test_days = res["test_days"]

    # Integer day numbers (relative to first test day *within this file*)
    base_day = test_days[0]
    day_nums = [(d.date() - base_day.date()).days for d in test_days]
    n_cv, n_days = arr.shape

    # Build tidy DF
    rows = [
        {
            "crossval": i_cv,
            "day_idx": i_day,
            "day_num": day_nums[i_day],
            "decoder": decoder,
            "vaf": arr[i_cv, i_day],
            "realign": realign_label,
        }
        for i_cv in range(n_cv)
        for i_day in range(n_days)
    ]
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # Outlier removal *within this realignment/decoder/day* group
    # ---------------------------------------------------------------------
    df_clean = (
        df.groupby(["realign", "day_num"], group_keys=False)
        .apply(remove_outliers, threshold=outlier_thresh)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # Relative VAF loss   (day0 − dayX)   *per cross‑val run*
    # ---------------------------------------------------------------------
    day0_refs = (
        df_clean[df_clean.day_num == 0]
        .set_index("crossval")["vaf"]
        .to_dict()
    )

    df_clean["rel_loss"] = df_clean.apply(
        lambda r: np.nan if r.crossval not in day0_refs else day0_refs[r.crossval] - r.vaf,
        axis=1,
    )

    return df_clean


# -----------------------------------------------------------------------------
# 2) Master function – loops over several pickle files
# -----------------------------------------------------------------------------

def plot_crossval_results_multi(
    pkl_paths: List[os.PathLike | str],
    labels: List[str],
    decoder: str,
    save_dir: os.PathLike | str = ".",
    dpi: int = 300,
    outlier_thresh: float = 2.0,
) -> None:
    """Overlay *one* decoder's performance from several realignment pickles."""

    if len(pkl_paths) != len(labels):
        raise ValueError("pkl_paths and labels must have the same length.")

    # ---------------------------------------------------------------------
    # 2.1) Stack all cleaned DataFrames together
    # ---------------------------------------------------------------------
    df_all = pd.concat(
        [
            _load_single_file(path, decoder, lbl, outlier_thresh)
            for path, lbl in zip(pkl_paths, labels)
        ],
        ignore_index=True,
    )

    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10", len(labels))
    colour_map = dict(zip(labels, palette))

    # ---------------------------------------------------------------------
    # 3) Plot #1 – Mean VAF over days
    # ---------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for lbl in labels:
        sub = df_all[df_all.realign == lbl]
        grp = sub.groupby("day_num").vaf
        mean, std = grp.mean(), grp.std()
        ax1.plot(mean.index, mean.values, "o-", label=lbl, color=colour_map[lbl])
        ax1.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=colour_map[lbl])
    ax1.set(
        xlabel="Days from day0 (per realignment)",
        ylabel="VAF (mean across CV runs)",
        title=f"{decoder} – VAF across Realignments",
    )
    ax1.legend(title="Realignment")
    plt.tight_layout()
    plt.savefig(Path(save_dir, f"mean_vaf_{decoder}.png"), dpi=dpi)

    # ---------------------------------------------------------------------
    # 4) Plot #2 – Relative VAF loss (day0 − dayX)
    # ---------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for lbl in labels:
        sub = df_all[df_all.realign == lbl]
        grp = sub.groupby("day_num").rel_loss
        mean, std = grp.mean(), grp.std()
        ax2.plot(mean.index, mean.values, "o-", label=lbl, color=colour_map[lbl])
        ax2.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=colour_map[lbl])
    ax2.set(
        xlabel="Days from day0",
        ylabel="Relative VAF Loss (day0 – dayX)",
        title=f"{decoder} – Relative VAF Loss",
    )
    ax2.legend(title="Realignment")
    plt.tight_layout()
    plt.savefig(Path(save_dir, f"rel_loss_{decoder}.png"), dpi=dpi)

    # ---------------------------------------------------------------------
    # 5) Plot #3 – Cumulative loss
    # ---------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for lbl in labels:
        sub = df_all[df_all.realign == lbl]
        # pivot: index=day_num, columns=crossval -> rel_loss
        piv = sub.pivot_table(index="day_num", columns="crossval", values="rel_loss", aggfunc="mean")
        piv = piv.sort_index()
        csum = piv.cumsum()
        mean_csum, std_csum = csum.mean(axis=1), csum.std(axis=1)
        ax3.plot(mean_csum.index, mean_csum.values, "o-", label=lbl, color=colour_map[lbl])
        ax3.fill_between(mean_csum.index, mean_csum - std_csum, mean_csum + std_csum, alpha=0.2, color=colour_map[lbl])
    ax3.set(
        xlabel="Days from day0",
        ylabel="Cumulative VAF Loss",
        title=f"{decoder} – Cumulative Loss",
    )
    ax3.legend(title="Realignment")
    plt.tight_layout()
    plt.savefig(Path(save_dir, f"cum_loss_{decoder}.png"), dpi=dpi)

    # ---------------------------------------------------------------------
    # 6) Plot #4 – Box/strip per day, coloured by realignment
    # ---------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_all,
        x="day_num",
        y="vaf",
        hue="realign",
        whis=[5, 95],
        showfliers=False,
        palette=colour_map,
        ax=ax4,
    )
    sns.stripplot(
        data=df_all,
        x="day_num",
        y="vaf",
        hue="realign",
        dodge=True,
        alpha=0.4,
        size=3,
        marker="o",
        palette=colour_map,
        ax=ax4,
    )
    # Deduplicate legend
    handles, labels_ = ax4.get_legend_handles_labels()
    ax4.legend(handles[:len(labels)], labels_[:len(labels)], title="Realignment", frameon=True)
    ax4.set(
        title=f"{decoder} – VAF Distribution per Day",
        xlabel="Day # (relative)",
        ylabel="VAF",
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir, f"boxplot_vaf_{decoder}.png"), dpi=dpi)

    plt.show()


# -----------------------------------------------------------------------------
# 3) Entry‑point – edit paths & labels here or call from another script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # <‑‑‑ EDIT THESE THREE LINES TO MATCH YOUR DATA ‑‑‑>
    pkl_paths = ["crossval_results_bland.pkl", "crossval_results_realignement.pkl", "crossval_results_recalc.pkl"]
    labels = ["Naive", "Realign", "Recalculated"]
    decoder = "GRU"  # pick your decoder

    plot_crossval_results_multi(pkl_paths, labels, decoder)
