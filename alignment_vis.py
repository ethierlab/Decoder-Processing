#!/usr/bin/env python3
"""
align_visualization.py
────────────────────────────────────────────────────────────────────────
Visualise across-day PCA alignment of smoothed spike counts.

Pipeline reproduced here
  • 20 ms binning               (original: 1 kHz → 20 ms bins)
  • Gaussian smoothing σ = 25 ms
  • optional z-score
  • PCA (k comps) on *reference* day
  • Linear least-squares rotation  R = pinv(V_tgt) @ V_ref
  • 2-D or 3-D scatter of reference, target-raw and target-aligned latents

Trial selection
  • keep only those whose `trial_target_dir` ≈ `--angle` deg (± `--tol`)  
  • for every such trial include **all bins in [−1 s, +3 s] w.r.t. trial_start_time**

Day selection
  • `--offsets` 0 Δ1 Δ2 … where 0 = earliest date in the pickle
    (0 ⇒ reference basis, others ⇒ targets)

Example
    python align_visualization.py \
        --pickle combined.pkl \
        --angle 180 --offsets 0 2 5 \
        --k 16 --samples 800 --dim 3 --save align_180.png
"""
import argparse, os, sys, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from sklearn.decomposition import PCA

# ───────── preprocessing parameters ──────────────────────────────────────────
BIN_FACTOR       = 20        # 20 × 1 ms → 20 ms bin
BIN_SIZE_ORIG_S  = 0.001     # original resolution
SMOOTHING_LEN_S  = 0.05      # window 50 ms  → σ = 25 ms
APPLY_ZSCORE     = False     # set True if desired
WINDOW_PRE_S     = 1.0       # seconds BEFORE trial_start_time
WINDOW_POST_S    = 3.0       # seconds AFTER  trial_start_time
SEED             = 42
# ──────────────────────────────────────────────────────────────────────────────

# utilities ───────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def downsample_spikes(df, factor):
    """Sum‐bin spike counts every <factor> rows."""
    if df.empty or df.shape[0] < factor:
        return np.empty((0, df.shape[1]))
    T_new = df.shape[0] // factor
    return df.values[:T_new*factor].reshape(T_new, factor, df.shape[1]).sum(1)

def smooth(arr, bin_size, smooth_len):
    sigma = (smooth_len / bin_size) / 2        # samples σ  (50 ms window → σ 25 ms)
    out = np.zeros_like(arr, dtype=float)
    for ch in range(arr.shape[1]):
        out[:, ch] = gaussian_filter1d(arr[:, ch].astype(float), sigma)
    return out

def zscore(arr, axis=0, eps=1e-8):
    return (arr - arr.mean(axis, keepdims=True)) / (arr.std(axis, keepdims=True) + eps)

def sub_sample(rng, x, n):
    """Randomly pick ≤ n rows for clearer scatter plots."""
    if x.shape[0] <= n:
        return x
    idx = rng.choice(x.shape[0], n, replace=False)
    return x[idx]

# trial extraction ────────────────────────────────────────────────────────────
def collect_window_samples(df_day, *, angle=None, tol=1.0):
    """
    For one calendar day (DataFrame subset) return an array whose rows are
    every 20-ms bin that falls inside

        [trial_start_time − 1 s,  trial_start_time + 3 s]

    **for trials whose `trial_target_dir` ≈ angle** (if angle given).
    """
    if df_day.empty:
        return np.empty((0, 0))

    rows = []
    for _, row in df_day.iterrows():
        t_start_vec = row.get("trial_start_time")   # (n_trials,)
        t_dir_vec   = row.get("trial_target_dir")   # (n_trials,)
        if t_start_vec is None or t_dir_vec is None:
            continue

        t_start_vec = np.asarray(t_start_vec).flatten()
        t_dir_vec   = np.asarray(t_dir_vec).flatten()
        n_trials    = min(len(t_start_vec), len(t_dir_vec))

        # preprocess spikes once for the whole xds row
        spk_df = row["spike_counts"]
        if not isinstance(spk_df, pd.DataFrame) or spk_df.empty:
            continue

        ds = downsample_spikes(spk_df, BIN_FACTOR)            # (T', n_units)
        if ds.size == 0:
            continue
        sm = smooth(ds, BIN_SIZE_ORIG_S*BIN_FACTOR, SMOOTHING_LEN_S)
        if APPLY_ZSCORE:
            sm = zscore(sm, 0)

        tf_raw = row["time_frame"].flatten()
        tf_ds  = tf_raw[:ds.shape[0]*BIN_FACTOR:BIN_FACTOR]   # centres of 20-ms bins

        for j in range(n_trials):
            if angle is not None and not np.isclose(t_dir_vec[j], angle, atol=tol):
                continue
            t0, t1 = t_start_vec[j] - WINDOW_PRE_S, t_start_vec[j] + WINDOW_POST_S
            idxs = np.where((tf_ds >= t0) & (tf_ds <= t1))[0]
            if idxs.size:
                rows.append(sm[idxs])                         # (m, n_units)

    return np.vstack(rows) if rows else np.empty((0, 0))

# rotation helper
def rotation(V_ref_T, V_tgt_T):          # both (p × k)
    return pinv(V_tgt_T) @ V_ref_T       # (k × k)

# main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True, help="Pickle produced by process_single_mat_file_new")
    ap.add_argument("--offsets", nargs="+", type=int, default=[0, 1],
                    help="Space-separated day offsets relative to the earliest "
                         "recording (0 = reference). Example: 0 2 5")
    ap.add_argument("--angle", type=float, default=None,
                    help="Keep only trials whose trial_target_dir ≈ this angle (deg).")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="Tolerance ± deg when matching --angle.")
    ap.add_argument("--k", type=int, default=16, help="PCA dimensionality")
    ap.add_argument("--samples", type=int, default=800, help="Sub-sample for plotting clarity")
    ap.add_argument("--dim", choices=[2, 3], type=int, default=3, help="Plot dimensionality")
    ap.add_argument("--save", default=None, metavar="FILE", help="If given, save figure instead of showing")
    args = ap.parse_args()

    set_seed(SEED)

    # ——— load DataFrame —————————————————————————————
    df = pd.read_pickle(args.pickle)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")

    days = sorted(df["date"].unique())
    if any(o < 0 or o >= len(days) for o in args.offsets):
        sys.exit(f"--offsets must be in 0–{len(days)-1}")

    # reference = earliest date (offset 0)
    if 0 not in args.offsets:
        args.offsets.insert(0, 0)
    day_ref = days[0]                               # earliest calendar day
    target_days = [days[o] for o in args.offsets if o != 0]

    # ——— collect data ——————————————————————————————
    X_ref = collect_window_samples(df[df["date"] == day_ref],
                                   angle=args.angle, tol=args.tol)
    if X_ref.size == 0:
        sys.exit("No data on reference day after filtering.")

    # build reference PCA once
    k       = args.k
    pca_ref = PCA(n_components=k, random_state=SEED).fit(X_ref)
    Z_ref   = pca_ref.transform(X_ref)
    V_ref_T = pca_ref.components_.T        # (p × k)

    rng = np.random.default_rng(SEED)
    Z_ref_plot = sub_sample(rng, Z_ref, args.samples)

    # ——— prepare figure ————————————————————————————
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d" if args.dim == 3 else None)

    if args.dim == 3:
        ax.scatter(Z_ref_plot[:, 0], Z_ref_plot[:, 1], Z_ref_plot[:, 2],
                   s=8, alpha=.6, label=f"{day_ref.date()} (ref)")
    else:
        ax.scatter(Z_ref_plot[:, 0], Z_ref_plot[:, 1],
                   s=8, alpha=.6, label=f"{day_ref.date()} (ref)")

    # ——— loop over every target day ——————————————————
    for d in target_days:
        X_tgt = collect_window_samples(df[df["date"] == d],
                                       angle=args.angle, tol=args.tol)
        if X_tgt.size == 0:
            print(f"⚠  No data on {d.date()} after filtering; skipping.")
            continue

        pca_tgt = PCA(n_components=k, random_state=SEED).fit(X_tgt)
        Z_tgt   = pca_tgt.transform(X_tgt)
        R       = rotation(V_ref_T, pca_tgt.components_.T)    # (k × k)
        Z_al    = Z_tgt @ R

        Z_tgt_plot = sub_sample(rng, Z_tgt, args.samples)
        Z_al_plot  = sub_sample(rng, Z_al,  args.samples)

        lab_raw = f"{d.date()} (raw)"
        lab_al  = f"{d.date()} (aligned)"
        if args.dim == 3:
            ax.scatter(Z_tgt_plot[:, 0], Z_tgt_plot[:, 1], Z_tgt_plot[:, 2],
                       s=8, marker='^', alpha=.55, label=lab_raw)
            ax.scatter(Z_al_plot[:, 0], Z_al_plot[:, 1], Z_al_plot[:, 2],
                       s=8, marker='s', alpha=.55, label=lab_al)
        else:
            ax.scatter(Z_tgt_plot[:, 0], Z_tgt_plot[:, 1],
                       s=8, marker='^', alpha=.55, label=lab_raw)
            ax.scatter(Z_al_plot[:, 0], Z_al_plot[:, 1],
                       s=8, marker='s', alpha=.55, label=lab_al)

    # ——— cosmetics ———————————————————————————————
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if args.dim == 3:
        ax.set_zlabel("PC3")
    title = "PCA alignment"
    if args.angle is not None:
        title += f" – angle {args.angle}°"
    ax.set_title(title)
    ax.grid(True, ls=":")
    ax.legend(fontsize=8)
    plt.tight_layout()

    # ——— finish ————————————————————————————————
    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
