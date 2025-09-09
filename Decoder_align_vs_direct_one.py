# Decoder_align_vs_direct_one.py
# One decoder, overlay per-day aligned vs direct (mean over muscles),
# + paired Wilcoxon per day and Stouffer-combined significance across days.

import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm

# ------------------------ IO / utils ------------------------

def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "crossday_results_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No PKL files found in {results_dir} (pattern crossday_results_*.pkl).")
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)

    # normalize essentials
    if "day_int" in df.columns:
        df["day_int"] = pd.to_numeric(df["day_int"], errors="coerce")
    elif "day" in df.columns:
        df["day_int"] = pd.to_numeric(df["day"], errors="coerce")
    else:
        raise ValueError("Need a 'day_int' or 'day' column in the PKLs.")

    if "fold" not in df.columns: df["fold"] = 0
    if "seed" not in df.columns: df["seed"] = 0
    if "emg_channel" not in df.columns: df["emg_channel"] = -1
    if "vaf" not in df.columns: raise ValueError("Need column 'vaf'.")

    if "decoder" not in df.columns or "dim_red" not in df.columns:
        raise ValueError("Need 'decoder' and 'dim_red' columns.")

    # align normalization -> 'aligned' / 'direct' / 'crossval'
    col = "align" if "align" in df.columns else ("alignment" if "alignment" in df.columns else None)
    if col is None:
        raise ValueError("Need 'align' column.")
    v = df[col].astype(str).str.lower()
    df["align_norm"] = np.where(v.str.contains("align"), "aligned",
                         np.where(v.str.contains("direct"), "direct",
                                  np.where(v.str.contains("cross"), "crossval", v)))
    return df

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def mean_over_muscles(df, exclude_channels=None):
    """Mean VAF over EMG channels per (decoder, dim_red, align, day, fold, seed)."""
    g = df.copy()
    if exclude_channels:
        g = g[~g["emg_channel"].isin(exclude_channels)]
    keys = ["decoder","dim_red","align_norm","day_int","fold","seed"]
    out = g.groupby(keys, dropna=False)["vaf"].mean().reset_index()
    out = out.rename(columns={"vaf":"vaf_mean_musc"})
    # add a pairing ID (fold×seed)
    out["pair_id"] = out["fold"].astype(str) + "_" + out["seed"].astype(str)
    return out

def holm(pvals):
    p = np.array(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float); prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        val = max(val, prev)
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj

# ------------------------ plotting ------------------------

def overlay_boxplots_per_day_one_decoder(df_avg, decoder, dim_red, out_path,
                                         ylim=(-0.5, 1.05), title_extra=""):
    """Two boxplots per day on the same x: aligned (blue) vs direct (orange, hatched)."""
    G = df_avg[(df_avg["decoder"]==decoder) & (df_avg["dim_red"]==dim_red)].copy()
    if G.empty:
        print(f"[plot] No data for {decoder}, {dim_red}.")
        return

    days = sorted(G["day_int"].dropna().unique())
    data_A = []  # aligned
    data_B = []  # direct
    keep_days = []

    for d in days:
        a = G[(G["day_int"]==d) & (G["align_norm"]=="aligned")]["vaf_mean_musc"].values
        b = G[(G["day_int"]==d) & (G["align_norm"]=="direct")]["vaf_mean_musc"].values
        if len(a)==0 and len(b)==0:
            continue
        data_A.append(a)
        data_B.append(b)
        keep_days.append(d)

    if not keep_days:
        print("[plot] Nothing to plot.")
        return

    x = np.arange(len(keep_days), dtype=float)
    delta = 0.12
    pos_A = x - delta
    pos_B = x + delta

    plt.figure(figsize=(16,5))
    bpA = plt.boxplot(data_A, positions=pos_A, widths=0.18, showfliers=False,
                      patch_artist=True)
    bpB = plt.boxplot(data_B, positions=pos_B, widths=0.18, showfliers=False,
                      patch_artist=True)

    # style
    for p in bpA["boxes"]:
        p.set_facecolor("#6BA6FF"); p.set_alpha(0.7)
    for p in bpB["boxes"]:
        p.set_facecolor("#FFA858"); p.set_alpha(0.7); p.set_hatch("//")
    for k in ["whiskers","caps","medians"]:
        for artist in bpA[k]+bpB[k]:
            artist.set_linewidth(1.0)

    plt.xticks(x, [str(int(d)) for d in keep_days])
    plt.xlabel("Day")
    plt.ylabel("VAF (mean over muscles)")
    ttl = f"{decoder} • {dim_red} • mean over muscles • aligned vs direct"
    if title_extra: ttl += f" ({title_extra})"
    # plt.title(ttl)
    plt.ylim(ylim)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#6BA6FF", alpha=0.7, label="aligned"),
               Patch(facecolor="#FFA858", alpha=0.7, hatch="//", label="direct")]
    plt.legend(handles=handles, title="Alignment", loc="upper right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved figure:", out_path)
    plt.close()

# ------------------------ stats ------------------------

def per_day_paired_wilcoxon(df_avg, decoder, dim_red):
    """Paired across fold×seed for each day (aligned vs direct)."""
    rows = []
    G = df_avg[(df_avg["decoder"]==decoder) & (df_avg["dim_red"]==dim_red)].copy()
    if G.empty: return pd.DataFrame()

    for d, gd in G.groupby("day_int"):
        a = gd[gd["align_norm"]=="aligned"][["pair_id","vaf_mean_musc"]].rename(
             columns={"vaf_mean_musc":"A"})
        b = gd[gd["align_norm"]=="direct"][["pair_id","vaf_mean_musc"]].rename(
             columns={"vaf_mean_musc":"B"})
        merged = pd.merge(a, b, on="pair_id", how="inner")
        if len(merged) < 2:
            continue

        diffs = merged["A"].values - merged["B"].values

        # Two-sided exact/approx depending on n
        mode = "exact" if len(diffs) <= 25 else "approx"
        W2, p2 = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode=mode)

        # One-sided for "aligned > direct" (used for Stouffer direction)
        Wg, pg = wilcoxon(diffs, zero_method="wilcox", alternative="greater", mode=mode)

        rows.append({
            "decoder": decoder,
            "dim_red": dim_red,
            "day": int(d),
            "test": "Wilcoxon paired (fold×seed)",
            "W_two_sided": float(W2),
            "p_two_sided": float(p2),
            "p_greater": float(pg),
            "n_pairs": int(len(diffs)),
            "median_aligned": float(np.median(merged["A"].values)),
            "median_direct": float(np.median(merged["B"].values)),
            "median_diff_AminusB": float(np.median(diffs))
        })
    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    if not out.empty:
        out["p_two_sided_holm"] = holm(out["p_two_sided"].values)
    return out

def stouffer_one_sided(p_greater, weights=None):
    """Combine one-sided p-values (aligned > direct) with Stouffer's Z."""
    p = np.asarray(p_greater, float)
    if weights is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(weights, float)
    # convert p to z (one-sided)
    z = norm.isf(p)  # Φ^{-1}(1 - p)
    Z = np.sum(w * z) / np.sqrt(np.sum(w**2))
    p_one = norm.sf(Z)  # one-sided
    p_two = min(1.0, 2.0 * p_one)  # report also two-sided
    return Z, p_one, p_two

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="One-decoder overlay (aligned vs direct) + stats.")
    ap.add_argument("--results_dir", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="figs_align_one")
    ap.add_argument("--decoder", type=str, default="LSTM")
    ap.add_argument("--dim_red", type=str, default="PCA", choices=["PCA","UMAP"])
    ap.add_argument("--exclude_channels", nargs="+", type=int, default=None,
                    help="EMG channels to exclude before averaging (e.g., 0 5 6).")
    args = ap.parse_args()

    df = load_results(args.results_dir)
    df_avg = mean_over_muscles(df, exclude_channels=args.exclude_channels)

    # Plot
    outdir = ensure_dir(args.out_dir)
    fig_path = os.path.join(outdir, f"overlay_{args.decoder}_{args.dim_red}.png")
    title_extra = None
    if args.exclude_channels:
        title_extra = f"excl ch {','.join(map(str,args.exclude_channels))}"
    overlay_boxplots_per_day_one_decoder(df_avg, args.decoder, args.dim_red, fig_path,
                                         ylim=(0, 1.05), title_extra=title_extra)

    # Stats
    per_day = per_day_paired_wilcoxon(df_avg, args.decoder, args.dim_red)
    if per_day.empty:
        print("No overlapping aligned/direct pairs -> no stats.")
        return

    # Stouffer combine (one-sided aligned>direct), weight by sqrt(n_pairs)
    Z, p_one, p_two = stouffer_one_sided(per_day["p_greater"].values,
                                         weights=np.sqrt(per_day["n_pairs"].values))

    per_day_path = os.path.join(outdir, f"stats_per_day_{args.decoder}_{args.dim_red}.csv")
    per_day.to_csv(per_day_path, index=False)
    print("Saved per-day stats:", per_day_path)

    overall = pd.DataFrame([{
        "decoder": args.decoder,
        "dim_red": args.dim_red,
        "days_used": int(per_day.shape[0]),
        "total_pairs": int(per_day["n_pairs"].sum()),
        "Z_stouffer_one_sided": float(Z),
        "p_stouffer_one_sided": float(p_one),
        "p_stouffer_two_sided": float(p_two)
    }])
    overall_path = os.path.join(outdir, f"overall_stouffer_{args.decoder}_{args.dim_red}.csv")
    overall.to_csv(overall_path, index=False)
    print("Saved overall Stouffer summary:", overall_path)

    # Console summary
    sig_days = (per_day["p_two_sided_holm"] < 0.05).sum()
    direction = "aligned > direct" if Z > 0 else "direct > aligned"
    print("\n=== SUMMARY ===")
    print(f"Decoder: {args.decoder}  •  DimRed: {args.dim_red}")
    if args.exclude_channels:
        print(f"Excluded channels: {args.exclude_channels}")
    print(f"Days with paired data: {per_day.shape[0]}  (significant days after Holm: {sig_days})")
    print(f"Stouffer Z (one-sided, aligned>direct): {Z:.3f}  -> p_one={p_one:.3g}, p_two={p_two:.3g}  [{direction}]")

if __name__ == "__main__":
    main()
