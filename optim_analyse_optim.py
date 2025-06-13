import pickle, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LogFormatterSciNotation

ROOT_PKL  = "gridsearch_results.pkl"   # pick ONE file
TOP_K     = 5
FIGSIZE   = (12, 8)
LEGEND_FNT = 8

# ─── load
rows = pickle.load(open(ROOT_PKL, "rb"))
df   = pd.DataFrame(rows)

# ─── drop failed runs
df = df[df.mean_vaf >= 0].copy()

# ─── aggregate across seeds
agg = (df.groupby(["decoder","n_pca","k_lag",
                   "hidden_dim","num_epochs","lr","num_params"],
                  as_index=False)
         .agg(mean_vaf  = ("mean_vaf","mean"),
              std_vaf   = ("mean_vaf","std"),
              mean_time = ("mean_time","mean"),
              n_seeds   = ("seed","nunique")))

# ─── TOP-K per decoder
topk = (agg.sort_values(["decoder","mean_vaf"],
                        ascending=[True,False])
            .groupby("decoder")
            .head(TOP_K))

print("\n# ===== BEST CONFIGS (mean over seeds) =====")
for _, r in topk.iterrows():
    print(f"[{r.decoder.upper():6}] VAF={r.mean_vaf:6.3f}±{r.std_vaf:4.3f} "
          f"| hid={r.hidden_dim:3} | n_pca={r.n_pca:2} | k_lag={r.k_lag:2} "
          f"| lr={r.lr:.1e} | epochs={r.num_epochs:3} "
          f"| params={r.num_params:,} | seeds={r.n_seeds}")

# ─── save JSON
best_cfg = (topk
            .drop(columns=["mean_vaf","std_vaf","n_seeds"])
            .groupby("decoder")
            .apply(lambda g: g.to_dict(orient="records"))
            .to_dict())
Path("best_gridsearch_hparams.json").write_text(json.dumps(best_cfg, indent=2))

# ─── plotting helpers
sns.set_context("talk")
def scatter(xcol, fname, xlab, logx=False):
    plt.figure(figsize=FIGSIZE)
    sns.scatterplot(data=agg, x=xcol, y="mean_vaf",
                    hue="decoder", style="decoder", s=90, alpha=.75)
    for _, row in agg.iterrows():
        if not np.isnan(row.std_vaf):
            plt.errorbar(row[xcol], row.mean_vaf,
                         yerr=row.std_vaf, fmt="none",
                         ecolor="gray", capsize=2, alpha=.6, lw=.8)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlab); plt.ylabel("VAF (mean over seeds)")
    plt.title(f"{xlab} vs VAF")
    plt.legend(title="Decoder", fontsize=LEGEND_FNT,
               title_fontsize=LEGEND_FNT, frameon=False)
    plt.tight_layout(); plt.savefig(fname, dpi=450); plt.close()

def scatter_3d(fname="scatter_3d_vaf_params_time.png"):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    decoders = agg.decoder.unique()
    markers  = ["o", "s", "^", "D", "P", "X"]
    palette  = sns.color_palette("tab10", len(decoders))

    for dec, mk, col in zip(decoders, markers, palette):
        sub = agg[agg.decoder == dec]
        ax.scatter(sub.num_params, sub.mean_time, sub.mean_vaf,
                   marker=mk, s=60, color=col, alpha=.8, label=dec.upper())

    ax.set_xlabel("# parameters", labelpad=10)
    ax.set_ylabel("train time / fold (s)", labelpad=10)
    ax.set_zlabel("mean VAF", labelpad=10)
    ax.set_title("VAF vs Complexity vs Training Time", pad=15)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(fname, dpi=450)
    plt.close()

scatter_3d()
scatter("hidden_dim", "scatter_hidden_dim.png", "hidden_dim")
scatter("num_epochs", "scatter_epochs.png",   "training epochs")
scatter("num_params", "scatter_params.png",
        "# trainable parameters (log-scale)", logx=True)
print("[INFO] plots written.")
