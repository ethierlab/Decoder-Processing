import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

with open("ALL_gridsearch_results.pkl", "rb") as f:
    rows = pickle.load(f)
df = pd.DataFrame(rows)

plt.figure(figsize=(14, 8))

# On "flatten" tous les folds en une seule colonne avec le nom du décodeur associé
flat = []
for i, row in df.iterrows():
    if "num_params" not in row or "decoder" not in row:
        continue  # skip les runs malformés
    vafs = row.get("fold_vafs", [])
    for fold_num, vaf in enumerate(vafs):
        flat.append({
            "decoder": row["decoder"],
            "num_params": row["num_params"],
            "hidden_dim": row.get("hidden_dim", None),
            "k_lag": row.get("k_lag", None),
            "n_pca": row.get("n_pca", None),
            "lr": row.get("lr", None),
            "num_epochs": row.get("num_epochs", None),
            "fold_num": fold_num,
            "vaf": float(vaf)
        })
flatdf = pd.DataFrame(flat)

# ======== PRINT TOP 5 PER DECODER ========
print("\n# ==== TOP 5 per decoder (by mean VAF) ====")
for decoder in ['ligru', 'lstm', 'gru', 'linear']:
    subdf = df[df.decoder == decoder]
    if subdf.empty:
        continue
    top = subdf.sort_values("mean_vaf", ascending=False).head(20)
    print(f"\n## {decoder.upper()} ##")
    for i, r in top.iterrows():
        print(f"VAF={r['mean_vaf']:.3f} | hid={r.get('hidden_dim', '-'):<3} | n_pca={r.get('n_pca', '-'):<2} | "
              f"k_lag={r.get('k_lag', '-'):<2} | lr={r.get('lr', '-'):<6} | epochs={r.get('num_epochs', '-'):<3} "
              f"| params={r.get('num_params', '-'):<5} | seed={r.get('seed', '-')}")
print("\n# ===== END TOP 5 PRINT =====\n")

# ========= SIMPLE SCATTERPLOT: ALL RUNS =========
plt.figure(figsize=(13, 7))
sns.scatterplot(
    data=df, x="num_params", y="mean_vaf", hue="decoder",
    palette="tab10", alpha=0.7, s=60, edgecolor=None
)
plt.xscale('log')
plt.xlabel("Nombre de paramètres (log)")
plt.ylabel("VAF moyen (sur les folds)")
plt.title("Tous les runs: VAF vs nombre de paramètres")
plt.tight_layout()
plt.savefig("scatter_vaf_vs_params_allpoints.png", dpi=350)
plt.show()

# ========= VIOLIN PLOT =========
sns.violinplot(data=flatdf, x="decoder", y="vaf", inner="point")
plt.title("Distribution des VAFs (tous folds, tous runs) par décodeur")
plt.ylabel("VAF par fold")
plt.xlabel("Décodeur")
plt.tight_layout()
plt.savefig("violin_vaf_folds_by_decoder.png", dpi=300)
plt.show()

# ========= BOXPLOT: TOP N CONFIGS =========
topN = 50
top = (
    flatdf.groupby(['decoder', 'num_params'])
          .agg(mean_vaf=('vaf', 'mean'), count=('vaf', 'size'))
          .reset_index()
          .sort_values(['decoder', 'mean_vaf'], ascending=[True, False])
)

keep_tuples = []
for dec in flatdf['decoder'].unique():
    keep_tuples += (
        top[top['decoder'] == dec]
        .head(topN)
        [['decoder', 'num_params']]
        .apply(tuple, axis=1)
        .tolist()
    )
flatdf_plot = flatdf[
    flatdf[['decoder', 'num_params']]
    .apply(tuple, axis=1)
    .isin(keep_tuples)
]

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=flatdf_plot,
    x="num_params",
    y="vaf",
    hue="decoder",
    palette="tab10",
    showmeans=True
)
plt.xlabel("Nombre de paramètres")
plt.ylabel("VAF (tous folds/tous seeds)")
plt.title(f"Distribution des VAFs par nombre de paramètres et par décodeur (top {topN} configs/décodeur)")
plt.legend(title="Décodeur", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig("boxplot_folds_by_numparams_decoder.png", dpi=350)
plt.show()
