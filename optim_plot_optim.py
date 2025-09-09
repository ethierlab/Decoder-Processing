import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

# ========= PARAMS =========
PKL_PATH = "ALL_gridsearch_results_early.pkl"
EXCLUDE_CHANNELS = {0, 5, 6}   # canaux à ignorer pour l'avg "across muscles"
DECODER_ORDER = ['ligru', 'lstm', 'gru', 'linear']  # pour l'impression des tops
TOPN = 50                      # pour le boxplot top configs
# ==========================

with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)
df = pd.DataFrame(rows)

# --- SANITIZE df ---
need = ["decoder", "mean_vaf", "num_params", "fold_vafs"]
for c in need:
    if c not in df.columns:
        df[c] = np.nan

# numériques robustes
df["mean_vaf"]   = pd.to_numeric(df["mean_vaf"], errors="coerce")
df["num_params"] = pd.to_numeric(df["num_params"], errors="coerce")

# trouver colonne de canal
CAND_CHANNEL_COLS = ['emg_channel','channel','muscle','emg','muscle_idx','chn','ch','chan']
chan_col = next((c for c in CAND_CHANNEL_COLS if c in df.columns), None)
if chan_col is None:
    raise ValueError(
        f"Aucune colonne de canal trouvée. Ajoute une des colonnes {CAND_CHANNEL_COLS} "
        "dans tes résultats pour pouvoir moyenner sur les muscles."
    )

# normaliser -> 'emg_channel' (entier nullable)
df["emg_channel"] = pd.to_numeric(df[chan_col], errors="coerce").astype("Int64")

# filtre runs valides
df = df.dropna(subset=["decoder", "mean_vaf", "num_params"])
df = df[df["num_params"] > 0]

# exclure canaux faibles
df = df[~df["emg_channel"].isin(EXCLUDE_CHANNELS)]

# ============ FLATTEN FOLDS (en gardant le canal) ============
flat = []
for _, row in df.iterrows():
    vafs = row["fold_vafs"] if isinstance(row["fold_vafs"], (list, tuple, np.ndarray)) else []
    for fold_num, v in enumerate(vafs):
        flat.append({
            "decoder":      row["decoder"],
            "num_params":   float(row["num_params"]),
            "hidden_dim":   row.get("hidden_dim", None),
            "k_lag":        row.get("k_lag", None),
            "n_pca":        row.get("n_pca", None),
            "lr":           row.get("lr", None),
            "num_epochs":   row.get("num_epochs", None),
            "seed":         row.get("seed", None),
            "emg_channel":  row.get("emg_channel", pd.NA),
            "fold_num":     fold_num,
            "vaf":          float(v),
        })

flatdf = pd.DataFrame(flat)
flatdf["vaf"]          = pd.to_numeric(flatdf["vaf"], errors="coerce")
flatdf["num_params"]   = pd.to_numeric(flatdf["num_params"], errors="coerce")
flatdf["emg_channel"]  = pd.to_numeric(flatdf["emg_channel"], errors="coerce").astype("Int64")
flatdf = flatdf.dropna(subset=["decoder", "vaf", "num_params"])
flatdf = flatdf[(flatdf["num_params"] > 0) & (flatdf["vaf"].between(-1, 1))]
flatdf = flatdf[~flatdf["emg_channel"].isin(EXCLUDE_CHANNELS)]

# ============ MOYENNES "ACROSS MUSCLES" ============
# Clés d'une config (sans le canal !)
CONFIG_KEYS = ['decoder','num_params','hidden_dim','k_lag','n_pca','lr','num_epochs','seed']
CONFIG_KEYS = [k for k in CONFIG_KEYS if k in df.columns]

# 1) Moyenne sur muscles pour la métrique "mean_vaf" (déjà moyenne sur folds)
df_avg = (
    df.groupby(CONFIG_KEYS, dropna=False)['mean_vaf']
      .mean()
      .reset_index()
      .rename(columns={'mean_vaf': 'mean_vaf_avg_muscles'})
)
# pour info: nb de muscles agrégés par config
n_muscles_per_cfg = (
    df.groupby(CONFIG_KEYS, dropna=False)['emg_channel']
      .nunique()
      .reset_index()
      .rename(columns={'emg_channel': 'n_muscles'})
)
df_avg = df_avg.merge(n_muscles_per_cfg, on=CONFIG_KEYS, how='left')

# 2) Moyenne sur muscles au niveau "fold" (distributions)
FOLD_KEYS = CONFIG_KEYS + ['fold_num']
FOLD_KEYS = [k for k in FOLD_KEYS if k in flatdf.columns]
flat_avg = (
    flatdf.groupby(FOLD_KEYS, dropna=False)['vaf']
          .mean()
          .reset_index()
          .rename(columns={'vaf': 'vaf_avg_muscles'})
)
# info: nb muscles par (config, fold)
n_muscles_fold = (
    flatdf.groupby(FOLD_KEYS, dropna=False)['emg_channel']
          .nunique()
          .reset_index()
          .rename(columns={'emg_channel': 'n_muscles'})
)
flat_avg = flat_avg.merge(n_muscles_fold, on=FOLD_KEYS, how='left')

# ========= DIAGNOSTICS =========
print(df_avg[['decoder','mean_vaf_avg_muscles']].dtypes)
print(df_avg[['decoder','mean_vaf_avg_muscles','n_muscles']].head(5))
print(flat_avg[['decoder','vaf_avg_muscles']].dtypes)

# ======== TOP (moyenné muscles) ========
print("\n# ==== TOP per decoder (by mean VAF averaged across muscles) ====")
for decoder in DECODER_ORDER:
    subdf = df_avg[df_avg.decoder == decoder]
    if subdf.empty:
        continue
    top = subdf.sort_values("mean_vaf_avg_muscles", ascending=False).head(20)
    print(f"\n## {decoder.upper()} ##")
    for _, r in top.iterrows():
        print(
            f"VAFµ={r['mean_vaf_avg_muscles']:.3f} | hid={r.get('hidden_dim','-'):<3} | "
            f"n_pca={r.get('n_pca','-'):<2} | k_lag={r.get('k_lag','-'):<2} | "
            f"lr={r.get('lr','-'):<8} | epochs={r.get('num_epochs','-'):<3} | "
            f"params={r.get('num_params','-'):<7} | seed={r.get('seed','-')} | "
            f"#muscles={r.get('n_muscles','-')}"
        )
print("\n# ===== END TOP PRINT =====\n")

# ========= SCATTER (avg muscles) =========
plt.figure(figsize=(13, 7))
sns.scatterplot(
    data=df_avg, x="num_params", y="mean_vaf_avg_muscles", hue="decoder",
    palette="tab10", alpha=0.7, s=60, edgecolor=None
)
plt.xscale('log')
plt.xlabel("Number of trainable parameters (log)")
plt.ylabel("mean VAF (avg across muscles)")
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig("scatter_vaf_vs_params_avg_muscles.png", dpi=350)
plt.show()

# ========= VIOLIN (folds, avg muscles) =========
plt.figure(figsize=(10, 6))
sns.violinplot(data=flat_avg, x="decoder", y="vaf_avg_muscles", inner="point")
plt.ylim(0, 1.2)
plt.title("Distribution des VAFs par décodeur (moyenne across muscles, tous folds)")
plt.ylabel("VAF par fold (avg muscles)")
plt.xlabel("Décodeur")
plt.tight_layout()
plt.savefig("violin_vaf_folds_by_decoder_avg_muscles.png", dpi=300)
plt.show()

# ========= BOXPLOT: TOP N CONFIGS (avg muscles) =========
top = (
    flat_avg.groupby(['decoder', 'num_params'], dropna=False)
            .agg(mean_vaf=('vaf_avg_muscles', 'mean'), count=('vaf_avg_muscles', 'size'))
            .reset_index()
            .sort_values(['decoder', 'mean_vaf'], ascending=[True, False])
)

keep_tuples = []
for dec in flat_avg['decoder'].dropna().unique():
    keep_tuples += (
        top[top['decoder'] == dec]
        .head(TOPN)
        [['decoder', 'num_params']]
        .apply(tuple, axis=1)
        .tolist()
    )

flatdf_plot = flat_avg[
    flat_avg[['decoder', 'num_params']]
    .apply(tuple, axis=1)
    .isin(keep_tuples)
]

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=flatdf_plot,
    x="num_params",
    y="vaf_avg_muscles",
    hue="decoder",
    palette="tab10",
    showmeans=True
)
plt.xlabel("Nombre de paramètres")
plt.ylabel("VAF (avg muscles • tous folds/seeds)")
plt.title(f"Distribution des VAFs par nb de paramètres et par décodeur (top {TOPN} configs/décodeur, avg muscles)")
plt.legend(title="Décodeur", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig("boxplot_folds_by_numparams_decoder_avg_muscles.png", dpi=350)
plt.show()
