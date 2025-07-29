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
    dec = row.get("decoder", "??")
    vafs = row.get("fold_vafs", [])
    for fold_num, vaf in enumerate(vafs):
        flat.append({"decoder": dec, "fold_num": fold_num, "vaf": float(vaf)})

flatdf = pd.DataFrame(flat)

sns.violinplot(data=flatdf, x="decoder", y="vaf", inner="point")
plt.title("Distribution des VAFs (tous folds, tous runs) par décodeur")
plt.ylabel("VAF par fold")
plt.xlabel("Décodeur")
plt.tight_layout()
plt.savefig("violin_vaf_folds_by_decoder.png", dpi=300)
plt.show()


# Example: scatter plot all individual fold VAFs by hidden_dim for a decoder
decoder_of_interest = "gru"
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=flatdf[flatdf.decoder == decoder_of_interest],
    x="fold_num",
    y="vaf",
    alpha=0.6
)
plt.title(f"VAF par fold pour tous les runs ({decoder_of_interest})")
plt.xlabel("Numéro de fold")
plt.ylabel("VAF")
plt.tight_layout()
plt.show()
