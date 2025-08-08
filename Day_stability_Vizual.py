import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cherche tous les fichiers .pkl
# Modifie 'results_dir' si tes fichiers sont ailleurs !
results_dir = "./"
pkl_files = sorted(glob.glob(os.path.join(results_dir, "crossval_results_GRU_*.pkl")))
print(f"Trouvé {len(pkl_files)} fichiers .pkl")

# 2. Concatène tous les DataFrame
dfs = []
for pkl in pkl_files:
    try:
        df = pd.read_pickle(pkl)
        print(f" - {pkl} : {df.shape}")
        dfs.append(df)
    except Exception as e:
        print(f"Erreur lors de la lecture de {pkl}: {e}")
        continue

if not dfs:
    raise RuntimeError("Aucun fichier .pkl n'a pu être chargé!")

df_all = pd.concat(dfs, ignore_index=True)
print("Shape finale du DataFrame:", df_all.shape)

# 3. Plots comme dans le code original

for dim_red in df_all["dim_red"].unique():
    plt.figure(figsize=(10,6))
    for dec in ["GRU","LSTM","Linear","LiGRU"]:
        sub = df_all[(df_all['decoder']==dec)&(df_all['dim_red']==dim_red)]
        means = sub.groupby("day_int")["vaf"].mean()
        stds = sub.groupby("day_int")["vaf"].std()
        plt.errorbar(means.index, means.values, yerr=stds.values, label=f"{dec} ({dim_red})")
    plt.legend()
    plt.xlabel("Days from day0")
    plt.ylabel("Mean VAF (crossval)")
    plt.title(f"VAF par jour/décodeur ({dim_red})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4. Boxplot pooled (optionnel)
sns.boxplot(data=df_all, x="day_int", y="vaf", hue="dim_red")
plt.title("VAF tous décodeurs, tous jours, par méthode de dim red")
plt.show()
