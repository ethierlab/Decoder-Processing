import pickle
from pathlib import Path
# import pandas as pd
result_dir = Path('C:/Users/Vincent/Downloads/Result_optim')  # Dossier où tu as mis tous tes fichiers .pkl
all_results = []

for pkl_file in sorted(result_dir.glob('gridsearch_results_*.pkl')):
    with open(pkl_file, 'rb') as f:
        try:
            data = pickle.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
        except Exception as e:
            print(f"Erreur lecture {pkl_file}: {e}")

print(f"Fusion terminé! {len(all_results)} runs.")

with open('ALL_gridsearch_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

