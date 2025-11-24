import pickle
import re
from pathlib import Path
import calendar

result_dir = Path('C:/Users/Vincent/Downloads/train/optim/')  # dossier des .pkl

# ---- pick your month(s) to EXCLUDE here ----
exclude_months_input = ['08']   # accepts: 8, '08', 'Aug', 'August', 'août', 'aout', etc.
# -------------------------------------------

# Build a lookup for month names/abbr (EN) + a couple FR variants
MONTH_LOOKUP = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_LOOKUP.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTH_LOOKUP.update({'août': 8, 'aout': 8})

def to_month_int(x):
    if isinstance(x, int):
        if 1 <= x <= 12:
            return x
        raise ValueError(f"Invalid month int: {x}")
    s = str(x).strip().lower()
    if s.isdigit():
        m = int(s)
        if 1 <= m <= 12:
            return m
        raise ValueError(f"Invalid month number: {x}")
    if s in MONTH_LOOKUP:
        return MONTH_LOOKUP[s]
    raise ValueError(f"Unknown month label: {x}")

exclude_months = {to_month_int(m) for m in exclude_months_input}

all_results = []
all_pkls = sorted(result_dir.glob('gridsearch_results_*_*.pkl'))

def extract_month_from_name(name: str):
    """Return month (1-12) parsed from first YYYYMMDD in filename; None if not found."""
    m = re.search(r'(\d{8})', name)  # find YYYYMMDD
    if not m:
        return None
    yyyymmdd = m.group(1)
    return int(yyyymmdd[4:6])

selected_pkls, skipped_pkls = [], []
for p in all_pkls:
    mon = extract_month_from_name(p.name)
    if mon is not None and mon in exclude_months:
        skipped_pkls.append(p)
    else:
        selected_pkls.append(p)

for pkl_file in selected_pkls:
    with open(pkl_file, 'rb') as f:
        try:
            data = pickle.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
        except Exception as e:
            print(f"Erreur lecture {pkl_file}: {e}")

print(f"Fichiers pris: {len(selected_pkls)} | ignorés (mois exclus): {len(skipped_pkls)}")
if skipped_pkls:
    for p in skipped_pkls:
        pass
        # print(f"- {p.name}")

print(f"Fusion terminée! {len(all_results)} runs.")

with open('ALL_gridsearch_results_1.pkl', 'wb') as f:
    pickle.dump(all_results, f)
