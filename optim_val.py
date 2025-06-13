import os, json, pickle
import optuna                                         # needed only to read the study
from pathlib import Path

OUTDIR = "optuna_out_VAF"        #  same as in the optimisation script
DECODERS = ["GRU", "LSTM", "Linear", "LiGRU"]
TOP_K   = 5                  #  how many best trials you want

best_cfgs = {}               #  decoder  →  [ {params…}, … ]
best_full = {}               #  same but includes seed & param‑count

for dec in DECODERS:
    study_pkl = Path(OUTDIR) / f"{dec}_study.pkl"
    if not study_pkl.exists():
        print(f"[WARN] {study_pkl} not found – skipped.") ; continue

    with study_pkl.open("rb") as f:
        study = pickle.load(f)

    # keep only completed trials, sort by objective (ascending)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value)[:TOP_K]

    best_cfgs[dec] = [t.params for t in top]
    best_full[dec] = [
        {
            "params"      : t.params,
            "objective"   : t.value,
            "param_count" : t.user_attrs.get("param_count"),
            "trial_seed"  : t.user_attrs.get("trial_seed"),
            "run_id"      : t.user_attrs.get("run_id"),
        }
        for t in top
    ]

# -------- pretty print to console -------------------------------------------
print("\n# ========= COPY/PASTE BELOW INTO YOUR TRAIN SCRIPT ========== #")
print("BEST_HPARAMS = ")
print(json.dumps(best_cfgs, indent=2))
print("# ============================================================= #\n")

# -------- save full details --------------------------------------
with open(Path(OUTDIR) / "top_trials_summary.json", "w") as f:
    json.dump(best_full, f, indent=2)
print(f"[INFO] Wrote detailed summary to {OUTDIR}/top_trials_summary.json")