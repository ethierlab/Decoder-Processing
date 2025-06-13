from __future__ import annotations
import argparse, itertools, json, pickle
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# ────────────────────────── GRID (same as training script) ──────────────────
GRID: Dict[str, Dict[str, List[Any]]] = {
    "gru": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "lstm": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "ligru": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [4, 8, 16, 32, 64, 96, 128],
        num_epochs = [100, 200, 300],
        lr         = [1e-3, 3e-3],
    ),
    "linear": dict(
        n_pca      = [8, 16, 24, 32],
        k_lag      = [5, 10, 15, 20, 25],
        hidden_dim = [32, 64, 128, 192, 256],
        num_epochs = [50, 100, 150],
        lr         = [1e-3, 1e-2],
    ),
}

# ───────────────────── helper: Cartesian product generator ──────────────────
def cartesian_product(param_dict: Dict[str, List[Any]]):
    keys, vals = zip(*param_dict.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

# ────────────────────────────────── main ─────────────────────────────────────
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results",  required=True, help="pickle produced by grid search")
    p.add_argument("--seeds",    type=int, required=True, help="number of seeds per config")
    p.add_argument("--decoders", nargs="+", required=True,
                   help="subset of decoders you actually ran")
    p.add_argument("--json_out", default="missing_runs.json",
                   help="file to save missing (decoder, cfg, seed) tuples")
    args = p.parse_args()

    # ─── load completed runs ────────────────────────────────────────────────
    rows = pickle.load(open(args.results, "rb"))
    df   = pd.DataFrame(rows)

    done_keys = {(r.decoder, r.n_pca, r.k_lag,
                  r.hidden_dim, r.num_epochs, r.lr, r.seed)
                 for r in df.itertuples()}

    missing = []  # collect missing runs

    print("\n# ===== GRID-SEARCH PROGRESS =====")
    for dec in args.decoders:
        grid = list(cartesian_product(GRID[dec]))
        total = len(grid) * args.seeds
        done  = 0

        for cfg in grid:
            for seed in range(args.seeds):
                key = (dec, cfg["n_pca"], cfg["k_lag"],
                       cfg["hidden_dim"], cfg["num_epochs"], cfg["lr"], seed)
                if key in done_keys:
                    done += 1
                else:
                    missing.append(dict(decoder=dec, seed=seed, **cfg))

        pct = 100.0 * done / total if total else 0
        print(f"[{dec.upper():6}]  {done:4}/{total:<4}  "
              f"({pct:6.2f} %)   missing: {total-done}")

    # ─── optional: write missing runs to JSON ───────────────────────────────
    if missing:
        Path(args.json_out).write_text(json.dumps(missing, indent=2))
        print(f"\nMissing runs written to {args.json_out}")
    else:
        print("\n🎉  All requested runs are complete!")

if __name__ == "__main__":
    main()