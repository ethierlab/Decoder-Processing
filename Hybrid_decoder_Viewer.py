#!/usr/bin/env python3
"""
Full-proof heatmap visualizer for your EMG decoding results.

✓ Lit un fichier pickle unique OU un dossier contenant des fichiers:
     RESULT_<DECODER>_(pca|umap).pkl
  → tague automatiquement 'decoder_type' et 'manifold' si absents.

✓ Construit des heatmaps (type "confusion plot"):
    - axes configurables: Decoder/Task/Monkey
    - filtres: monkeys, tasks, decoders, dates
    - moyennes: jours, folds
    - modes musculaires:
        * scalar: un score par run (avec agrégation none/all/base)
        * per:    par canal → un facet par muscle_label (ou base)
        * one:    un muscle précis (ECR ou ECR_1)
    - facet: Monkey / Task / Decoder / manifold / muscle_label / muscle_base
    - export PNG + CSV

Exemples:
---------
# comparer PCA vs UMAP par décodeur×tâche
python viz_heatmap_results.py --input ./RESULTS_DIR \
  --rows Decoder --cols Task --avg-days --avg-folds \
  --facet-by manifold --annotate --vmin 0 --vmax 1 --out heatmaps/

# un heatmap par muscle (labels complets)
python viz_heatmap_results.py --input ./RESULTS_DIR \
  --channel-mode per --facet-by muscle_label \
  --rows Decoder --cols Task --avg-days --avg-folds \
  --out heatmaps_per_muscle/

# muscle précis (groupe de base ECR), facet manifold
python viz_heatmap_results.py --input ./RESULTS_DIR \
  --channel-mode one --muscle ECR \
  --rows Decoder --cols Task --avg-days --avg-folds \
  --facet-by manifold --annotate --out heatmaps_ECR/
"""
import argparse
import os
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Matplotlib defaults (sobre, lisible) ----
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ----------------------------- small helpers -----------------------------
def _canon_dim(name):
    if name is None: return None
    m = {"Monkey":"monkey","Task":"task","Decoder":"decoder_type"}
    return m.get(name, name)

def _dedupe(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out
def _norm(s):
    # Nettoie juste les espaces accidentels. Retire cette fonction si tu veux 0 normalisation.
    return s.strip() if isinstance(s, str) else s

def BASE_MUSCLE(name):
    return name.split("_")[0] if isinstance(name, str) else name


# ----------------------------- loading -----------------------------
def load_results_any(path: str) -> pd.DataFrame:
    """
    Read either a single pickle or a directory with RESULT_<DECODER>_(pca|umap).pkl files.
    Injects columns: 'manifold' (pca/umap from filename) and 'decoder_type' (from filename)
    when missing. Normalizes common fields and builds a scalar 'vaf' column.
    Skips non-matching or non-convertible pickles robustly.
    """
    import re
    import numpy as np
    import pandas as pd
    import os

    def _norm(s):  # keep your tiny string clean-up; remove if unwanted
        return s.strip() if isinstance(s, str) else s

    def _ensure_df(obj, src_name: str):
        """Try to convert various pickle payloads to a DataFrame."""
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, list):
            try:
                return pd.DataFrame(obj)
            except Exception:
                # print(f"[SKIP] {src_name}: list could not be converted to DataFrame")
                return None
        if isinstance(obj, dict):
            # First try dict-of-lists/arrays
            try:
                return pd.DataFrame(obj)
            except Exception:
                # Then try common keys holding the table
                for k in ("all_results", "results", "df", "data"):
                    if k in obj:
                        sub = obj[k]
                        if isinstance(sub, pd.DataFrame):
                            return sub
                        if isinstance(sub, list):
                            try:
                                return pd.DataFrame(sub)
                            except Exception:
                                pass
                print(f"[SKIP] {src_name}: dict not convertible to DataFrame")
                return None
        print(f"[SKIP] {src_name}: unsupported pickle type {type(obj)}")
        return None

    def _normalize_df(df: pd.DataFrame, dec_hint=None, man_hint=None) -> pd.DataFrame:
        for c in ["scenario_name","train_monkey","test_monkey","train_task","test_task","decoder_type","manifold"]:
            if c in df.columns:
                df[c] = df[c].map(_norm)
        if dec_hint and ("decoder_type" not in df.columns or df["decoder_type"].isna().all()):
            df["decoder_type"] = dec_hint
        if man_hint and ("manifold" not in df.columns or df["manifold"].isna().all()):
            df["manifold"] = man_hint
        # unify monkey/task
        if "test_monkey" in df.columns:
            df["monkey"] = df["test_monkey"].fillna(df.get("train_monkey"))
        if "test_task" in df.columns:
            df["task"] = df["test_task"].fillna(df.get("train_task"))
        # scalar vaf
        if "fold_mean_VAF" in df.columns:
            m = df["fold_mean_VAF"].notna()
            df.loc[m, "vaf"] = df.loc[m, "fold_mean_VAF"].astype(float)
        if "mean_VAF" in df.columns:
            m = df["mean_VAF"].notna()
            df.loc[m, "vaf"] = df.loc[m, "mean_VAF"].astype(float)
        # per-channel arrays
        if "per_channel_VAF" in df.columns:
            df["per_channel_VAF"] = df["per_channel_VAF"].apply(
                lambda x: np.asarray(x) if isinstance(x, (list, np.ndarray)) else np.array([])
            )
        if "emg_labels" in df.columns:
            df["emg_labels"] = df["emg_labels"].apply(
                lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
            )
        return df

    # ---- Directory mode: strictly load only RESULT_*_(pca|umap).pkl ----
    if os.path.isdir(path):
        rx = re.compile(r"RESULT_([^_]+)_(pca|umap)\.pkl$", re.IGNORECASE)
        frames = []
        for fname in os.listdir(path):
            m = rx.match(fname)
            if not m:
                # print(f"[SKIP] {fname}: does not match RESULT_<DECODER>_(pca|umap).pkl")
                continue
            dec_hint, man_hint = m.group(1), m.group(2).lower()
            fp = os.path.join(path, fname)
            try:
                obj = pd.read_pickle(fp)
            except Exception as e:
                print(f"[SKIP] {fp}: cannot read pickle ({e})")
                continue
            df_i = _ensure_df(obj, fname)
            if df_i is None:
                continue
            frames.append(_normalize_df(df_i, dec_hint, man_hint))
        if not frames:
            raise RuntimeError(f"No valid RESULT_*_(pca|umap).pkl files in {path}")
        return pd.concat(frames, ignore_index=True)

    # ---- Single file mode ----
    base = os.path.basename(path)
    rx_one = re.compile(r"RESULT_([^_]+)_(pca|umap)\.pkl$", re.IGNORECASE)
    dec_hint = man_hint = None
    m = rx_one.match(base)
    if m:
        dec_hint, man_hint = m.group(1), m.group(2).lower()

    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        raise RuntimeError(f"Cannot read pickle {path}: {e}")

    df = _ensure_df(obj, base)
    if df is None:
        raise RuntimeError(f"{path} is not a DataFrame and could not be converted")
    return _normalize_df(df, dec_hint, man_hint)



# ----------------------------- per-channel handling -----------------------------
def explode_per_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows so each muscle channel becomes a row.
    Adds: muscle_label (ECR_1), muscle_base (ECR), vaf_channel.
    Falls back to scalar vaf when per_channel arrays are missing.
    """
    rows = []
    for _, r in df.iterrows():
        v = r.get("per_channel_VAF")
        labs = r.get("emg_labels")
        if isinstance(v, np.ndarray) and v.size and labs:
            for score, lab in zip(v, labs):
                rr = r.copy()
                rr["muscle_label"] = str(lab)
                rr["muscle_base"] = BASE_MUSCLE(lab)
                rr["vaf_channel"] = float(score)
                rows.append(rr)
        else:
            rr = r.copy()
            rr["muscle_label"] = None
            rr["muscle_base"] = None
            rr["vaf_channel"] = float(r.get("vaf", np.nan))
            rows.append(rr)
    return pd.DataFrame(rows)


def muscle_reduce(row: pd.Series, mode: str) -> Optional[float]:
    """
    Aggregate VAF across muscles for scalar mode.
    mode in {"none","all","base"}
      none → use scalar vaf if present
      all  → mean over channels
      base → mean of muscle-base group means
    """
    if mode == "none":
        return float(row.get("vaf", np.nan))
    v = row.get("per_channel_VAF")
    labels = row.get("emg_labels")
    if not isinstance(v, np.ndarray) or v.size == 0 or not labels:
        return float(row.get("vaf", np.nan))
    v = v.astype(float)
    labels = list(labels)
    if mode == "all":
        return np.nanmean(v) if v.size else np.nan
    elif mode == "base":
        by_base: Dict[str, List[float]] = {}
        for score, lab in zip(v, labels):
            base = BASE_MUSCLE(lab)
            by_base.setdefault(base, []).append(score)
        group_means = [np.nanmean(vals) for vals in by_base.values() if len(vals) > 0]
        return float(np.nanmean(group_means)) if group_means else np.nan
    else:
        raise ValueError(f"Unknown muscle aggregation mode: {mode}")


# ----------------------------- matrix building & plotting -----------------------------
def _pivot(df: pd.DataFrame, rows: str, cols: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keymap = {"Decoder": "decoder_type", "Task": "task", "Monkey": "monkey"}
    rkey, ckey = keymap[rows], keymap[cols]
    raw = df[[rkey, ckey, "value"]].groupby([rkey, ckey]).mean().reset_index()
    mat = raw.pivot(index=rkey, columns=ckey, values="value").sort_index(axis=0).sort_index(axis=1)
    return mat, raw

def _facet_and_pivot(agg: pd.DataFrame, rows: str, cols: str, facet_by: Optional[str]):
    if facet_by and facet_by in agg.columns:
        facets = sorted([str(x) for x in agg[facet_by].dropna().unique()]) or ["ALL"]
        facet_map, raw_map = {}, {}
        for f in facets:
            sub = agg[agg[facet_by] == f].copy()
            mat, raw = _pivot(sub, rows, cols)
            facet_map[f] = mat
            raw_map[f] = raw
        return facet_map, raw_map
    else:
        mat, raw = _pivot(agg, rows, cols)
        return {"ALL": mat}, {"ALL": raw}

def build_matrix(
    df: pd.DataFrame,
    rows: str,
    cols: str,
    monkey: List[str],
    task: List[str],
    decoder: List[str],
    date_from: Optional[str],
    date_to: Optional[str],
    muscle_agg: str,
    avg_days: bool,
    facet_by: Optional[str],
    channel_mode: str,
    muscle_filter: Optional[str],
):
    keep = df.copy()
    facet_by = _canon_dim(facet_by)
    # Date filter
    if "date" in keep.columns:
        if date_from: keep = keep[keep["date"] >= pd.to_datetime(date_from)]
        if date_to:   keep = keep[keep["date"] <= pd.to_datetime(date_to)]
    # Filters
    if monkey and monkey != ["ALL"]:
        keep = keep[keep["monkey"].isin(monkey)]
    if task and task != ["ALL"]:
        keep = keep[keep["task"].isin(task)]
    if decoder and decoder != ["ALL"]:
        keep = keep[keep["decoder_type"].isin(decoder)]

    if channel_mode == "scalar":
        keep = keep.copy()
        keep["value"] = keep.apply(lambda r: muscle_reduce(r, muscle_agg), axis=1)
        group_keys = ["monkey","task","decoder_type"]
        if not avg_days and "date" in keep.columns: group_keys.append("date")
        if facet_by and facet_by in keep.columns and facet_by not in group_keys: group_keys.append(facet_by)
        group_keys = _dedupe(group_keys)
        agg = keep.groupby(group_keys, dropna=False)["value"].mean().reset_index()

        return _facet_and_pivot(agg, rows, cols, facet_by)

    # per-channel modes
    per = explode_per_channel(keep)
    if channel_mode == "one" and muscle_filter:
        per = per[(per.muscle_label == muscle_filter) | (per.muscle_base == muscle_filter)]
    per = per.rename(columns={"vaf_channel": "value"})
    group_keys = ["monkey","task","decoder_type"]
    if not avg_days and "date" in per.columns: group_keys.append("date")
    if channel_mode == "per" and (facet_by is None):
        facet_by = "muscle_label"
    if facet_by and facet_by in per.columns and facet_by not in group_keys: group_keys.append(facet_by)
    group_keys = _dedupe(group_keys)
    agg = per.groupby(group_keys, dropna=False)["value"].mean().reset_index()
    return _facet_and_pivot(agg, rows, cols, facet_by)

def plot_heatmap(matrix: pd.DataFrame, title: str, out_png: Optional[str],
                 annotate: bool, vmin: Optional[float], vmax: Optional[float]):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=0)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(matrix.columns.name or "Test Task")
    ax.set_ylabel(matrix.index.name or "Decoder")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean_VAF")
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close(fig)


# ----------------------------- CLI -----------------------------
def main():
    p = argparse.ArgumentParser(description="Heatmap visualizer for EMG decoding results")
    p.add_argument("--input", required=True,
                   help="Path to a results pickle OR a directory containing RESULT_*_(pca|umap).pkl files")
    p.add_argument("--rows", default="Decoder", choices=["Decoder","Task","Monkey"], help="Row dimension")
    p.add_argument("--cols", default="Task", choices=["Decoder","Task","Monkey"], help="Column dimension")
    p.add_argument("--monkey", nargs="+", default=["ALL"], help="Monkeys to include (names or ALL)")
    p.add_argument("--task", nargs="+", default=["ALL"], help="Tasks to include (names or ALL)")
    p.add_argument("--decoder", nargs="+", default=["ALL"], help="Decoders to include (names or ALL)")
    p.add_argument("--date-from", default=None)
    p.add_argument("--date-to", default=None)

    p.add_argument("--channel-mode", default="scalar", choices=["scalar","per","one"],
                   help="scalar: one value per run; per: per-muscle facet; one: single muscle only")
    p.add_argument("--muscle-agg", default="none", choices=["none","all","base"],
                   help="Aggregation across channels for scalar mode")
    p.add_argument("--muscle", default=None, help="Muscle filter (ECR, ECR_1, ...) when --channel-mode one")

    p.add_argument("--avg-days", action="store_true", help="Average across days")
    p.add_argument("--facet-by", default=None,
                   choices=[None,"Monkey","Task","Decoder","manifold","muscle_label","muscle_base"],
                   help="Create one heatmap per facet")
    p.add_argument("--annotate", action="store_true", help="Write values on cells")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--out", default=None, help="Path to save PNG (or directory if multiple facets)")
    p.add_argument("--save-csv", default=None, help="Also save the numeric matrix to CSV")
    p.add_argument("--monkeygrid", action="store_true",
                   help="Convenience: one figure per monkey (rows=Decoder, cols=Task)")

    args = p.parse_args()

    df = load_results_any(args.input)
    print("[INFO] Loaded:", df.shape, "rows")
    if "manifold" in df.columns:
        print("[INFO] Manifold counts:\n", df["manifold"].value_counts(dropna=False))
    if "decoder_type" in df.columns:
        print("[INFO] Decoder counts:\n", df["decoder_type"].value_counts(dropna=False).sort_index())
    # ---- SPECIAL: single-muscle for ALL muscles ----
    if args.channel_mode == "one" and args.muscle and args.muscle.upper() in ("ALL", "*"):
        per = explode_per_channel(load_results_any(args.input))
        muscles = sorted([m for m in per["muscle_label"].dropna().unique()])
        outdir = args.out or "heatmaps_single_muscle"
        os.makedirs(outdir, exist_ok=True)
        for m in muscles:
            mats, _ = build_matrix(
                df,
                rows=args.rows, cols=args.cols,
                monkey=args.monkey, task=args.task, decoder=args.decoder,
                date_from=args.date_from, date_to=args.date_to,
                muscle_agg=args.muscle_agg, avg_days=args.avg_days,
                facet_by=args.facet_by,
                channel_mode="one", muscle_filter=m,
            )
            # save 1 or many facets per muscle
            if len(mats) == 1:
                mat = next(iter(mats.values()))
                title = f"{m} | {args.rows}×{args.cols}"
                fname = f"heatmap_{m.replace('/','-')}.png"
                plot_heatmap(mat, title, os.path.join(outdir, fname), args.annotate, args.vmin, args.vmax)
                if args.save_csv:
                    mat.to_csv(os.path.join(outdir, fname.replace(".png",".csv")))
            else:
                for facet, mat in mats.items():
                    title = f"{m} | {args.facet_by}={facet} | {args.rows}×{args.cols}"
                    fname = f"heatmap_{m.replace('/','-')}_{args.facet_by}_{str(facet).replace(' ','_')}.png"
                    plot_heatmap(mat, title, os.path.join(outdir, fname), args.annotate, args.vmin, args.vmax)
                    if args.save_csv:
                        mat.to_csv(os.path.join(outdir, fname.replace(".png",".csv")))
        return

    # Convenience grid: one PNG per monkey
    if args.monkeygrid:
        monkeys = sorted(df["monkey"].dropna().unique())
        outdir = args.out or "heatmaps"
        for m in monkeys:
            mats, _ = build_matrix(
                df, rows="Decoder", cols="Task",
                monkey=[m], task=["ALL"], decoder=["ALL"],
                date_from=args.date_from, date_to=args.date_to,
                muscle_agg=args.muscle_agg, avg_days=True,
                facet_by=None, channel_mode="scalar", muscle_filter=None,
            )
            mat = mats["ALL"].copy()
            title = f"{m}: mean VAF (rows=Decoder, cols=Task)"
            out_png = os.path.join(outdir, f"heatmap_{m}.png")
            plot_heatmap(mat, title, out_png, args.annotate, args.vmin, args.vmax)
            if args.save_csv:
                os.makedirs(outdir, exist_ok=True)
                mat.to_csv(os.path.join(outdir, f"heatmap_{m}.csv"))
        return

    # General path
    mats, raws = build_matrix(
        df,
        rows=args.rows, cols=args.cols,
        monkey=args.monkey, task=args.task, decoder=args.decoder,
        date_from=args.date_from, date_to=args.date_to,
        muscle_agg=args.muscle_agg, avg_days=args.avg_days,
        facet_by=args.facet_by, channel_mode=args.channel_mode,
        muscle_filter=args.muscle,
    )

    # Plot facets
    if len(mats) == 1:
        title = f"Heatmap ({args.rows}×{args.cols})"
        if args.facet_by: title += f" | {args.facet_by}=ALL"
        out_png = args.out
        plot_heatmap(next(iter(mats.values())), title, out_png, args.annotate, args.vmin, args.vmax)
        if args.save_csv:
            next(iter(mats.values())).to_csv(args.save_csv)
    else:
        outdir = args.out or "heatmaps"
        os.makedirs(outdir, exist_ok=True)
        for facet, mat in mats.items():
            title = f"{args.facet_by}={facet} | {args.rows}×{args.cols}"
            out_png = os.path.join(outdir, f"heatmap_{args.facet_by}_{str(facet).replace(' ', '_')}.png")
            plot_heatmap(mat, title, out_png, args.annotate, args.vmin, args.vmax)
            if args.save_csv:
                mat.to_csv(os.path.join(outdir, f"heatmap_{args.facet_by}_{str(facet).replace(' ', '_')}.csv"))

if __name__ == "__main__":
    main()
