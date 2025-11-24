# merge_and_plot_xtask_heatmaps.py  (with validation diagonals injected)
import os, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("Merge EMG cross-task .pkl results and plot heatmaps.")
    ap.add_argument("--root", required=True, help="Folder with strict_*.pkl files (recurses).")
    ap.add_argument("--outdir", default="./plots_xtask1", help="Where to save PNGs.")
    ap.add_argument("--vmin", type=float, default=-0.2)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

# ---------- helpers ----------
TASK_FIX = {"mg-pt":"mgpt"}
PAIR_TASKS = {
    ("Jango",): ["iso","wm","spr"],
    ("JacB",):  ["iso","wm","spr"],
    ("Jaco",):  ["ball","mgpt"],
    ("Theo",):  ["ball","mgpt"],
    ("Jango","JacB"): ["iso","wm","spr"],
    ("Jaco","Theo"):  ["ball","mgpt"],
}

def canon_task(x: str):
    if x is None: return None
    t = str(x).strip().lower()
    return TASK_FIX.get(t, t)

def parse_scenario(s: str):
    """'JacB_iso' -> ('JacB','iso') | 'Jaco_ball' -> ('Jaco','ball')"""
    if not isinstance(s, str): return None, None
    m = re.match(r"^(Jango|JacB|Jaco|Theo)_(iso|wm|spr|ball|mgpt|mg-pt)$", s)
    if not m: return None, None
    mon, tr = m.group(1), canon_task(m.group(2))
    return mon, tr

def find_pkls(root):
    root = Path(root)
    pats = ("*.pkl","*.PKL","*.pickle","*.pkl.gz")
    return [str(p) for pat in pats for p in root.rglob(pat)]

def load_one(path):
    try:
        return pd.read_pickle(path)
    except Exception:
        return None

def ensure_out(p): Path(p).mkdir(parents=True, exist_ok=True)

def pivot_mat(df):
    return df.pivot_table(index="train_task", columns="test_task",
                          values="mean_VAF", aggfunc="mean")

def draw_heatmap(M, rows, cols, title, outpath, vmin, vmax, dpi):
    M = M.reindex(index=rows, columns=cols).astype(float)
    if M.size == 0: return
    fig, ax = plt.subplots(figsize=(6,5), dpi=dpi)
    im = ax.imshow(M.values, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel("Test Task"); ax.set_ylabel("Train Task"); ax.set_title(title)
    A = M.values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i,j]):
                ax.text(j, i, f"{A[i,j]:.2f}",
                        ha="center", va="center",
                        color=("white" if A[i,j] < 0.5 else "black"), fontsize=10)
    cb = fig.colorbar(im, ax=ax); cb.set_label("mean_VAF")
    ensure_out(os.path.dirname(outpath))
    plt.tight_layout(); plt.savefig(outpath); plt.close(fig)

def tasks_for_monkey(monkey, df_monkey):
    default = sorted(set(df_monkey["train_task"]).union(df_monkey["test_task"]))
    return PAIR_TASKS.get((monkey,), default)

def diag_map_from_internal_cv(df_scope):
    """Compute mean diagonal VAF per task from rows with align=='internal_cv' inside df_scope."""
    d = df_scope[df_scope["align"].astype(str).str.lower().eq("internal_cv")]
    if d.empty: return {}
    # internal_cv already has test_task==train_task after normalization
    return d.groupby("train_task")["mean_VAF"].mean().to_dict()

def inject_diagonal(M: pd.DataFrame, diag_map: dict):
    if not diag_map: return M
    for t, val in diag_map.items():
        if (t in M.index) and (t in M.columns):
            M.loc[t, t] = val
    return M

# ---------- normalize to your schema ----------
def normalize_df(df_raw: pd.DataFrame, src: str, debug=False) -> pd.DataFrame | None:
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        if debug: print(f"[SKIP] {src}: not a non-empty DataFrame")
        return None

    df = df_raw.copy()

    for c in ("scenario","decoder","dimred","align","mean_VAF"):
        if c not in df.columns:
            if debug: print(f"[SKIP] {src}: missing column '{c}'")
            return None

    mon, tr = parse_scenario(str(df["scenario"].iloc[0]))
    if mon is None:
        if debug: print(f"[SKIP] {src}: could not parse scenario='{df['scenario'].iloc[0]}'")
        return None
    df["monkey"] = mon
    df["train_task"] = canon_task(tr)

    if "test_task" not in df.columns:
        df["test_task"] = None
    df.loc[df["align"].astype(str).str.lower().eq("internal_cv"), "test_task"] = df["train_task"]
    df["test_task"] = df["test_task"].apply(canon_task)

    df["align"] = df["align"].astype(str).str.replace("latent(fallback)","latent", regex=False)
    df["dimred"]  = df["dimred"].astype(str).str.lower()
    df["decoder"] = df["decoder"].astype(str)

    if "train_day" in df.columns:
        df["day"] = df["train_day"].astype(str)
    elif "day" not in df.columns:
        df["day"] = "unknown"

    df = df.dropna(subset=["train_task","test_task"])
    if df.empty:
        if debug: print(f"[SKIP] {src}: no rows with both train/test task")
        return None

    keep = ["monkey","decoder","dimred","align","day","train_task","test_task","mean_VAF"]
    return df[keep]

# ---------- plotting modes (with diagonal injection) ----------
def plot_mode_1_per_monkey(df, outdir, vmin, vmax, dpi):
    # each monkey, avg over day/seed/fold; diagonal from internal_cv averaged over same scope
    for monkey in sorted(df["monkey"].unique()):
        dfm = df[df["monkey"]==monkey]
        order = tasks_for_monkey(monkey, dfm)
        for (decoder, dimred, align), dfg in dfm.groupby(["decoder","dimred","align"], dropna=False):
            # diagonal derived from *same monkey/decoder/dimred* but align=='internal_cv'
            diag_map = diag_map_from_internal_cv(dfm[(dfm["decoder"]==decoder)&(dfm["dimred"]==dimred)])
            M = pivot_mat(dfg)
            rows = [t for t in order if t in M.index]
            cols = [t for t in order if t in M.columns]
            if rows and cols:
                M2 = inject_diagonal(M, diag_map)
                title = f"{decoder}: {', '.join(order)}\n{monkey} | {dimred}/{align} (avg over day/seed/fold/ch)"
                out = os.path.join(outdir, "1_per_monkey", monkey, decoder, f"{dimred}_{align}.png")
                draw_heatmap(M2.loc[rows, cols], rows, cols, title, out, vmin, vmax, dpi)

def plot_mode_2_pairs(df, outdir, vmin, vmax, dpi):
    # paired monkeys; diagonal averaged across both monkeys' internal_cv in same decoder/dimred
    for m1, m2 in (("Jango","JacB"), ("Jaco","Theo")):
        dfp = df[df["monkey"].isin([m1,m2])]
        if dfp.empty: continue
        order = PAIR_TASKS[(m1,m2)]
        for (decoder, dimred, align), dfg in dfp.groupby(["decoder","dimred","align"], dropna=False):
            diag_map = diag_map_from_internal_cv(dfp[(dfp["decoder"]==decoder)&(dfp["dimred"]==dimred)])
            M = pivot_mat(dfg)
            rows = [t for t in order if t in M.index]
            cols = [t for t in order if t in M.columns]
            if rows and cols:
                M2 = inject_diagonal(M, diag_map)
                title = f"{decoder}: {', '.join(order)}\n{m1}+{m2} | {dimred}/{align} (avg over monkeys/day/seed/fold/ch)"
                out = os.path.join(outdir, "2_pairs", f"{m1}+{m2}", decoder, f"{dimred}_{align}.png")
                draw_heatmap(M2.loc[rows, cols], rows, cols, title, out, vmin, vmax, dpi)

def plot_mode_3_per_day(df, outdir, vmin, vmax, dpi):
    # per monkey per day; diagonal from internal_cv of that exact (monkey, day, decoder, dimred)
    for monkey in sorted(df["monkey"].unique()):
        dfm = df[df["monkey"]==monkey]
        order = tasks_for_monkey(monkey, dfm)
        for day, dfd in dfm.groupby("day"):
            for (decoder, dimred, align), dfg in dfd.groupby(["decoder","dimred","align"], dropna=False):
                diag_map = diag_map_from_internal_cv(
                    dfd[(dfd["decoder"]==decoder)&(dfd["dimred"]==dimred)]
                )
                M = pivot_mat(dfg)
                rows = [t for t in order if t in M.index]
                cols = [t for t in order if t in M.columns]
                if rows and cols:
                    M2 = inject_diagonal(M, diag_map)
                    title = f"{decoder}: {', '.join(order)}\n{monkey} | day={day} | {dimred}/{align} (avg over seed/fold/ch)"
                    out = os.path.join(outdir, "3_per_monkey_per_day", monkey, str(day), decoder, f"{dimred}_{align}.png")
                    draw_heatmap(M2.loc[rows, cols], rows, cols, title, out, vmin, vmax, dpi)

# ---------- main ----------
def main():
    args = parse_args()
    paths = find_pkls(args.root)
    if args.debug:
        print(f"[DEBUG] found {len(paths)} files under {args.root}")
        for p in paths[:25]: print("  ", p)
        if len(paths) > 25: print("  ...")

    if not paths:
        print(f"[WARN] no .pkl files found under: {args.root}")
        return

    frames, skipped = [], 0
    for p in paths:
        obj = load_one(p)
        if obj is None:
            skipped += 1
            if args.debug: print(f"[SKIP] unreadable: {p}")
            continue
        try:
            nf = normalize_df(obj, p, debug=args.debug)
        except Exception as e:
            nf = None
            if args.debug: print(f"[SKIP] normalize error in {p}: {e}")
        if nf is not None and not nf.empty:
            frames.append(nf)
        else:
            skipped += 1

    if not frames:
        print(f"[WARN] no usable .pkl files found under: {args.root}")
        if args.debug:
            print(f"[DEBUG] Total files existed: {len(paths)} | skipped: {skipped}")
        return

    df = pd.concat(frames, ignore_index=True)

    ensure_out(args.outdir)
    plot_mode_1_per_monkey(df, args.outdir, args.vmin, args.vmax, args.dpi)
    plot_mode_2_pairs(df, args.outdir, args.vmin, args.vmax, args.dpi)
    plot_mode_3_per_day(df, args.outdir, args.vmin, args.vmax, args.dpi)

    print(f"[OK] Plots saved to: {args.outdir}")
    if args.debug:
        print(f"[DEBUG] merged rows={len(df)}  skipped files={skipped}")

if __name__ == "__main__":
    main()
