# ───────────────────────── plot_results_rows.py ─────────────────────────
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DF_RESULTS_PATH = "df_results_emg_validation_hybrid_200.pkl"
ALIGNMENT_LABEL = "time‑split"            # what your trainer saved
DECODERS        = ["GRU", "LSTM", "LIN", "LiGRU"]
SAVE_DIR        = "."

# ────────────────────────── helpers ─────────────────────────────────────
def _draw(pivot, title, fname):
    plt.figure(figsize=(1.5 + 1.6*pivot.shape[1], 1.3 + 0.9*pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="viridis", cbar_kws={"label": "mean_VAF"})
    plt.title(title)
    plt.xlabel("Test Task")
    plt.ylabel("Decoder")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, fname)
    plt.savefig(path, dpi=700)
    plt.close()
    print(f"[INFO] saved → {path}")

def build_decoder_vs_tasks(df, tasks):
    """
    Returns a pivot table: rows = decoder_type, columns = task, values = mean_VAF
    averaged over all monkeys & days that match ALIGNMENT_LABEL.
    Missing task/decoder combos are filled with NaN (so they show as blank).
    """
    df_f = (df[(df.train_task == "hybrid") &
               (df.test_task.isin(tasks)) &
               (df.alignment_mode == ALIGNMENT_LABEL)])

    # average over all days & monkeys first
    grouped = (df_f.groupby(["decoder_type", "test_task"])
                    .mean_VAF.mean()
                    .reset_index())

    pivot = (grouped.pivot(index="decoder_type",
                           columns="test_task",
                           values="mean_VAF")
                    .reindex(index=DECODERS))          # keep row order
    # keep column order consistent with *tasks*
    pivot = pivot.reindex(columns=list(tasks))
    return pivot

# ──────────────────────────── main ──────────────────────────────────────
def main():
    if not os.path.exists(DF_RESULTS_PATH):
        print(f"[ERROR] cannot find {DF_RESULTS_PATH}")
        sys.exit(1)

    df = pd.read_pickle(DF_RESULTS_PATH)
    print("[INFO] loaded results:", df.shape)

    # ---------- Hybrid vs mgpt / ball (4 × 2) ---------------------------
    tasks_mg = ["mgpt", "ball"]          # keep display order
    pivot_mg = build_decoder_vs_tasks(df, tasks_mg)
    _draw(pivot_mg,
          "Hybrid‑trained → mgpt / ball (all monkeys)",
          "Hybrid_vs_mgpt_ball_ALL.png")

    # ---------- Hybrid vs iso / wm / spr (4 × 3) ------------------------
    tasks_iso = ["iso", "wm", "spr"]
    pivot_iso = build_decoder_vs_tasks(df, tasks_iso)
    _draw(pivot_iso,
          "Hybrid‑trained → iso / wm / spr (all monkeys)",
          "Hybrid_vs_iso_wm_spr_ALL.png")

if __name__ == "__main__":
    main()
# for monkey in df.train_monkey.unique():
#     df_m = df[df.train_monkey == monkey]
#     pivot_mg = build_decoder_vs_tasks(df_m, tasks_mg)
#     _draw(pivot_mg,
#           f"{monkey}: Hybrid → mgpt / ball",
#           f"Hybrid_vs_mgpt_ball_{monkey}.png")

#     pivot_iso = build_decoder_vs_tasks(df_m, tasks_iso)
#     _draw(pivot_iso,
#           f"{monkey}: Hybrid → iso / wm / spr",
#           f"Hybrid_vs_iso_wm_spr_{monkey}.png")