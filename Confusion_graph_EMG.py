import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


MGPT_BALL_MONKEYS   = {"Jaco", "Theo"}        # monkeys that did mgpt / ball
ISO_WM_SPR_MONKEYS  = {"Jango", "JacB"}       # monkeys that did iso / wm / spr

# ───────────────────────── UPDATED HELPERS ─────────────────────────
def _apply_family_filter(df, tasks):
    """Return df restricted to the monkeys that actually did those tasks."""
    if set(tasks) <= {"mgpt", "ball"}:
        keep = MGPT_BALL_MONKEYS
    elif set(tasks) <= {"iso", "wm", "spr"}:
        keep = ISO_WM_SPR_MONKEYS
    else:                         # mixed list – keep everything
        return df
    return df[df['train_monkey'].isin(keep)]

def _heatmap(piv, title, fname):
    plt.figure(figsize=(1.5 + 1.6*piv.shape[1], 1.3 + 0.9*piv.shape[0]))
    sns.heatmap(piv, annot=True, fmt=".2f",
                cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "mean_VAF"})
    plt.title(title)
    plt.xlabel("Test Task")
    plt.ylabel(piv.index.name.capitalize())
    plt.tight_layout()
    plt.savefig(fname, dpi=700)
    plt.close()
    print(f"[INFO] saved → {fname}")

def _base(df, tasks, alignment_mode, decoder):
    """Return dataframe filtered by test‑tasks & alignment/decoder labels."""
    df_f = df[df['test_task'].isin(tasks)]
    if alignment_mode is not None:
        df_f = df_f[df_f['alignment_mode'] == alignment_mode]
    if decoder is not None:
        df_f = df_f[df_f['decoder_type'] == decoder]
    return df_f

def plot_confusion_mgpt_ball(df, metric='mean_VAF', *,
                             alignment_mode=None,
                             decoder=None,
                             group_by=None,      # None | 'decoder' | 'monkey'
                             grouped=False):
    tasks = ["mgpt", "ball"]
    df_f  = _base(df, tasks, alignment_mode, decoder)
    df_f  = _apply_family_filter(df_f, tasks)
    if df_f.empty:
        print("[WARN] no mgpt/ball rows match current filters")
        return

    # ------------ rows = decoders -----------------------------------
    if group_by == "decoder":
        piv = (df_f.groupby(['decoder_type', 'test_task'])[metric]
                    .mean().unstack()
                    .reindex(index=["GRU","LSTM","LIN","LiGRU"]))
        _heatmap(piv, "mgpt / ball – decoders as rows",
                 "mgpt_ball_rows=decoder.png")
        return

    # ------------ rows = monkeys ------------------------------------
    if group_by == "monkey":
        if grouped:                     # one PNG per monkey
            for mky, blk in df_f.groupby('train_monkey'):
                piv = (blk.groupby('test_task')[metric]
                          .mean().to_frame(mky).T)
                _heatmap(piv,
                         f"{mky}: mgpt / ball – mean over decoders",
                         f"mgpt_ball_{mky}.png")
        else:                           # all monkeys together
            piv = (df_f.groupby('test_task')[metric]
                        .mean().to_frame("all").T)
            _heatmap(piv,
                     "mgpt / ball – mean over decoders & monkeys",
                     "mgpt_ball_ALL.png")
        return

    # ------------ original per‑decoder single‑row version -----------
    if decoder is None:
        print("[WARN] decoder=None with group_by=None – nothing plotted")
    else:
        piv = (df_f.groupby(['train_task','test_task'])[metric]
                   .mean().unstack())
        _heatmap(piv,
                 f"{decoder}: mgpt / ball",
                 f"mgpt_ball_{decoder}.png")

def plot_confusion_iso_wm_spr(df, metric='mean_VAF', *,
                              alignment_mode=None,
                              decoder=None,
                              group_by=None,      # None | 'decoder' | 'monkey'
                              grouped=False):
    tasks = ["iso", "wm", "spr"]
    df_f  = _base(df, tasks, alignment_mode, decoder)
    df_f  = _apply_family_filter(df_f, tasks)
    if df_f.empty:
        print("[WARN] no iso/wm/spr rows match current filters")
        return

    if group_by == "decoder":
        piv = (df_f.groupby(['decoder_type', 'test_task'])[metric]
                    .mean().unstack()
                    .reindex(index=["GRU","LSTM","LIN","LiGRU"]))
        _heatmap(piv, "iso / wm / spr – decoders as rows",
                 "iso_wm_spr_rows=decoder.png")
        return

    if group_by == "monkey":
        if grouped:
            for mky, blk in df_f.groupby('train_monkey'):
                piv = (blk.groupby('test_task')[metric]
                          .mean().to_frame(mky).T)
                _heatmap(piv,
                         f"{mky}: iso / wm / spr – mean over decoders",
                         f"iso_wm_spr_{mky}.png")
        else:
            piv = (df_f.groupby('test_task')[metric]
                        .mean().to_frame("all").T)
            _heatmap(piv,
                     "iso / wm / spr – mean over decoders & monkeys",
                     "iso_wm_spr_ALL.png")
        return

    if decoder is None:
        print("[WARN] decoder=None with group_by=None – nothing plotted")
    else:
        piv = (df_f.groupby(['train_task','test_task'])[metric]
                   .mean().unstack())
        _heatmap(piv,
                 f"{decoder}: iso / wm / spr",
                 f"iso_wm_spr_{decoder}.png")

df = pd.read_pickle("df_results_emg_validation_whithin.pkl")
align_type = "recalculated"
# 1) old behaviour: one PNG per decoder
for dec in ["GRU","LSTM","Linear","LiGRU"]:
    plot_confusion_mgpt_ball(df, alignment_mode=align_type, decoder=dec)
    plot_confusion_iso_wm_spr(df, alignment_mode=align_type, decoder=dec)

# 2) decoders as **rows** (4×N)
# plot_confusion_mgpt_ball(df, alignment_mode=align_type,
#                          decoder=None, group_by="decoder")
# plot_confusion_iso_wm_spr(df, alignment_mode=align_type,
#                           decoder=None, group_by="decoder")

# # 3) mean of all decoders, **one PNG per monkey**
# plot_confusion_mgpt_ball(df, alignment_mode=align_type,
#                          decoder=None, group_by="monkey", grouped=True)
# plot_confusion_iso_wm_spr(df, alignment_mode=align_type,
#                           decoder=None, group_by="monkey", grouped=True)

# # 4) mean of all decoders, all monkeys together
# plot_confusion_mgpt_ball(df, alignment_mode=align_type,
#                          decoder=None, group_by="monkey", grouped=False)
# plot_confusion_iso_wm_spr(df, alignment_mode=align_type,
#                           decoder=None, group_by="monkey", grouped=False)
