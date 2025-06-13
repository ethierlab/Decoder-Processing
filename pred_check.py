import math, sys, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ------------------------------------------------------------------
# ------------   WHAT DO YOU WANT TO SEE?  -------------------------
RESULTS_PKL   = "df_results_emg_validation_caseA.pkl"

SCENARIO      = "Jango_wm"   # value of 'scenario_name'
DECODER       = "GRU"                   # GRU | LSTM | Linear | LiGRU
ALIGNMENT     = "realign"                 # bland | recalculated | realign
HOLDOUT_TYPE  = "internal"              # internal | external
# ------------------------------------------------------------------

def pick_row(df):
    """Return the first row that satisfies all four filters."""
    mask = (
        (df["scenario_name"] == SCENARIO) &
        (df["decoder_type"]  == DECODER)  &
        (df["alignment_mode"]== ALIGNMENT) &
        (df.get("holdout_type", "internal") == HOLDOUT_TYPE)
    )
    sub = df[mask]
    if sub.empty:
        raise ValueError(
            f"No row matches "
            f"scenario='{SCENARIO}', decoder='{DECODER}', "
            f"alignment='{ALIGNMENT}', holdout='{HOLDOUT_TYPE}'")
    if len(sub) > 1:
        print(f"[INFO] {len(sub)} rows match; taking the first one (index {sub.index[0]}).")
    return sub.iloc[0]

def plot_emg(preds, truth, ch_names=None, per_ax=3):
    """Three channels per subplot + mean trace."""
    n_t, n_ch = truth.shape
    if ch_names is None:
        ch_names = [f"ch{c+1}" for c in range(n_ch)]

    groups = math.ceil(n_ch / per_ax)
    fig, axes = plt.subplots(groups, 1, figsize=(12, 3*groups),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    t = np.arange(n_t)

    for g, ax in enumerate(axes):
        a, b = g*per_ax, min((g+1)*per_ax, n_ch)
        for c in range(a, b):
            ax.plot(t, truth[:, c],  lw=1, label=f"{ch_names[c]} true")
            ax.plot(t, preds[:, c],  lw=1, ls="--", label=f"{ch_names[c]} pred")
        ax.plot(t, truth[:, a:b].mean(1), lw=2, label="mean true")
        ax.plot(t, preds[:, a:b].mean(1), lw=2, ls="--", label="mean pred")
        ax.set_ylabel("EMG")
        ax.legend(fontsize="small")
        ax.set_title(f"Channels {a+1}–{b}")

    axes[-1].set_xlabel("time bin")
    plt.suptitle(f"{SCENARIO} | {DECODER} | {ALIGNMENT} | {HOLDOUT_TYPE}", y=1.02)
    plt.show()

def main():
    pkl = pathlib.Path(RESULTS_PKL)
    if not pkl.exists():
        sys.exit(f"File not found: {pkl.resolve()}")

    df = pd.read_pickle(pkl)
    row = pick_row(df)

    # preds = np.asarray(row["preds"])
    # truth = np.asarray(row.get("Y_true", row.get("ground_truth")))
    vafs = np.asarray(row.get("VAF_ch", []))          # shape (n_ch,)  or  () if empty
    vafs = np.ravel(vafs) 
    ch_names = row.get("emg_names")                   # list of strings if present
    if ch_names is None or len(ch_names) != len(vafs):
        ch_names = [f"ch{i+1}" for i in range(len(vafs))]

    # ---- print -----------------------------------------------------------
    print("Per‑channel VAFs:")
    for name, v in zip(ch_names, vafs):
        print(f"  {name}: {v:.3f}")
    print(f"Mean VAF: {np.nanmean(vafs):.3f}")
    # if preds.shape != truth.shape:
    #     sys.exit(f"preds {preds.shape} vs truth {truth.shape} mismatch")

    # plot_emg(preds, truth, ch_names=row.get("emg_names"))

if __name__ == "__main__":
    main()
