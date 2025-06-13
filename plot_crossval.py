import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def remove_outliers(group, threshold=2.0):
    """
    Remove rows where 'vaf' is outside mean ± threshold*std for each group.
    """
    if len(group) < 2:
        return group  # not enough data for a std-based cutoff
    mean_vaf = group["vaf"].mean()
    std_vaf  = group["vaf"].std()
    if std_vaf == 0:
        return group  # all points identical => no outliers in that group
    lower_bound = mean_vaf - threshold * std_vaf
    upper_bound = mean_vaf + threshold * std_vaf
    return group[(group["vaf"] >= lower_bound) & (group["vaf"] <= upper_bound)]

def plot_crossval_results(pkl_path, save_dir='.', dpi=300):
    # =========================================================================
    # 1) Load the pickle results
    # =========================================================================
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Could not find pickle file: {pkl_path}")
    results_dict = pd.read_pickle(pkl_path)
    print(f"[INFO] Loaded results from {pkl_path}")

    # Extract data
    test_days      = results_dict['test_days']      # shape (n_days,) date/time
    all_gru_vafs   = results_dict['gru_vafs']       # shape (n_cv_runs, n_days)
    all_lstm_vafs  = results_dict['lstm_vafs']
    all_lin_vafs   = results_dict['lin_vafs']
    all_ligru_vafs = results_dict['ligru_vafs']

    n_cv_runs = all_gru_vafs.shape[0]
    # Convert test_days to integer day offsets from day0
    base_day = test_days[0]
    day_nums = [(d.date() - base_day.date()).days for d in test_days]

    # =========================================================================
    # 2) Build a single DataFrame for all decoders, days, and CV runs
    # =========================================================================
    decoders_list = ["GRU", "LSTM", "Linear", "LiGRU"]
    arrays_dict = {
        "GRU":    all_gru_vafs,
        "LSTM":   all_lstm_vafs,
        "Linear": all_lin_vafs,
        "LiGRU":  all_ligru_vafs
    }

    df_list = []
    for decoder in decoders_list:
        arr = arrays_dict[decoder]  # shape (n_cv_runs, n_days)
        for i_cv in range(n_cv_runs):
            for i_day, _ in enumerate(test_days):
                df_list.append({
                    "crossval": i_cv,
                    "day_idx":  i_day,           # index in test_days
                    "day_num":  day_nums[i_day], # integer offset from day0
                    "decoder":  decoder,
                    "vaf":      arr[i_cv, i_day]
                })

    df_all = pd.DataFrame(df_list)

    # =========================================================================
    # 3) Remove outliers groupwise (Decoder x Day)
    #    i.e., exclude data points outside ±2 std for that day+decoder.
    # =========================================================================
    df_filtered = (
        df_all.groupby(["decoder", "day_num"], group_keys=False)
              .apply(remove_outliers, threshold=2.0)
              .reset_index(drop=True)
    )

    # After this step, df_filtered is your "cleaned" data set
    # for *all* subsequent plots.

    # =========================================================================
    # 4) From df_filtered, re-compute each array or do direct group-based stats
    # =========================================================================

    # A. Pivot each decoder into separate columns for day/crossval
    #    (Optional: if you want to keep the same "array of shape (n_cv, n_days)" approach)
    #    Note: Some crossval runs or entire day+decoder groups may now have fewer rows
    #          if outliers were removed.
    #
    #    We'll show a direct group-based approach using df_filtered instead.

    # 4.1) Plot #1: Mean across folds per day (line plot)
    # --------------------------------------------------
    # For each day+decoder, we find the mean across crossval runs. Then line-plot vs day_num.
    # We'll store them in a small dictionary for convenience:
    plot_dict = {}
    for dec in decoders_list:
        # Filter for that decoder
        df_dec = df_filtered[df_filtered["decoder"] == dec]
        # Group by day_num and compute the average
        grp_dec = df_dec.groupby("day_num")["vaf"]
        mean_dec = grp_dec.mean()
        std_dec  = grp_dec.std()
        # We'll store as a new subdict
        plot_dict[dec] = {
            "day_num": mean_dec.index.values,
            "mean":    mean_dec.values,
            "std":     std_dec.values
        }

    # Make a figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for dec in decoders_list:
        xvals = plot_dict[dec]["day_num"]
        yvals = plot_dict[dec]["mean"]
        ax.plot(xvals, yvals, marker='o', label=dec)
    ax.set_xlabel("Days from day0 (int)")
    ax.set_ylabel("VAF (mean across CV & EMG channels)")
    ax.set_title("VAF over Days (Mean of CV runs)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'day_evo_filtered_recalign.png'), dpi=dpi)
    plt.show()

    # 4.2) Plot #2: Relative VAF Loss (day0 - dayX) with mean±std shading
    # -------------------------------------------------------------------
    # For relative difference, we still need each crossval run's day0 reference.
    # We'll pivot or do a self-merge. Let's just do a dictionary approach:

    # We'll create a helper dict { (decoder, i_cv): day0_vaf }
    day0_ref = {}
    for dec in decoders_list:
        df0 = df_filtered[(df_filtered["decoder"] == dec) & (df_filtered["day_num"] == 0)]
        # For each crossval run in day0, get the VAF
        for i_cv, row in df0.iterrows():
            key = (dec, row["crossval"])
            day0_ref[key] = row["vaf"]  # store day0's VAF

    # Now compute day0 - dayX for each row (where day0 exists)
    def compute_rel_loss(row):
        key = (row["decoder"], row["crossval"])
        if key not in day0_ref:
            return np.nan  # no day0 reference => can't compute
        return day0_ref[key] - row["vaf"]

    df_filtered["rel_loss"] = df_filtered.apply(compute_rel_loss, axis=1)

    # For each day+decoder, compute mean/stdev of these relative losses
    plot_loss = []
    for dec in decoders_list:
        df_dec = df_filtered[df_filtered["decoder"] == dec]
        grp = df_dec.groupby("day_num")["rel_loss"]
        mean_l = grp.mean()
        std_l  = grp.std()
        for d_num in mean_l.index:
            plot_loss.append({
                "decoder": dec,
                "day_num": d_num,
                "mean_loss": mean_l.loc[d_num],
                "std_loss":  std_l.loc[d_num]
            })
    df_plot_loss = pd.DataFrame(plot_loss)

    # Plot
    fig2, ax2 = plt.subplots(figsize=(7,5))
    for dec in decoders_list:
        sub = df_plot_loss[df_plot_loss.decoder == dec].sort_values("day_num")
        x = sub["day_num"]
        m = sub["mean_loss"]
        s = sub["std_loss"]
        ax2.plot(x, m, '-o', label=dec)
        ax2.fill_between(x, m-s, m+s, alpha=0.2)
    ax2.set_xlabel("Days from day0")
    ax2.set_ylabel("Relative VAF Loss (day0 - dayX)")
    ax2.set_title("Relative VAF Loss vs. Days (Mean ± Std)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "relative_vaf_loss_filtered_recalign.png"), dpi=dpi)
    plt.show()

    # 4.3) Plot #3: Cumulative sum of VAF Loss (mean ± std)
    # -----------------------------------------------------
    # We need a per-crossval-run sequence of losses to cumsum day-wise
    # Then average across crossval runs. We'll pivot:

    # Let's pivot on (crossval) so each row is day_num, each column is crossval run
    # for a given decoder. Then cumsum row-wise, then average.
    def compute_csum_stats(dec):
        df_dec = df_filtered[df_filtered.decoder == dec].copy()
        # Just keep day_num, crossval, rel_loss
        # Pivot so that index=day_num, columns=crossval, values=rel_loss
        pivoted = df_dec.pivot_table(
            index="day_num", 
            columns="crossval", 
            values="rel_loss", 
            aggfunc='mean'  # just in case multiple rows
        )
        # sort rows by day_num
        pivoted = pivoted.sort_index()
        # cumsum each column
        csum = pivoted.cumsum(axis=0)
        # mean & std across columns => crossval dimension
        mean_csum = csum.mean(axis=1)
        std_csum  = csum.std(axis=1)
        return mean_csum, std_csum

    fig3, ax3 = plt.subplots(figsize=(7,5))
    for dec in decoders_list:
        mean_csum, std_csum = compute_csum_stats(dec)
        xvals = mean_csum.index.values
        ax3.plot(xvals, mean_csum, '-o', label=dec)
        ax3.fill_between(xvals, mean_csum - std_csum, mean_csum + std_csum, alpha=0.2)
    ax3.set_xlabel("Days from day0")
    ax3.set_ylabel("Cumulative Relative VAF Loss")
    ax3.set_title("Cumulative VAF Loss (Mean ± Std, Outliers Removed)")
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_vaf_loss_filtered_recalign.png"), dpi=dpi)
    plt.show()

    # 4.4) Plot #4: Box + Scatter plot for each day & decoder
    # -------------------------------------------------------
    # We can use the same df_filtered directly with Seaborn:
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_filtered,
        x="day_num",
        y="vaf",
        hue="decoder",
        whis=[5, 95],
        showfliers=False,
        ax=ax4
    )
    sns.stripplot(
        data=df_filtered,
        x="day_num",
        y="vaf",
        hue="decoder",
        dodge=True,
        alpha=0.4,
        size=3,
        color="black",
        marker="o",
        ax=ax4
    )
    # Remove duplicate legend from stripplot
    ax4.legend_.remove()
    box_handles = ax4.artists  # each element here is the coloured box Seaborn drew

    # 3) your decoder names in the same order
    decoders = ["GRU", "LSTM", "Linear", "LiGRU"]

    # 4) draw a new legend re‑using those artists
    ax4.legend(
        box_handles,
        decoders,
        title="Decoder",
        loc="upper right",
        frameon=True
)
    # Re-create a single legend from the boxplot handles
    decoders = ["GRU", "LSTM","Linear", "Linear", ]
    handles, labels = ax4.get_legend_handles_labels()
    unique_handles = handles[:len(decoders)]
    unique_labels  = labels[:len(decoders)]
    ax4.legend(unique_handles, unique_labels, title="Decoder",
               loc="upper right", frameon=True)

    ax4.set_title("Boxplot of VAF per Day/Decoder (Outliers Removed)")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("VAF")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boxplot_vaf_smaller_scatter_filtered_realign.png"), dpi=dpi)
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_file = "crossval_results_realign.pkl"
    plot_crossval_results(pkl_file, save_dir=script_dir, dpi=700)
