import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# 1) Generic Confusion Matrix Plotter
###############################################################################
def plot_confusion_matrices(
    df,
    row_key="train_task",
    col_key="test_task",
    value_key="VAF",
    groupby_cols=["alignment_mode","decoder_type"],
    aggfunc="mean"
):
    """
    Create multiple "confusion matrix"-style heatmaps (using matplotlib.imshow) 
    for each combination of the grouping columns (e.g. alignment_mode, decoder_type).

    Parameters
    ----------
    df : pd.DataFrame
        The results DataFrame, containing columns for row_key, col_key, value_key, etc.
    row_key : str
        Column name to use for confusion matrix rows (e.g. "train_task").
    col_key : str
        Column name to use for confusion matrix columns (e.g. "test_task").
    value_key : str
        Column name whose numeric values we plot in the heatmap (e.g. "VAF" or "MSE").
    groupby_cols : list of str
        Columns used to create subplots. We'll group df by all combinations in these columns.
        For each combination, we pivot row_key x col_key → average of value_key.
    aggfunc : str
        How to aggregate multiple rows (e.g., "mean", "median"). By default "mean".

    Example usage:
        plot_confusion_matrices(
            df_results,
            row_key="train_task",
            col_key="test_task",
            value_key="VAF",
            groupby_cols=["alignment_mode","decoder_type"],
            aggfunc="mean"
        )
    """
    # unique combos in groupby_cols
    if len(groupby_cols) == 2:
        # We'll interpret groupby_cols[0] => subplot rows, groupby_cols[1] => subplot cols
        col_a, col_b = groupby_cols
        unique_a = df[col_a].unique()
        unique_b = df[col_b].unique()
        nrows = len(unique_a)
        ncols = len(unique_b)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

        for i, val_a in enumerate(unique_a):
            for j, val_b in enumerate(unique_b):
                ax = axes[i,j]
                # filter
                sub_df = df[(df[col_a] == val_a) & (df[col_b] == val_b)]
                if sub_df.empty:
                    ax.set_title(f"No data for {col_a}={val_a}, {col_b}={val_b}")
                    ax.axis("off")
                    continue

                # group and pivot
                gp = sub_df.groupby([row_key,col_key])[value_key]
                if aggfunc == "mean":
                    pivot_data = gp.mean()
                elif aggfunc == "median":
                    pivot_data = gp.median()
                else:
                    pivot_data = gp.mean()
                
                pivot_df = pivot_data.reset_index().pivot(index=row_key, columns=col_key, values=value_key)

                row_labels = pivot_df.index.tolist()
                col_labels = pivot_df.columns.tolist()
                mat = pivot_df.values

                im = ax.imshow(mat, aspect="auto", origin="upper")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_xticks(np.arange(len(col_labels)))
                ax.set_yticks(np.arange(len(row_labels)))
                ax.set_xticklabels(col_labels, rotation=45, ha="right")
                ax.set_yticklabels(row_labels)

                for rr in range(len(row_labels)):
                    for cc in range(len(col_labels)):
                        val = mat[rr, cc]
                        if not np.isnan(val):
                            ax.text(cc, rr, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)

                ax.set_title(f"{col_a}={val_a}, {col_b}={val_b}", fontsize=10)
                ax.set_xlabel(col_key)
                ax.set_ylabel(row_key)

        plt.tight_layout()
        plt.show()

    else:
        # single row of subplots or single col
        unique_combos = df[groupby_cols].drop_duplicates()
        n_subplots = len(unique_combos)
        fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots,5), squeeze=False)

        for idx, combo_vals in enumerate(unique_combos.itertuples(index=False)):
            ax = axes[0, idx]

            # build filter mask
            mask = pd.Series(True, index=df.index)
            for col_i, col_n in enumerate(groupby_cols):
                mask = mask & (df[col_n] == combo_vals[col_i])
            sub_df = df[mask]
            if sub_df.empty:
                ax.set_title(f"No data for combo={combo_vals}")
                ax.axis("off")
                continue

            gp = sub_df.groupby([row_key,col_key])[value_key]
            if aggfunc == "mean":
                pivot_data = gp.mean()
            elif aggfunc == "median":
                pivot_data = gp.median()
            else:
                pivot_data = gp.mean()
            
            pivot_df = pivot_data.reset_index().pivot(index=row_key, columns=col_key, values=value_key)

            row_labels = pivot_df.index.tolist()
            col_labels = pivot_df.columns.tolist()
            mat = pivot_df.values

            im = ax.imshow(mat, aspect="auto", origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
            ax.set_yticklabels(row_labels)

            for rr in range(len(row_labels)):
                for cc in range(len(col_labels)):
                    val = mat[rr, cc]
                    if not np.isnan(val):
                        ax.text(cc, rr, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)

            title_str = ", ".join(f"{c}={v}" for c,v in zip(groupby_cols, combo_vals))
            ax.set_title(title_str)
            ax.set_xlabel(col_key)
            ax.set_ylabel(row_key)

        plt.tight_layout()
        plt.show()

###############################################################################
# 2) Quick “Task Confusion” function
###############################################################################
def plot_task_confusion(df_results, alignment_modes=None, decoders=None):
    """
    Plots confusion matrix subplots for each (alignment_mode, decoder_type),
    with rows=train_task, columns=test_task, color=mean VAF.
    """
    if alignment_modes is None:
        alignment_modes = df_results["alignment_mode"].unique()
    if decoders is None:
        decoders = df_results["decoder_type"].unique()

    alignment_modes = list(alignment_modes)
    decoders = list(decoders)

    nrows = len(alignment_modes)
    ncols = len(decoders)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for i, mode in enumerate(alignment_modes):
        for j, dec in enumerate(decoders):
            ax = axes[i,j]
            sub_df = df_results[(df_results["alignment_mode"] == mode) &
                                (df_results["decoder_type"]   == dec)]
            if sub_df.empty:
                ax.set_title(f"No data for mode={mode}, dec={dec}")
                ax.axis("off")
                continue

            pivot_data = (sub_df.groupby(["train_task","test_task"])["VAF"]
                                  .mean()
                                  .reset_index())
            pivot_df = pivot_data.pivot(index="train_task", columns="test_task", values="VAF")

            row_labels = pivot_df.index.tolist()
            col_labels = pivot_df.columns.tolist()
            mat = pivot_df.values

            im = ax.imshow(mat, aspect="auto", origin="upper", cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
            ax.set_yticklabels(row_labels)
            
            for rr in range(len(row_labels)):
                for cc in range(len(col_labels)):
                    val = mat[rr, cc]
                    if not np.isnan(val):
                        ax.text(cc, rr, f"{val:.2f}", va="center", ha="center", color="white", fontsize=8)
            
            ax.set_title(f"Mode={mode}, Decoder={dec}")
            ax.set_xlabel("Test Task")
            ax.set_ylabel("Train Task")

    fig.suptitle("Task-based Confusion (VAF)", fontsize=14)
    plt.tight_layout()
    plt.show()

###############################################################################
# 3) Quick “Day Confusion” function
###############################################################################
def plot_day_confusion(df_results, alignment_modes=None, decoders=None):
    """
    Similar approach but for day-based confusion:
    rows=train_date, columns=test_date, color=mean VAF.
    """
    if alignment_modes is None:
        alignment_modes = df_results["alignment_mode"].unique()
    if decoders is None:
        decoders = df_results["decoder_type"].unique()

    alignment_modes = list(alignment_modes)
    decoders = list(decoders)

    nrows = len(alignment_modes)
    ncols = len(decoders)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for i, mode in enumerate(alignment_modes):
        for j, dec in enumerate(decoders):
            ax = axes[i,j]
            sub_df = df_results[(df_results["alignment_mode"] == mode) &
                                (df_results["decoder_type"]   == dec)]
            if sub_df.empty:
                ax.set_title(f"No data for mode={mode}, dec={dec}")
                ax.axis("off")
                continue

            pivot_data = (sub_df.groupby(["train_date","test_date"])["VAF"]
                                  .mean()
                                  .reset_index())
            pivot_df = pivot_data.pivot(index="train_date", columns="test_date", values="VAF")

            row_labels = pivot_df.index.tolist()
            col_labels = pivot_df.columns.tolist()
            mat = pivot_df.values

            im = ax.imshow(mat, aspect="auto", origin="upper", cmap="cividis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
            ax.set_yticklabels(row_labels)

            for rr in range(len(row_labels)):
                for cc in range(len(col_labels)):
                    val = mat[rr, cc]
                    if not np.isnan(val):
                        ax.text(cc, rr, f"{val:.2f}", va="center", ha="center", color="white", fontsize=8)

            ax.set_title(f"Mode={mode}, Decoder={dec}", fontsize=10)
            ax.set_xlabel("Test Date")
            ax.set_ylabel("Train Date")

    fig.suptitle("Day-based Confusion (VAF)", fontsize=14)
    plt.tight_layout()
    plt.show()

###############################################################################
# 4) Bar Plot: alignment mode vs. decoder
###############################################################################
def plot_bar_alignment_decoder(df_results):
    """
    Show a bar chart of average VAF grouped by [alignment_mode, decoder_type].
    Each alignment_mode is a group on the x-axis, each bar is a different decoder.
    """
    grouped = df_results.groupby(["alignment_mode","decoder_type"])["VAF"].mean().reset_index()

    alignment_modes = grouped["alignment_mode"].unique()
    decoders        = grouped["decoder_type"].unique()

    pivot_table = grouped.pivot(index="alignment_mode", columns="decoder_type", values="VAF")

    x = np.arange(len(pivot_table.index))  # each alignment_mode
    width = 0.2  # bar width

    fig, ax = plt.subplots(figsize=(8,5))
    n_decoders = len(decoders)
    # offset for each bar group
    offsets = np.linspace(-width*(n_decoders-1)/2, width*(n_decoders-1)/2, n_decoders)

    for i, dec in enumerate(decoders):
        if dec not in pivot_table.columns:
            continue
        y_vals = pivot_table[dec].values
        bar_x  = x + offsets[i]
        ax.bar(bar_x, y_vals, width=width, label=f"{dec}")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_table.index, rotation=45, ha="right")
    ax.set_ylabel("Mean VAF")
    ax.set_title("Comparison of decoders & alignment modes (mean VAF)")
    ax.legend()
    plt.tight_layout()
    plt.show()

###############################################################################
# 5) MAIN PLOTTING
###############################################################################
def main_plot():
    # 1) Load your results DataFrame from the .pkl
    results_path = "my_alignment_results_debug.pkl"  # adapt path as needed
    df_results = pd.read_pickle(results_path)
    print("[INFO] Loaded df_results:", df_results.shape)
    print(df_results.head())

    # 2) Optionally filter if you only want certain monkeys or tasks
    # e.g. sub_df = df_results[df_results["train_monkey"] == "Jango"]
    # or df_results => no filtering => everything

    # 3) Plot (A) a confusion matrix for tasks
    plot_task_confusion(df_results)

    # 4) Plot (B) a confusion matrix for days
    plot_day_confusion(df_results)

    # 5) Plot (C) alignment-mode vs. decoder bar chart
    plot_bar_alignment_decoder(df_results)

    # 6) Or use the more general function that can pivot any row_key / col_key
    # Example: row=train_task, col=test_task, groupby=[alignment_mode, decoder_type]
    # plot_confusion_matrices(
    #     df_results,
    #     row_key="train_task",
    #     col_key="test_task",
    #     value_key="VAF",
    #     groupby_cols=["alignment_mode","decoder_type"],
    #     aggfunc="mean"
    # )

if __name__ == "__main__":
    main_plot()
