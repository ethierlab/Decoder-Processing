import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(files, hidden_dim_min=1, hidden_dim_max=19):
    """
    Load data from multiple pickle files, returning a list of DataFrames.
    Each DataFrame has an extra 'dataset' column corresponding to the file.
    """
    all_dfs = []
    for file in files:
        try:
            with open(file, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"Error loading '{file}': {e}")
            continue

        df = pd.DataFrame(results)

        # Filter hidden_dim
        df = df[(df['hidden_dim'] >= hidden_dim_min) & (df['hidden_dim'] <= hidden_dim_max)]

        # Tag with file name so we know which dataset is which
        if not df.empty:
            df['dataset'] = os.path.basename(file)
            all_dfs.append(df)
        else:
            print(f"No valid entries remain after filtering hidden_dim in {file}. Skipping.")
    return all_dfs

def filter_by_params(all_dfs, param_filter):
    """
    Given a list of DataFrames and a parameter filter dict like:
        param_filter = {'N': 14, 'k': 16}
    keep only the rows that match all specified parameter values.
    Returns a new list of DataFrames.
    """
    filtered_dfs = []
    for df in all_dfs:
        # Work on a copy so you don't mutate the original
        temp_df = df.copy()
        for param_name, param_value in param_filter.items():
            temp_df = temp_df[temp_df[param_name] == param_value]

        if not temp_df.empty:
            filtered_dfs.append(temp_df)
    return filtered_dfs

def print_top_3(df, dataset_name=None):
    """
    Identify and print the top 3 entries with highest VAF in a given DataFrame.
    """
    if df.empty:
        return
    
    top_3_df = df.nlargest(3, 'VAF').reset_index(drop=True)
    title = f"=== Top 3 Entries by VAF for {dataset_name} ===" if dataset_name else "=== Top 3 Entries by VAF ==="
    print(f"\n{title}")
    for idx, row in top_3_df.iterrows():
        print(f"--- Top {idx + 1}: VAF = {row['VAF']} ---")
        for key, value in row.to_dict().items():
            print(f"{key}: {value}")
    print()

def plot_scatter_overlay(all_dfs):
    """
    Overlays multiple datasets in a single scatter plot.
    We color by dataset so you can distinguish them.
    (Hidden dim is NOT used as hue here, so each dataset is a different color.)
    """
    plt.figure(figsize=(8,6))
    ax = plt.gca()

    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        sns.scatterplot(
            data=df,
            x='num_params',
            y='VAF',
            alpha=0.7,
            label=dataset_name,
            # legend=False,        
            ax=ax
        )
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("VAF")
    ax.set_title("Scatter: VAF vs. # Params (Overlay)")
    # ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"SCATTER_TRAINNINGS.png", dpi=700)
    plt.show()

def plot_scatter_separate(all_dfs):
    """
    One scatter plot PER dataset in separate figures, color-coded by hidden_dim.
    """
    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        sns.scatterplot(
            data=df,
            x='num_params',
            y='VAF',
            hue='hidden_dim',
            alpha=0.7,
            # legend=False,        
            ax=ax
            
        )
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("VAF")
        ax.set_title(f"Scatter: VAF vs. # Params\nDataset: {dataset_name}")
        # ax.legend(title='hidden_dim', loc='best')
        plt.tight_layout()
        plt.savefig(f"SCATTER_TRAINNINGS_single_{df}.png", dpi=700)
        plt.show()

def plot_max_vaf_overlay(all_dfs):
    """
    Overlays lines of max VAF vs. num_params for each dataset on ONE plot.
    """
    plt.figure(figsize=(8,6))
    ax = plt.gca()

    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        # Group by num_params, get max VAF
        grouped = df.groupby('num_params', as_index=False)['VAF'].max()
        sns.lineplot(
            data=grouped, 
            x='num_params', 
            y='VAF', 
            marker='o', 
            label=dataset_name,
            # legend=False,        
            ax=ax
            
        )
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Max VAF")
    ax.set_title("Max VAF vs. # Params (Overlay)")
    # ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Max_VAF.png", dpi=700)
    plt.show()

def plot_max_vaf_separate(all_dfs):
    """
    One figure per dataset for max VAF vs. num_params.
    """
    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        grouped = df.groupby('num_params', as_index=False)['VAF'].max()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        sns.lineplot(
            data=grouped,
            x='num_params',
            y='VAF',
            marker='o',
            ax=ax
        )
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("Max VAF")
        ax.set_title(f"Max VAF vs. # Params\nDataset: {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"Max_VAF_single_{df}.png", dpi=700)
        plt.show()

def plot_mean_std_vaf_overlay(all_dfs):
    """
    Overlays lines (with shaded ±1 std) for each dataset in ONE plot.
    """
    plt.figure(figsize=(8,6))
    ax = plt.gca()

    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        grouped = df.groupby('num_params')['VAF'].agg(['mean','std']).reset_index()

        ax.plot(grouped['num_params'], grouped['mean'], marker='o', label=dataset_name)
        ax.fill_between(
            grouped['num_params'],
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2
        )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Mean VAF")
    ax.set_title("Mean ± Std VAF vs. # Params (Overlay)")
    # ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Mean_STD.png", dpi=700)
    plt.show()

def plot_mean_std_vaf_separate(all_dfs):
    """
    One figure per dataset for mean ± std VAF vs. num_params.
    """
    for df in all_dfs:
        dataset_name = df['dataset'].iloc[0]
        grouped = df.groupby('num_params')['VAF'].agg(['mean','std']).reset_index()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        ax.plot(grouped['num_params'], grouped['mean'], marker='o', label='Mean VAF')
        ax.fill_between(
            grouped['num_params'],
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2,
            label='±1 Std'
        )
        
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("VAF")
        ax.set_title(f"Mean ± Std VAF vs. # Params\nDataset: {dataset_name}")
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig("Mean_STD.png", dpi=700)
        plt.savefig(f"Max_VAF_single_{df}.png", dpi=700)
        plt.show()

def main():
    # Provide paths to your pickle files here:
    files = [
        "experiment_results_gru_seeds_PCA_5_14_16.pkl",
        "experiment_results_gru_seeds_UMAP_5_14_16.pkl"
        
        # "experiment_results_gru_seeds.pkl",
        # "experiment_results_gru_seeds_UMAP_1-10.pkl",
        # "experiment_results_gru_seeds_PCA.pkl",
        # "experiment_results_lstm_seeds_UMAP.pkl"
        # Add more if needed
    ]


    overlay = True  # set to False for separate plots per dataset

    # Filter range for hidden_dim
    hidden_dim_min = 1
    hidden_dim_max = 15

    # additional param filter:
    param_filter = {
        'N': 14,
        'k': 16
    }

    # ---------------------------
    # Load data and proceed
    # ---------------------------

    all_dfs = load_data(files, hidden_dim_min=hidden_dim_min, hidden_dim_max=hidden_dim_max)
    if not all_dfs:
        print("No valid data loaded. Exiting.")
        return

    # Filter DataFrames by additional parameters
    # Comment this out if you do NOT want to filter by these params
    # all_dfs = filter_by_params(all_dfs, param_filter)
    if not all_dfs:
        print(f"No rows remained after applying parameter filter: {param_filter}")
        return

    # Print top 3 for each dataset
    for df in all_dfs:
        ds_name = df['dataset'].iloc[0]
        print_top_3(df, dataset_name=ds_name)

    # Now decide how to plot
    if overlay:
        # Overlaid scatter
        plot_scatter_overlay(all_dfs)

        # Overlaid max VAF
        plot_max_vaf_overlay(all_dfs)

        # Overlaid mean ± std
        plot_mean_std_vaf_overlay(all_dfs)
    else:
        # Separate scatter (color-coded by hidden_dim)
        plot_scatter_separate(all_dfs)

        # Separate max VAF
        plot_max_vaf_separate(all_dfs)

        # Separate mean ± std
        plot_mean_std_vaf_separate(all_dfs)


if __name__ == "__main__":
    main()
