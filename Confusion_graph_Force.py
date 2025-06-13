import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------
# 1) Optionally parse train_task/test_task from your scenario_name/test_name
#    if you don't explicitly store them in df_results
# ------------------------------------------------------------------------
def parse_train_task(row):
    """
    Example parse:
      If scenario_name is 'TaskComp_Jango_iso' => parse the last substring 'iso' as train_task
      Or if scenario_name is 'MonkeyComp_iso_Jango2JacB', parse 'iso'
    Adjust to match your real naming.
    """
    name = row.get('scenario_name','')
    # naive approach: split by '_' and pick last substring if it matches iso/wm/spr
    parts = name.split('_')
    # e.g. 'TaskComp','Jango','iso'
    possible_tasks = ['iso','wm','spr']
    for p in reversed(parts):
        if p in possible_tasks:
            return p
    return None

def parse_test_task(row):
    """
    If test_name is 'val_25pct', interpret it as the same as row['train_task'] 
    so it appears on the diagonal of the confusion matrix.
    Otherwise, parse 'iso','wm','spr' from test_name.
    """
    tname = row.get('test_name','')
    if 'val_25pct' in tname:
        # Force it to match the train_task
        return row.get('train_task', 'val_25pct')
    
    possible_tasks = ['iso','wm','spr']
    parts = tname.split('_')
    for p in parts:
        if p in possible_tasks:
            return p
    return None

def add_task_columns(df):
    # If 'train_task' not in columns, parse from scenario_name
    if 'train_task' not in df.columns:
        df['train_task'] = df.apply(parse_train_task, axis=1)
    # If 'test_task' not in columns, parse from test_name
    if 'test_task' not in df.columns:
        df['test_task'] = df.apply(parse_test_task, axis=1)
    return df

# ------------------------------------------------------------------------
# 2) Plot a 3×3 tasks confusion (train_task vs. test_task) for a single alignment mode or decoder
# ------------------------------------------------------------------------
def plot_confusion_tasks(df, metric='VAF', alignment_mode=None, decoder=None):
    """
    Example: produce a single heatmap with rows= train_task, cols= test_task.
    We filter df to a specific alignment_mode (if not None) and decoder (if not None).
    Then pivot on train_task vs test_task => mean of metric.
    """
    df_filt = df.copy()
    if alignment_mode:
        df_filt = df_filt[df_filt['alignment_mode']==alignment_mode]
    if decoder:
        df_filt = df_filt[df_filt['decoder_type']==decoder]
    
    # group by train_task, test_task => mean metric
    grouped = df_filt.groupby(['train_task','test_task'])[metric].mean().reset_index()
    if grouped.empty:
        print(f"[WARNING] No data for alignment={alignment_mode}, decoder={decoder}")
        return
    
    pivot_df = grouped.pivot(index='train_task', columns='test_task', values=metric)
    pivot_df = pivot_df.fillna(np.nan)

    plt.figure(figsize=(5,4))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
    title = f"Tasks Confusion (metric={metric})"
    if alignment_mode: title += f", mode={alignment_mode}"
    if decoder:        title += f", dec={decoder}"
    plt.title(title)
    plt.xlabel("test_task")
    plt.ylabel("train_task")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# 3) Plot confusion subplots for each alignment mode or each decoder
# ------------------------------------------------------------------------
def plot_confusion_tasks_subplots(df, metric='VAF', 
                                  alignment_modes=['none','realignment','monkey_level'], 
                                  decoders=['GRU','LSTM','LiGRU','Linear']):
    """
    We'll make a grid of subplots:
     rows = alignment_modes
     cols = decoders
    Each subplot is train_task vs test_task confusion.
    """
    nrows = len(alignment_modes)
    ncols = len(decoders)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows), sharex=False, sharey=False)

    for i, mode in enumerate(alignment_modes):
        for j, dec in enumerate(decoders):
            ax = axes[i][j] if nrows>1 else axes[j]
            df_filt = df.copy()
            df_filt = df_filt[df_filt['alignment_mode']==mode]
            df_filt = df_filt[df_filt['decoder_type']==dec]
            grouped = df_filt.groupby(['train_task','test_task'])[metric].mean().reset_index()
            if grouped.empty:
                ax.set_title(f"No data mode={mode}, dec={dec}")
                ax.axis('off')
                continue

            pivot_df = grouped.pivot(index='train_task', columns='test_task', values=metric)
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='viridis', ax=ax)
            ax.set_title(f"{mode}, {dec}")
            ax.set_xlabel("test_task")
            ax.set_ylabel("train_task")

    fig.suptitle(f"Tasks Confusion (metric={metric})")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# 4) Plot a monkey confusion (train_monkey vs test_monkey)
# ------------------------------------------------------------------------
def plot_confusion_monkeys(df, metric='VAF', alignment_mode=None, decoder=None):
    """
    Single heatmap with rows=train_monkey, cols=test_monkey, cell=mean metric
    Ignores tasks. Good for cross-monkey scenarios.
    """
    df_filt = df.copy()
    if alignment_mode:
        df_filt = df_filt[df_filt['alignment_mode']==alignment_mode]
    if decoder:
        df_filt = df_filt[df_filt['decoder_type']==decoder]
    
    grouped = df_filt.groupby(['train_monkey','test_monkey'])[metric].mean().reset_index()
    if grouped.empty:
        print(f"[WARNING] No data for alignment={alignment_mode}, decoder={decoder}")
        return

    pivot_df = grouped.pivot(index='train_monkey', columns='test_monkey', values=metric)

    plt.figure(figsize=(4,4))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='viridis')
    title = f"Monkey Confusion (metric={metric})"
    if alignment_mode: title += f", mode={alignment_mode}"
    if decoder:        title += f", dec={decoder}"
    plt.title(title)
    plt.xlabel("test_monkey")
    plt.ylabel("train_monkey")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# Example main
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=str, default="train_val_3squares_results.pkl",
                        help="Path to the pickled df_results")
    parser.add_argument("--metric", type=str, default="VAF",
                        help="Which metric to visualize (VAF or MSE)")
    args = parser.parse_args()

    if not os.path.exists(args.df):
        print(f"[ERROR] File {args.df} not found")
        return

    df_results = pd.read_pickle(args.df)
    print("[INFO] df_results shape=", df_results.shape)
    print("[INFO] columns=", df_results.columns.tolist())

    # 1) If needed, parse tasks
    df_results = add_task_columns(df_results)

    # 2) Plot a single tasks confusion for alignment_mode='none', decoder='GRU'
    plot_confusion_tasks(df_results, metric=args.metric, alignment_mode='none', decoder='GRU')

    # 3) Subplots => each row=alignment_mode, col=decoder
    plot_confusion_tasks_subplots(df_results, metric=args.metric,
        alignment_modes=['none','realignment','monkey_level'],
        decoders=['GRU','LSTM','LiGRU','Linear'])

    # 4) Plot a single monkey confusion => e.g. alignment_mode='monkey_level'
    plot_confusion_monkeys(df_results, metric=args.metric, alignment_mode='monkey_level', decoder='GRU')

if __name__=="__main__":
    main()
