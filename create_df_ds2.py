import os
import h5py
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# ---- 1. Define a dictionary to map raw EMG labels to base muscle names ----
GLOBAL_MUSCLE_MAP = {
    # ---- FDS / FDP with radial/ulnar compartments ----
    'FDSR': 'FDS',    # FDS radial slip => FDS
    'FDSU': 'FDS',    # FDS ulnar slip  => FDS
    'FDSM': 'FDS',    # "m" often means medial slip => FDS
    'FDS2': 'FDS',    # multiple electrodes => still FDS

    'FDPR': 'FDP',    # FDP radial slip => FDP
    'FDPU': 'FDP',    # FDP ulnar slip  => FDP
    'FDPM': 'FDP',    # sometimes "m" => FDP
    'FDP3': 'FDP',    # numeric suffix => still FDP

    # ---- ECR / ECU / FCR / FCU with numeric or letter suffixes ----
    'ECR1': 'ECR',    
    'ECR2': 'ECR',    
    'ECRB': 'ECR',    # "b" often brevis => unify with ECR
    'ER':   'ECR',
    'EM':   'ECR',

    'ECU1': 'ECU',
    'ECU2': 'ECU',
    'EU':   'ECU',

    'FCR1': 'FCR',
    'FCR2': 'FCR',
    'FR':   'FCR',

    'FCU1': 'FCU',
    'FCU2': 'FCU',
    'FU':   'FCU',

    # ---- Single-letter rema (extensors/flexors) you’ve used previously ----
    'FM':   'FDS',  # flexor mass => FDS
    # Already covered 'ER','EM' => ECR, 'EU' => ECU, 'FR' => FCR, 'FU' => FCU

    # ---- Extensor Digitorum Communis ----
    # 'EDC' is the standard label; variants with compartments:
    'EDC1': 'EDC',
    'EDC2': 'EDC',
    'EDCU': 'EDC',  # sometimes "ulnar slip" => unify as EDC
    # If you see plain 'EDC' or 'EDC_???', map them to 'EDC' as well:
    'EDC':  'EDC',

    # ---- "EPL", "OP", "FPB", etc. typically stay as-is ----
    'EPL': 'EPL',
    'OP':  'OP',
    'FPB': 'FPB',

    # ---- "APB" = Abductor Pollicis Brevis, "APL" = Abductor Pollicis Longus ----
    'APB': 'APB',
    'APL': 'APL',
    # Some labs have "ADL" but actually mean "APL"? If so:
    'ADL': 'APL',

    # ---- "PT" = Pronator Teres? "SUP" = Supinator, "BRAD" => Brachioradialis ----
    'PT':   'PT',
    'SUP':  'SUP',
    'BRAD': 'BR',   # if you prefer a short code for Brachioradialis
    # you might also see 'BRA' -> 'BR'

    # ---- "FDI" = First Dorsal Interosseous ----
    'FDI': 'FDI',

    # ---- "1/2IO" => "IO"? Possibly Interossei muscle? ----
    # If you want to unify all "IO" lumps:
    '1/2IO': 'IO',

    # ---- Catch-alls: if you see 'FDP' or 'FDS' with no suffix, keep them ---
    'FDP': 'FDP',
    'FDS': 'FDS',

    # ---- Single-labeled examples that match exactly to themselves ----
    'ECR': 'ECR',
    'ECU': 'ECU',
    'FCR': 'FCR',
    'FCU': 'FCU',
}


# ---- 2. Helper function: rename columns (keeping multiple electrodes separate) ----
def replace_emg_columns(emg_df, muscle_map):
    new_cols = {}
    count_for_muscle = {}

    for col in emg_df.columns:
        raw_label = col.strip().upper()
        # 1) Possibly strip radial/ulnar suffix, or digits.
        # 2) muscle_map.get(...)
        # 3) If found, we unify. If not, skip or keep raw.

        # EXAMPLE:
        base = muscle_map.get(raw_label, raw_label)  # fallback
        count_for_muscle.setdefault(base, 0)
        count_for_muscle[base] += 1
        new_label = f"{base}_{count_for_muscle[base]}"

        new_cols[new_label] = emg_df[col]

    return pd.DataFrame(new_cols)
def read_uint16_string(ds):
    """Decodes a MATLAB-style uint16 dataset into a Python string."""
    arr = np.array(ds).flatten()
    decoded = ''.join(chr(int(x)) for x in arr if int(x) != 0)
    return decoded

def read_object_string(ds):
    """Decodes a MATLAB-style cell array of references/bytes into a list of strings."""
    arr = np.array(ds).flatten()
    out = []
    for elem in arr:
        if isinstance(elem, h5py.Reference):
            resolved = ds.file[elem][()]
            if resolved.dtype == np.uint16:
                s = ''.join(chr(int(x)) for x in resolved.flatten() if int(x) != 0)
                out.append(s)
            elif isinstance(resolved, bytes):
                out.append(resolved.decode('utf-8'))
            else:
                out.append(str(resolved))
        elif isinstance(elem, bytes):
            out.append(elem.decode('utf-8'))
        else:
            if hasattr(elem, 'dtype') and elem.dtype == np.uint16:
                s = ''.join(chr(int(x)) for x in elem.flatten() if int(x) != 0)
                out.append(s)
            else:
                out.append(str(elem))
    return out

def process_single_mat_file_new(mat_file_path, verbose=True):
    """
    Processes a .mat file with a top-level 'datasets' cell array.

    Each cell in 'datasets' is a 1×1 struct containing top-level fields:
      - date, monkey, labels
    Inside each struct, the field 'binned_data' (a cell array) contains task-specific data:
      - trialtablelabels and trialtable (to extract trial_start_time)
      - emgdatabin, emgguide -> EMG data
      - forcedatabin, forcelabels -> Force data
      - spikeratedata, neuronIDs -> Spike counts
      - timeframe -> Bin width

    Returns a DataFrame with one row per task.
    """
    rows = []
    if verbose: 
        print("== Starting processing of file ==")
        print(f"Opening MATLAB file: {mat_file_path}")

    with h5py.File(mat_file_path, 'r') as f:
        datasets_ds = f['datasets']
        shape = datasets_ds.shape

        if verbose: 
            print(f"'datasets' shape: {shape}")

        dataset_refs = []
        if shape[0] == 1 and shape[1] > 1:
            # 1 row, multiple columns
            for j in range(shape[1]):
                dataset_refs.append(datasets_ds[0, j])
        elif shape[1] == 1 and shape[0] > 1:
            # multiple rows, 1 column
            for i in range(shape[0]):
                dataset_refs.append(datasets_ds[i, 0])
        else:
            dataset_refs = list(datasets_ds[:].flatten())

        if verbose:
            print(f"Found {len(dataset_refs)} dataset cells to process.")

        # Iterate over each "dataset" in 'datasets'
        for idx, dref in enumerate(dataset_refs):
            if verbose:
                print(f"\n-- Processing dataset {idx+1}/{len(dataset_refs)} --")

            row_group = f[dref]

            # (A) Read top-level: date, monkey, tasks
            date_str = read_uint16_string(row_group['date'])
            if date_str.endswith('.mat'):
                date_only = date_str[:-4]
            else:
                date_only = date_str

            # Attempt to parse date
            if "/" in date_only:
                try:
                    year, month, day = date_only.split("/")
                except ValueError as e:
                    if verbose: 
                        print(f"Warning: Could not parse date '{date_only}' with slashes: {e}")
                    year, month, day = None, None, None
            elif len(date_only) == 8 and date_only.isdigit():
                year = date_only[:4]
                month = date_only[4:6]
                day = date_only[6:]
            else:
                if verbose:
                    print(f"Warning: Date format not recognized: '{date_only}'")
                year, month, day = None, None, None

            if verbose:
                print(f"Parsed date: {date_only} -> year={year}, month={month}, day={day}")

            monkey_name = read_uint16_string(row_group['monkey'])
            if verbose:
                print(f"Monkey: {monkey_name}")

            tasks_labels = read_object_string(row_group['labels'])
            if verbose:
                print(f"Task labels: {tasks_labels}")

            # (B) Access 'binned_data'
            if verbose: 
                print("Accessing 'binned_data' cell array...")

            binned_data_ds = row_group['binned_data']
            bd_shape = binned_data_ds.shape

            if verbose:
                print(f"'binned_data' shape: {bd_shape}")

            if bd_shape[0] == 1 and bd_shape[1] >= 1:
                n_tasks = bd_shape[1]
                indices = [(0, j) for j in range(n_tasks)]
            elif bd_shape[1] == 1 and bd_shape[0] >= 1:
                n_tasks = bd_shape[0]
                indices = [(i, 0) for i in range(n_tasks)]
            else:
                indices = [(i, j) for i in range(bd_shape[0]) for j in range(bd_shape[1])]
                n_tasks = len(indices)

            if verbose:
                print(f"Found {n_tasks} tasks in binned_data.")

            # (B) Loop over each task
            for (i_bd, j_bd) in indices:
                if verbose:
                    print(f"\n-- Processing task cell at index ({i_bd},{j_bd}) --")

                # Decide how to index task name
                task_idx = j_bd if bd_shape[0] == 1 else i_bd
                if task_idx < len(tasks_labels):
                    task_name = tasks_labels[task_idx]
                else:
                    task_name = f"Task{task_idx+1}"

                if verbose:
                    print(f"Task name: {task_name}")

                sub_ref = binned_data_ds[i_bd, j_bd]
                sub_group = f[sub_ref]

                # (B-1) Trial Table
                if verbose:
                    print("B-1: Extracting trialtable and trialtablelabels...")
                trialtablelabels_arr = read_object_string(sub_group['trialtablelabels'])
                trialtable_data = sub_group['trialtable'][()]

                if verbose:
                    print(f"trialtablelabels: {trialtablelabels_arr}")
                    print(f"trialtable_data shape: {trialtable_data.shape}")

                trial_start_index = None
                for idx_tt, label in enumerate(trialtablelabels_arr):
                    if "trial start" in label.lower():
                        trial_start_index = idx_tt
                        break

                if trial_start_index is not None:
                    trial_start_time = trialtable_data[trial_start_index, :]
                    if verbose: 
                        print("Trial start found.")
                else:
                    trial_start_time = None
                    if verbose: 
                        print("Warning: 'trial start' not found in trialtablelabels.")

                # (B-2) EMG Data
                if verbose:
                    print("B-2: Extracting EMG data (emgdatabin + emgguide)...")
                emg_data = sub_group['emgdatabin'][()]
                emg_guide = read_object_string(sub_group['emgguide'])
                # Build raw EMG DataFrame
                emg_df = pd.DataFrame(emg_data.T, columns=emg_guide)
                print(emg_df.shape)
                if verbose:
                    print(f"EMG DataFrame shape (raw): {emg_df.shape}")

                # ---- RENAME columns for consistent muscle naming ----
                emg_df = replace_emg_columns(emg_df, GLOBAL_MUSCLE_MAP)

                if verbose:
                    print(f"EMG DataFrame shape (after rename): {emg_df.shape}")
                    print(f"EMG columns: {list(emg_df.columns)}")

                # (B-3) Force Data
                if verbose:
                    print("B-3: Extracting Force data (forcedatabin + forcelabels)...")
                force_data = sub_group['forcedatabin'][()]
                force_labels = read_object_string(sub_group['forcelabels'])

                if not force_labels or all(label.strip() == "0" for label in force_labels):
                    if verbose:
                        print("No valid force labels found; skipping Force data for this task.")
                    force_df = None
                else:
                    expected_force_labels = ["Force_X", "Force_Y"]
                    if force_labels != expected_force_labels:
                        if verbose:
                            print(f"Warning: Unexpected force labels: {force_labels} "
                                  f"(expected {expected_force_labels}). Skipping Force data.")
                        force_df = None
                    else:
                        if force_data.ndim > 1 and force_data.shape[1] != len(force_labels):
                            force_data = force_data.T
                            if verbose:
                                print("Transposed force_data for correct orientation.")
                        if force_data.ndim > 1 and force_data.shape[1] == 1 and len(force_labels) == 2:
                            sample_count = emg_df.shape[0]
                            force_data = np.zeros((sample_count, 2))
                            if verbose:
                                print(f"Created zero-filled force array with shape {force_data.shape}")
                        if verbose:
                            print(f"Force data shape (pre-DF): {force_data.shape}")
                        force_df = pd.DataFrame(force_data, columns=force_labels)
                        if verbose:
                            print(f"Force DataFrame shape: {force_df.shape}")

                # (B-4) Spike Data
                if verbose:
                    print("B-4: Extracting Spike data (spikeratedata + neuronIDs)...")
                spike_rate_data = sub_group['spikeratedata'][()]
                nid_data = sub_group['neuronIDs'][()]

                if nid_data.ndim == 2:
                    if nid_data.shape[0] == 2:
                        nid_data = nid_data[0, :]
                    elif nid_data.shape[1] == 2:
                        nid_data = nid_data[:, 0]

                neuron_ids = [f"neuron{int(x)}" for x in nid_data]
                spike_df = pd.DataFrame(spike_rate_data.T, columns=neuron_ids)
                spike_df = spike_df.round().astype(int)

                valid_neurons = [f'neuron{i}' for i in range(1, 97)]
                spike_df = spike_df[[col for col in spike_df.columns if col in valid_neurons]]

                if verbose:
                    print(f"Spike DataFrame shape: {spike_df.shape}")

                # (B-5) Timeframe & Bin Width
                if verbose:
                    print("B-5: Extracting timeframe and computing bin width...")
                time_frame = sub_group['timeframe'][()]
                time_frame = np.array(time_frame).flatten()
                bin_width_val = None
                if len(time_frame) >= 2:
                    bin_width_val = time_frame[1] - time_frame[0]

                if verbose:
                    print(f"Timeframe length: {len(time_frame)}, bin width: {bin_width_val}")
                # ------------------------------------------------------------------
                # (B-5b)  Trial-target direction  <───  ⬅︎  NEW BLOCK
                # ------------------------------------------------------------------
                # This lives in the same `sub_group` that holds EMG / force / spikes.
                # It is a 1 × N vector of angles (one per trial).
                if 'trial_target_dir' in sub_group:
                    trial_target_dir = np.asarray(sub_group['trial_target_dir'][()]).flatten()
                    if verbose:
                        print(f"trial_target_dir length: {len(trial_target_dir)}")
                else:
                    trial_target_dir = None
                    if verbose:
                        print("Warning: 'trial_target_dir' not found in this task.")

                # (B-6) Build a dictionary for this task
                data_dict = {
                    'year': year,
                    'month': month,
                    'day': day,
                    'date': date_only,
                    'monkey': monkey_name,
                    'task': task_name,
                    'EMG': emg_df,
                    'spike_counts': spike_df,
                    'bin_width': bin_width_val,
                    'time_frame': time_frame,
                    'force': force_df,
                    'trial_start_time': trial_start_time,
                    'trial_target_dir': trial_target_dir,
                }

                rows.append(data_dict)

    # ---- After collecting all tasks/days, unify the EMG columns across all rows ----
    all_cols = set()
    for data_dict in rows:
        all_cols.update(data_dict['EMG'].columns)

    all_cols = sorted(all_cols)

    # Reindex each EMG DataFrame to have the same columns
    for data_dict in rows:
        current_df = data_dict['EMG']
        data_dict['EMG'] = current_df.reindex(columns=all_cols, fill_value=0)

    final_df = pd.DataFrame(rows)

    if verbose:
        print(f"\nFinal DataFrame constructed with {len(final_df)} rows.")
    return final_df

def print_summary(df):
    """Prints a compact summary of the DataFrame."""
    print("\n===== Data Summary =====")
    if df.empty:
        print("DataFrame is empty.")
        return
    monkeys = df['monkey'].unique()
    print(f"Monkeys found: {', '.join(monkeys)}")
    for monkey in monkeys:
        monkey_df = df[df['monkey'] == monkey]
        experiments = monkey_df['date'].unique()
        print(f"\nMonkey '{monkey}':")
        print(f"  Number of experiments: {len(experiments)}")
        for exp in experiments:
            exp_df = monkey_df[monkey_df['date'] == exp]
            task_counts = exp_df['task'].value_counts()
            print(f"  Experiment {exp}:")
            for task, count in task_counts.items():
                force_count = exp_df[(exp_df['task'] == task) & (exp_df['force'].notnull())].shape[0]
                print(f"    Task '{task}': {count} occurrence(s), {force_count} with Force data")
        # Overall tasks for this monkey
        overall_task_counts = monkey_df['task'].value_counts()
        print("  Overall task counts:")
        for task, count in overall_task_counts.items():
            print(f"    {task}: {count}")

def print_detailed_emg_summary(df):
    """
    Prints a detailed row-by-row summary of EMG column headers and dataset shapes.
    """
    print("\n===== Detailed EMG Summary =====")
    if df.empty:
        print("DataFrame is empty.")
        return

    for i, row in df.iterrows():
        monkey = row['monkey']
        date = row['date']
        task = row['task']
        emg_df = row['EMG']
        spike_df = row['spike_counts']
        force_df = row['force']

        print(f"\nRow {i+1}/{len(df)}:")
        print(f"  Monkey: {monkey}, Date: {date}, Task: {task}")

        if emg_df is not None and not emg_df.empty:
            print(f"  EMG columns ({len(emg_df.columns)}): {list(emg_df.columns)}")
            print(f"  EMG DataFrame shape: {emg_df.shape}")
        else:
            print("  No EMG data available.")

        if spike_df is not None and not spike_df.empty:
            print(f"  Spike columns ({len(spike_df.columns)}): {list(spike_df.columns)}")
            print(f"  Spike DataFrame shape: {spike_df.shape}")
        else:
            print("  No spike data available.")

        if force_df is not None:
            print(f"  Force columns ({len(force_df.columns)}): {list(force_df.columns)}")
            print(f"  Force DataFrame shape: {force_df.shape}")
        else:
            print("  No force data available.")

# ---- Example usage ----
if __name__ == '__main__':
    input_file = 'all_manifold_datasets.mat'  # update with your file path
    df_result = process_single_mat_file_new(input_file, verbose=True)
    out_path = os.path.join(os.path.dirname(input_file), 'output.pkl')
    df_result.to_pickle(out_path)

    print(f"\nProcessing complete. DataFrame saved to {out_path}")

    # Print compact summary
    print_summary(df_result)

    # Print detailed EMG summary
    print_detailed_emg_summary(df_result)
