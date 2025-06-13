import os
import glob
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# --- Helper Functions for Data Conversion ---

def read_uint16_string(ds):
    """
    Convert an h5py dataset (or numpy array) of uint16 values into a string.
    Ignores any null characters.
    """
    raw = np.array(ds).flatten()
    return ''.join(chr(int(x)) for x in raw if int(x) != 0)

def read_object_string(ds):
    """
    Convert an h5py dataset of type object (often from MATLAB cell arrays)
    into a list of strings.
    
    This version checks if an element is an HDF5 reference and resolves it.
    """
    raw = np.array(ds)
    raw = raw.flatten()
    out = []
    for elem in raw:
        if isinstance(elem, h5py.Reference):
            # Resolve the reference using the file handle from the dataset.
            resolved = ds.file[elem][()]
            # If the resolved data is of type uint16, use the helper function to decode it.
            if isinstance(resolved, np.ndarray) and resolved.dtype == np.uint16:
                out.append(''.join(chr(int(x)) for x in resolved.flatten() if int(x) != 0))
            elif isinstance(resolved, bytes):
                out.append(resolved.decode('utf-8'))
            else:
                out.append(str(resolved))
        elif isinstance(elem, bytes):
            out.append(elem.decode('utf-8'))
        elif isinstance(elem, np.ndarray):
            # If it is a uint16 array, decode it:
            if elem.dtype == np.uint16:
                out.append(''.join(chr(int(x)) for x in elem.flatten() if int(x) != 0))
            else:
                out.append(str(elem))
        else:
            out.append(str(elem))
    return out

def read_scalar(ds):
    """
    Extract a scalar value from an h5py dataset, handling (1,1) arrays.
    """
    val = ds[()]
    if isinstance(val, np.ndarray):
        return val.flatten()[0]
    return val

def extract_date_from_mat(mat_file_path):
    """
    Extracts the date (as a datetime object) from the MATLAB file's meta/dateTime field.
    Expects a format like "2016/9/27 21:1:18.825". Only the date part is used.
    """
    try:
        with h5py.File(mat_file_path, 'r') as f:
            dateTime_str = read_uint16_string(f['xds/meta/dateTime'])
            date_only = dateTime_str.split(' ')[0]  # e.g., "2016/9/27"
            dt = datetime.strptime(date_only, "%Y/%m/%d")
            return dt
    except Exception as e:
        print(f"Error extracting date from {mat_file_path}: {e}")
        return datetime.max

# --- Function to Update Spike Headers Using a Fixed Standard (from Day 0) ---

def update_spike_headers_fixed(new_spike_df, standard_headers):
    """
    Given a new spike_counts DataFrame and the fixed standard_headers (from day 0),
    ensure that new_spike_df has exactly all the standard headers.
    
    Missing columns (neurons) will be added (filled with zeros) and a message is printed.
    Extra columns are dropped (with a printed message).
    The resulting DataFrame's columns are ordered exactly as per standard_headers.
    
    Parameters:
        new_spike_df (pd.DataFrame): New spike counts DataFrame (samples x units).
        standard_headers (list): The fixed, ordered list of unit names from day 0.
        
    Returns:
        new_spike_df (pd.DataFrame): Updated DataFrame with exactly the standard headers.
    """
    # Check for missing neurons in new_spike_df compared to standard_headers
    missing_neurons = [header for header in standard_headers if header not in new_spike_df.columns]
    if missing_neurons:
        for neuron in missing_neurons:
            print(f"Neuron '{neuron}' is missing in this file. Filling with zeros.")
            new_spike_df[neuron] = 0  # add missing column with zeros

    # Check for extra neurons that are not in the standard
    extra_neurons = [col for col in new_spike_df.columns if col not in standard_headers]
    if extra_neurons:
        for neuron in extra_neurons:
            print(f"Extra neuron '{neuron}' found in this file. Dropping it.")
        new_spike_df = new_spike_df.drop(columns=extra_neurons)
    
    # Reorder the columns to match the standard order.
    new_spike_df = new_spike_df[standard_headers]
    
    return new_spike_df

# --- Main Function to Process a Single .mat File ---
# If standard_headers is None, this file is assumed to be day 0.
def process_mat_file(mat_file_path, output_dir, standard_headers=None):
    try:
        with h5py.File(mat_file_path, 'r') as f:
            # --- META INFORMATION ---
            dateTime_str = read_uint16_string(f['xds/meta/dateTime'])
            date_only = dateTime_str.split(' ')[0]  # e.g., "2016/9/27"
            try:
                year, month, day = date_only.split('/')
            except ValueError:
                year, month, day = None, None, None

            monkey_name = read_uint16_string(f['xds/meta/monkey'])
            task_name   = read_uint16_string(f['xds/meta/task'])
            
            # --- DATA ARRAYS ---
            EMG_data = f['xds/EMG'][()]  # shape: (channels, samples)
            EMG_names = read_object_string(f['xds/EMG_names'])
            # Combine EMG data with headers (transpose so that rows = samples)
            emg_df = pd.DataFrame(EMG_data.T, columns=EMG_names)
            
            # Spike counts: shape (n_units, samples)
            spike_counts = f['xds/spike_counts'][()]
            # Unit names: stored in xds/unit_names
            unit_names = read_object_string(f['xds/unit_names'])
            # Create a DataFrame for spike counts: transpose so that rows = samples, columns = units.
            spike_df = pd.DataFrame(spike_counts.T, columns=unit_names)

            if spike_df.empty:
                print(f"[WARNING] {mat_file_path} => 'spike_counts' is empty (shape={spike_df.shape}).")
            
            # If this is the first (day 0) file, use its unit names as the fixed standard.
            if standard_headers is None:
                standard_headers = list(unit_names)
            else:
                # For subsequent files, update spike_df to match the fixed standard.
                spike_df = update_spike_headers_fixed(spike_df, standard_headers)
            
            print(f"[DEBUG] After fixing headers => {mat_file_path}, spike_df shape={spike_df.shape}")
            
            # Process force data
            force_data = f['xds/force'][()]
            if force_data.ndim > 1:
                # Check orientation: we want time along rows and 2 columns for x and y.
                # If the number of rows is less than the number of columns, assume force_data is (2, T)
                if force_data.shape[0] < force_data.shape[1]:
                    force_data = force_data.T
                # Now force_data should be (T,2) — convert it to a DataFrame with column labels.
                force_data = pd.DataFrame(force_data, columns=['x', 'y'])
            
            # --- MAIN TREE VALUES ---
            bin_width_val = read_scalar(f['xds/bin_width'])
            trial_start_time = f['xds/trial_start_time'][()]  # may be an array
            trial_start_time = np.squeeze(trial_start_time)
            # Use the provided time frame from the file.
            time_frame = f['xds/time_frame'][()]
            time_frame = np.array(time_frame).flatten()

            trial_target_dir = f['xds/trial_target_dir'][()]    # (n_trials,)
            trial_target_dir = np.squeeze(trial_target_dir)
            # --- Assemble Data Dictionary ---
            data_dict = {
                'year': year,
                'month': month,
                'day': day,
                'date': date_only,
                'monkey': monkey_name,
                'task': task_name,
                'EMG': emg_df,            # Combined EMG DataFrame with headers
                'spike_counts': spike_df, # Spike counts DataFrame with fixed standard headers
                'bin_width': bin_width_val,
                'time_frame': time_frame,
                'force': force_data,      # Now, force_data is a DataFrame with columns 'x' and 'y'
                'trial_start_time': trial_start_time,
                'trial_target_dir': trial_target_dir
            }
            
            # Create a DataFrame (one row for this file/trial)
            df = pd.DataFrame([data_dict])
        
        # --- Save DataFrame as a Pickle File ---
        base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
        monkey_clean = monkey_name.replace(" ", "_")
        output_file = os.path.join(output_dir, f"{base_name}.pkl")
        df.to_pickle(output_file)
        print(f"Processed {mat_file_path} -> {output_file}")
        return standard_headers  # Return fixed standard headers (for day 0) if set
    except Exception as e:
        print(f"Error processing {mat_file_path}: {e}")
        return standard_headers


# --- Function to Process All .mat Files (Sorted by Date) with a Fixed Standard ---
def process_all_mat_files_fixed(input_dir, output_dir):
    # Recursively find all .mat files.
    mat_files = glob.glob(os.path.join(input_dir, '**', '*.mat'), recursive=True)
    print(f"Found {len(mat_files)} .mat files.")
    
    # Create list of tuples (file_path, date) and sort by date (oldest first)
    mat_files_with_dates = [(f, extract_date_from_mat(f)) for f in mat_files]
    mat_files_with_dates.sort(key=lambda x: x[1])
    sorted_files = [t[0] for t in mat_files_with_dates]
    print(sorted_files)
    standard_headers = None
    for i, mat_file in enumerate(sorted_files):
        if i == 0:
            # Use the first (oldest) file to set the standard.
            standard_headers = process_mat_file(mat_file, output_dir, standard_headers)
            print("Standard unit headers (from day 0):", standard_headers)
        else:
            # Process subsequent files using the fixed standard.
            process_mat_file(mat_file, output_dir, standard_headers)
    
    print("Final fixed standard unit headers:", standard_headers)

# === Usage ===
if __name__ == '__main__':
    input_directory = 'C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015'
    output_directory = input_directory
    process_all_mat_files_fixed(input_directory, output_directory)
