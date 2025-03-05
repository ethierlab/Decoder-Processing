import pickle
import os

def load_pkl_file(file_path):
    """
    Loads a pickle file and returns its content.

    Parameters:
    - file_path (str): Path to the .pkl file.

    Returns:
    - data (list of dict): List containing dictionaries with parameters.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        if not isinstance(data, list):
            print(f"Error: Data in '{file_path}' is not a list. Skipping this file.")
            return None
        return data
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None

def get_top_n_vaf(data, n=3):
    """
    Retrieves the top N entries with the highest VAF values from the data.

    Parameters:
    - data (list of dict): List containing dictionaries with parameters.
    - n (int): Number of top entries to retrieve based on VAF.

    Returns:
    - top_entries (list of dict): Top N dictionaries sorted by VAF descending.
    """
    # Filter out entries that do not have 'VAF' key or have non-numeric VAF
    valid_entries = []
    for idx, entry in enumerate(data, start=1):
        if isinstance(entry, dict):
            vaf = entry.get('VAF', None)
            if isinstance(vaf, (int, float)):
                valid_entries.append(entry)
            else:
                print(f"Warning: Entry {idx} in file skipped due to invalid or missing 'VAF': {entry}")
        else:
            print(f"Warning: Non-dictionary entry at index {idx} skipped: {entry}")

    if not valid_entries:
        print("No valid entries with 'VAF' found in this file.")
        return []

    # Sort the entries by 'VAF' in descending order
    sorted_entries = sorted(valid_entries, key=lambda x: x['VAF'], reverse=True)

    # Return top N entries
    return sorted_entries[:n]

def print_entry(entry, rank):
    """
    Prints the dictionary entries in a readable format.

    Parameters:
    - entry (dict): Dictionary containing parameters.
    - rank (int): Rank of the entry based on VAF.
    """
    print(f"\n--- Rank {rank}: VAF = {entry['VAF']} ---")
    for key, value in entry.items():
        print(f"{key}: {value}")

def process_files(file_list, top_n=3):
    """
    Processes each file to find and print the top N VAF entries.

    Parameters:
    - file_list (list of str): List of .pkl file paths.
    - top_n (int): Number of top entries to retrieve based on VAF.
    """
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing File: {file_name}")
        data = load_pkl_file(file_path)

        if data is None:
            print(f"Skipping '{file_name}' due to loading errors.")
            continue

        top_entries = get_top_n_vaf(data, top_n)

        if not top_entries:
            print(f"No valid VAF entries found in '{file_name}'.")
            continue

        for idx, entry in enumerate(top_entries, start=1):
            print_entry(entry, idx)

def main():
    """
    Main function to execute the script.
    """
    # Define your list of .pkl file paths here
    pkl_files = [
        'experiment_results_gru_hidden_long_2.pkl',
        'experiment_results_lstm_hidden.pkl',
        # 'path/to/file3.pkl',
        # Add more file paths as needed
    ]

    # Verify that all provided paths are valid files
    valid_files = []
    for file_path in pkl_files:
        if os.path.isfile(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: '{file_path}' is not a valid file and will be skipped.")

    if not valid_files:
        print("No valid .pkl files to process. Exiting.")
        return

    # Process each file to find top N VAF entries
    process_files(valid_files, top_n=3)

if __name__ == "__main__":
    main()
