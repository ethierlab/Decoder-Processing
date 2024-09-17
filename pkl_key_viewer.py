import pickle

def print_dict_keys(pkl_file_path, level=0):
    """
    Recursively prints the keys in a dictionary stored in a pickle file.

    Parameters:
    - pkl_file_path (str): The path to the pickle file.
    - level (int): The current level in the dictionary, used for indentation.
    """
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    def recurse_keys(d, level):
        indent = "  " * level
        if isinstance(d, dict):
            for key, value in d.items():
                print(f"{indent}- {key}")
                if isinstance(value, dict):
                    recurse_keys(value, level + 1)
        else:
            print(f"{indent}- (Non-dict type found)")

    print(f"\nKeys in {pkl_file_path}:")
    recurse_keys(data, level)
    print("\n" + "="*50 + "\n")

# Example usage with all files
file_paths = ['kinematics.pkl', 'tdt_signals.pkl', 'experiment_data.pkl']

for file_path in file_paths:
    print_dict_keys(file_path)

