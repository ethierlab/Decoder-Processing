import pickle
import numpy as np
import pandas as pd

def print_dict_info(pkl_file_path, level=0):
    """
    Recursively prints detailed information about the keys in a dictionary 
    stored in a pickle file, including data types and shapes/lengths when available.
    
    Parameters:
    - pkl_file_path (str): The path to the pickle file.
    - level (int): The current indentation level.
    """
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    def recurse_info(d, level):
        indent = "  " * level
        if isinstance(d, dict):
            for key, value in d.items():
                print(f"{indent}- {key}")
                
                # Prepare indentation for the next lines of info
                value_indent = "  " * (level + 1)
                
                # Print the type of the value
                print(f"{value_indent}Type: {type(value).__name__}")
                
                # Handle different types
                if isinstance(value, dict):
                    # If dictionary, print length and recurse
                    print(f"{value_indent}Length: {len(value)}")
                    recurse_info(value, level + 1)
                elif isinstance(value, list):
                    # If list, print length and possibly first element info
                    print(f"{value_indent}Length: {len(value)}")
                    if len(value) > 0:
                        # Optional: show type of first element
                        print(f"{value_indent}First element type: {type(value[0]).__name__}")
                elif isinstance(value, (str, bytes)):
                    # For strings or bytes, print length
                    print(f"{value_indent}Length: {len(value)}")
                elif isinstance(value, np.ndarray):
                    # For NumPy arrays, print shape and dtype
                    print(f"{value_indent}Shape: {value.shape}")
                    print(f"{value_indent}Dtype: {value.dtype}")
                elif isinstance(value, (pd.DataFrame, pd.Series)):
                    # For pandas DataFrame/Series, print shape, columns (if DF)
                    print(f"{value_indent}Shape: {value.shape}")
                    if isinstance(value, pd.DataFrame):
                        print(f"{value_indent}Columns: {value.columns.tolist()}")
                else:
                    # For other types, try to display length if possible
                    # For example, if it's a tuple or set
                    if hasattr(value, '__len__'):
                        print(f"{value_indent}Length: {len(value)}")
                    
                # Print a separator line for clarity between keys at the same level
                print()

        else:
            # If the top-level object is not a dict, handle it here
            print(f"{indent}- (Non-dict type: {type(d).__name__})")
            if hasattr(d, '__len__'):
                print(f"{indent}  Length: {len(d)}")
            if isinstance(d, np.ndarray):
                print(f"{indent}  Shape: {d.shape}, Dtype: {d.dtype}")
            elif isinstance(d, (pd.DataFrame, pd.Series)):
                print(f"{indent}  Shape: {d.shape}")
                if isinstance(d, pd.DataFrame):
                    print(f"{indent}  Columns: {d.columns.tolist()}")

    print(f"\nDetailed structure of {pkl_file_path}:")
    recurse_info(data, level)
    print("\n" + "="*50 + "\n")


# Example usage with files
# file_paths = ['Jango_dataset.pkl', 'projected_data_test.pkl']
# file_paths = ['Jango_dataset.pkl']
file_paths = ['experiment_results_lstm.pkl']

for file_path in file_paths:

    print_dict_info(file_path)
