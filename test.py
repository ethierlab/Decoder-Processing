import pandas as pd
import numpy as np

def inspect_pkl(file_path):
    # Load the object from the pickle file
    try:
        obj = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"\n--- Loaded object from {file_path} ---")
    print("Type:", type(obj))
    
    # If the object has a shape attribute (e.g. DataFrame or numpy array)
    if hasattr(obj, 'shape'):
        print("Shape:", obj.shape)
    else:
        try:
            # For lists, print the length
            print("Length:", len(obj))
        except Exception as e:
            print("No shape or length available.")
    
    # Detailed inspection if object is a DataFrame:
    if isinstance(obj, pd.DataFrame):
        print("\nColumns:")
        print(list(obj.columns))
        print("\n--- DataFrame Info ---")
        obj.info()
        print("\n--- DataFrame Head ---")
        print(obj.head())
        print("\n--- Column Data Types and Sample Shapes ---")
        for col in obj.columns:
            sample = obj[col].iloc[0]
            if isinstance(sample, np.ndarray):
                print(f"Column '{col}': numpy.ndarray, shape {sample.shape}")
            else:
                print(f"Column '{col}': type {type(sample)}")
    # If the object is a list, inspect the first 10 elements:
    elif isinstance(obj, list):
        print(f"\nThe loaded object is a list with {len(obj)} elements.")
        num_to_print = min(20, len(obj))
        for i in range(num_to_print):
            item = obj[i]
            print(f"\n--- Element {i} ---")
            print("Type:", type(item))
            if hasattr(item, 'shape'):
                print("Shape:", item.shape)
            else:
                # Print a truncated representation if it's long
                print("Content:", str(item)[:200])
    else:
        print("\nObject content:", obj)
        
    print("\n--- End of Inspection ---")
    return obj

if __name__ == '__main__':
    # Change this to your pkl file path or prompt for input:
    file_path = "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Pop_PG_2021/Pop_20210602_pg.pkl"
    inspect_pkl(file_path)
