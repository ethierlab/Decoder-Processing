import os
import pandas as pd
import numpy as np
import pickle

# Fixed paths and constants
BASE_PATH = "T:/Projects/ForcePredictBMI/296/18-07-2024/Images/Experience_2024-07-18_12-55-47_296"
DOC_MODEL = "Docs_from_model"
MODEL_NAME = 'DLC_Resnet50_Force_predictJun26shuffle1_snapshot_015'
EXT = ".h5"
LIKELIHOOD_THRESHOLD = 0.6
SAVING_PATH = "C:/Users/Ethier Lab/Desktop"

def extract_data(file_path, model_name, likelihood_threshold=0.6):
    """Extract data from the DLC model HDF file."""
    data = pd.read_hdf(file_path)
    bodyparts = ['start', 'middle', 'tip', 'angle_left', 'angle_right']
    coords = ['x', 'y', 'likelihood']
    
    # Initialize the dictionary structure
    results = {coord: {bp: [] for bp in bodyparts} for coord in coords}

    for bp in bodyparts:
        for coord in coords:
            values = np.array(data[(model_name, bp, coord)])
            if coord == 'likelihood':
                values[values < likelihood_threshold] = np.nan
            results[coord][bp] = values.tolist()  # Store the entire list, not individual points
    
    return results

def process_all_trials(base_path, doc_model, model_name, ext=".h5", likelihood_threshold=0.6):
    """Process all trials and store them in a dictionary."""
    all_trials = {coord: {bp: [] for bp in ['start', 'middle', 'tip', 'angle_left', 'angle_right']} for coord in ['x', 'y', 'likelihood']}
    trial_index = 1
    
    while True:
        path_data = os.path.join(base_path, doc_model, f"Essai{trial_index}{model_name}{ext}")
        if not os.path.exists(path_data):
            print(f"No more files found after Essai{trial_index - 1}. Stopping.")
            break

        print(f"Processing: {path_data}")
        trial_data = extract_data(path_data, model_name, likelihood_threshold)
        
        for coord in all_trials:
            for bp in all_trials[coord]:
                all_trials[coord][bp].append(trial_data[coord][bp])  # Append the entire trial data
        
        trial_index += 1
    
    return all_trials

def save_to_pickle(data, save_path, filename="kinematics.pkl"):
    """Save the dictionary to a pickle file."""
    file_path = os.path.join(save_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")
    
    # Print a small part of the data to verify structure
    print("Sample of the saved data:")
    for coord, bodyparts in data.items():
        print(f"Coordinate: {coord}")
        for bp, trials in bodyparts.items():
            print(f"  Body part: {bp}, Number of trials: {len(trials)}, First trial data length: {len(trials[0]) if trials else 'No data'}")
        print()  # For better readability

if __name__ == "__main__":
    # Process all trials and save the results
    trials_data = process_all_trials(BASE_PATH, DOC_MODEL, MODEL_NAME, EXT, LIKELIHOOD_THRESHOLD)
    save_to_pickle(trials_data, SAVING_PATH)
