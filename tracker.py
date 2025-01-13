import pickle
import time
import os

output_file = 'experiment_results_lstm.pkl'

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"{file_path} does not exist yet.")
        return []

def display_latest_results(results, num=5):
    if not results:
        print("No results to display.")
        return
    print(f"Displaying the latest {min(num, len(results))} results:")
    for res in results[-num:]:
        print(res)

while True:
    results = load_results(output_file)
    display_latest_results(results)
    print(f"Total completed experiments: {len(results)}")
    print("Waiting for 5000 seconds before next update...\n")
    time.sleep(5000)  # Wait for 60 seconds before checking again