import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

output_file = 'experiment_results1.pkl'

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"{file_path} does not exist yet.")
        return []

def plot_results(results, x_param='N', metrics=['test_loss', 'R2', 'VAF']):
    """
    Plot given metrics as a function of one of the parameters (e.g., N, k, etc.)
    
    Parameters:
        results (list of dict): The experimental results.
        x_param (str): The key in the results dictionary to use for the x-axis.
        metrics (list of str): The keys in the results dictionary to plot on the y-axis.
    """
    if not results:
        print("No results to plot.")
        return

    # Extract x values and metrics
    x_values = [res[x_param] for res in results]
    
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        y_values = [res[metric] for res in results]
        plt.scatter(x_values, y_values, marker='o', label=metric)
    
    plt.title(f'Metrics vs. {x_param}')
    plt.xlabel(x_param)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    results = load_results(output_file)
    if not results:
        return
    # Gather all keys from every dictionary in results
    all_keys = set()
    for exp_dict in results:
        all_keys.update(exp_dict.keys())

    # Now all_keys contains all possible keys (parameters, metrics, etc.)
    print("Possible x-parameters (keys) are:")
    for key in all_keys:
        print(key)
    # Example 1: Plot all metrics versus N parameters
    plot_results(results, x_param='N', metrics=[ 'R2', 'VAF'])
    # Example 2: Plot all metrics versus k past time steps
    plot_results(results, x_param='k', metrics=['R2', 'VAF'])
    # Example 3: Plot all metrics versus learning rate (stable at the moment)
    plot_results(results, x_param='learning_rate', metrics=[ 'R2', 'VAF'])
    # Example 4: Plot all metrics versus hidden dimensions
    plot_results(results, x_param='hidden_dim', metrics=[ 'R2', 'VAF'])
    # Example 5: Plot all metrics versus N
    plot_results(results, x_param='batch_size', metrics=[ 'R2', 'VAF'])
    # # Example 3: If you just want to see how results changed over time (e.g., 
    # # by experiment index), you can do:
    # experiment_indices = np.arange(1, len(results)+1)
    # plt.figure(figsize=(10, 6))
    # #plt.scatter(experiment_indices, [r['test_loss'] for r in results], marker='o', label='Test Loss')
    # plt.scatter(experiment_indices, [r['R2'] for r in results], marker='o', label='R2')
    # plt.scatter(experiment_indices, [r['VAF'] for r in results], marker='o', label='VAF')
    # plt.title('Metrics over Experiments')
    # plt.xlabel('Experiment Index')
    # plt.ylabel('Metric Value')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()