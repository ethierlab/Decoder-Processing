import pickle
import os

output_file = 'experiment_results1.pkl'

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"{file_path} does not exist yet.")
        return []

def rank_experiments_single_objective_with_priority(results):
    # Sort primarily by test_loss ascending
    results_sorted = sorted(results, key=lambda x: x['test_loss'])
    return results_sorted

def print_top_experiments_simple(results_sorted, top_n=10):
    # Among the top_n with the lowest test_loss, pick best R2 or VAF
    top_subset = results_sorted[:top_n]
    # Sort that subset by R2 descending (or VAF) to pick the single best
    best_in_subset = sorted(top_subset, key=lambda x: x['R2'], reverse=True)[0]

    print("Best experiment from the top_n test_loss subset:")
    print(best_in_subset)

def main():
    results = load_results(output_file)
    if not results:
        return

    # Sort all experiments by test_loss ascending
    results_sorted = rank_experiments_single_objective_with_priority(results)
    
    # Print the top 10 by test_loss
    print("Top 10 experiments by lowest test_loss:")
    for i, exp in enumerate(results_sorted[:10], start=1):
        print(f"{i}. {exp}")

    # Then find the single best within those top 10 by R2
    print_top_experiments_simple(results_sorted, top_n=10)

if __name__ == "__main__":
    main()