import pickle
import os
import math

output_file = 'experiment_results_gru_seeds_PCA.pkl'
PAGE_SIZE = 10

def load_results(file_path):
    """Load the list of experiments from the pickle file."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"{file_path} does not exist yet.")
        return []

def save_results(file_path, results):
    """Save the results list to the pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def display_page(results, page, page_size=PAGE_SIZE):
    """
    Display experiments in a page.
    The experiments are shown in their original order (oldest first),
    but pagination is computed from the end (most recent experiments).
    """
    total = len(results)
    if total == 0:
        print("No results to display.")
        return
    
    # Compute the slice indices so that page=0 shows the last PAGE_SIZE experiments.
    start_index = max(0, total - (page + 1) * page_size)
    end_index = total - page * page_size
    print(f"\nDisplaying experiments {start_index + 1} to {end_index} (of {total}):")
    for i in range(start_index, end_index):
        print(f"\n--- Experiment {i + 1} ---")
        for key, value in results[i].items():
            print(f"{key}: {value}")

def remove_from_page(results, page, num_to_remove, page_size=PAGE_SIZE):
    """
    Remove the last num_to_remove experiments from the currently displayed page.
    (i.e. from the end of that page)
    Returns the new results list.
    """
    total = len(results)
    start_index = max(0, total - (page + 1) * page_size)
    end_index = total - page * page_size

    displayed_count = end_index - start_index
    if displayed_count == 0:
        print("No experiments are displayed on this page to remove from.")
        return results

    if num_to_remove > displayed_count:
        print(f"Cannot remove {num_to_remove} experiments; only {displayed_count} are displayed.")
        return results

    # We remove the experiments from the original list.
    # The ones at the "end" of the displayed page are the ones with indices:
    # from (end_index - num_to_remove) to end_index.
    removal_start = end_index - num_to_remove
    print(f"Removing experiments {removal_start + 1} to {end_index} (the last {num_to_remove} in this page).")
    # Remove those experiments:
    new_results = results[:removal_start] + results[end_index:]
    return new_results

def main():
    page = 0  # Page 0 will show the last PAGE_SIZE experiments (most recent)
    while True:
        results = load_results(output_file)
        total = len(results)
        if total == 0:
            print("No results available.")
        else:
            total_pages = math.ceil(total / PAGE_SIZE)
            # Adjust the current page if needed.
            if page >= total_pages:
                page = total_pages - 1
            print("\n" + "="*50)
            print(f"Page {page + 1} of {total_pages}")
            display_page(results, page, PAGE_SIZE)
        print("\nOptions:")
        print(" [n] Next page (more recent experiments)")
        print(" [p] Previous page (older experiments)")
        print(" [r] Remove experiments from the end of the current page")
        print(" [q] Quit")
        choice = input("Enter your choice: ").strip().lower()

        if choice == 'n':
            if total == 0:
                print("No results available.")
            else:
                if page > 0:
                    page -= 1
                else:
                    print("Already at the most recent experiments (page 1).")
        elif choice == 'p':
            if total == 0:
                print("No results available.")
            else:
                if page < total_pages - 1:
                    page += 1
                else:
                    print("Already at the oldest experiments.")
        elif choice == 'r':
            num_str = input("Enter the number of experiments (from the end of the displayed page) to remove: ").strip()
            try:
                num_to_remove = int(num_str)
                if num_to_remove <= 0:
                    print("Please enter a positive number.")
                    continue
                new_results = remove_from_page(results, page, num_to_remove, PAGE_SIZE)
                save_results(output_file, new_results)
                print("Experiments removed. File updated.")
            except ValueError:
                print("Invalid number entered. Please try again.")
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please choose n, p, r, or q.")

if __name__ == '__main__':
    main()
