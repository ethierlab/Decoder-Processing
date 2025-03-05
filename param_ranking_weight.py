import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection in older Matplotlib
from collections import defaultdict

output_file = 'experiment_results_lstm_hidden.pkl'
plot_mode = "2D"  # Choose "2D" or "3D"

def main():
    # 1. Load your results
    with open(output_file, 'rb') as f:
        results = pickle.load(f)
    
    # 2. Gather unique values
    unique_N  = sorted(set(exp['N'] for exp in results))
    unique_HD = sorted(set(exp['hidden_dim'] for exp in results))

    if plot_mode == "2D":
        # ----------------------------------------------------------------------
        # (A) 2D PLOTS: One figure per N, VAF vs. k, color-coded by hidden_dim
        # ----------------------------------------------------------------------
        
        for N_val in unique_N:
            # Filter for this N
            exps_for_N = [exp for exp in results if exp['N'] == N_val]
            
            # Group by hidden_dim => dict of hidden_dim -> (k, VAF) pairs
            hidden_dim_dict = defaultdict(list)
            for exp in exps_for_N:
                k    = exp['k']
                vaf  = exp['VAF']
                hdim = exp['hidden_dim']
                hidden_dim_dict[hdim].append((k, vaf))
            
            # Create a new figure
            plt.figure(figsize=(8, 6))
            
            # Plot each hidden_dim as a separate line (or scatter)
            for hdim, kv_pairs in hidden_dim_dict.items():
                # Sort by k
                kv_pairs.sort(key=lambda x: x[0])
                k_vals   = [x[0] for x in kv_pairs]
                vaf_vals = [x[1] for x in kv_pairs]
                plt.scatter(k_vals, vaf_vals, label=f'{hdim}', s=25)
            
            plt.title(f"VAF vs. k (N = {N_val})")
            plt.xlabel("k (Time Lag)")
            plt.ylabel("VAF")
            plt.grid(True)
            
            # Move legend to the right (outside plot)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            
            # Save each figure if desired
            plt.savefig(f'plot_2D_N{N_val}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    elif plot_mode == "3D":
        # ----------------------------------------------------------------------
        # (B) 3D PLOT: Single figure, x=N, y=k, z=VAF, color-coded by hidden_dim
        # ----------------------------------------------------------------------
        
        # Prepare arrays for 3D scatter
        X = []
        Y = []
        Z = []
        C = []  # color-coded by hidden_dim
        
        # We'll map hidden_dim to integer indices so we can use a colormap
        hdim_to_idx = {hd: i for i, hd in enumerate(unique_HD)}
        
        for exp in results:
            X.append(exp['N'])
            Y.append(exp['k'])
            Z.append(exp['VAF'])
            C.append(hdim_to_idx[exp['hidden_dim']])
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use a colormap for hidden_dim
        sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', s=40)
        
        # Axis labels
        ax.set_xlabel("N (PCA Components)")
        ax.set_ylabel("k (Time Lag)")
        ax.set_zlabel("VAF")

        # Create a colorbar that shows hidden_dim
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label("Hidden Dim (index)")
        # Manually set tick labels if desired:
        # cbar.set_ticks(range(len(unique_HD)))
        # cbar.set_ticklabels(unique_HD)
        
        plt.title("3D Visualization: N vs. k vs. VAF (color by hidden_dim)")
        plt.tight_layout()
        
        plt.savefig('plot_3D.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    else:
        print("Invalid plot_mode. Please choose '2D' or '3D'.")

if __name__ == "__main__":
    main()
