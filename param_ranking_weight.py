import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting in matplotlib

output_file = 'experiment_results1.pkl'

def compute_score(exp, alpha=1.0, beta=1.0, gamma=1.0):
    loss = exp['test_loss']
    r2   = exp['R2']
    vaf  = exp['VAF']
    score = alpha * (1 - loss) + beta * r2 + gamma * vaf
    return score

def main():
    # 1. Load your results
    with open(output_file, 'rb') as f:
        results = pickle.load(f)
    
    # 2. Create arrays for (X, Y, Z) and color
    X = []
    Y = []
    Z = []
    colors = []  # e.g., hidden_dim
    
    # Adjust these weights as you like
    alpha, beta, gamma = 1.0, 1.0, 1.0
    
    for exp in results:
        N = exp['N']
        k = exp['k']
        hidden_dim = exp['hidden_dim']
        
        # Compute the score
        sc = compute_score(exp, alpha=alpha, beta=beta, gamma=gamma)
        
        X.append(N)
        Y.append(k)
        Z.append(sc)
        colors.append(hidden_dim)  # or some other variable you want to visualize
    
    # 3. Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(X, Y, Z, c=colors, cmap='viridis', s=50)
    ax.set_xlabel("N (PCA Components)")
    ax.set_ylabel("k (Time Lag)")
    ax.set_zlabel("Score (higher is better)")
    
    # Add colorbar to show hidden_dim scale
    cbar = plt.colorbar(sc)
    cbar.set_label("Hidden Dim")
    
    plt.title("3D Visualization of Hyperparameters vs. Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()