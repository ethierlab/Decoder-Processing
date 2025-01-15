import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load data
    output_file = 'experiment_results2.pkl'
    with open(output_file, 'rb') as f:
        results = pickle.load(f)
    
    # 2. Convert to a DataFrame
    df = pd.DataFrame(results)
    
    # 3. Filter so we keep only hidden_dim in [32, 64]
    df_filtered = df[df['hidden_dim'].isin([32, 64])]
    
    # 4. Scatterplot, coloring by hidden_dim
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_filtered, 
        x='num_params', 
        y='VAF', 
        hue='hidden_dim',
        alpha=0.7
    )
    
    # 5. Optionally add separate regression lines for each hidden_dim
    # (we do regplot separately for each subset)
    df_32 = df_filtered[df_filtered['hidden_dim'] == 32]
    df_64 = df_filtered[df_filtered['hidden_dim'] == 64]
    
    sns.regplot(
        data=df_32, 
        x='num_params', 
        y='VAF', 
        scatter=False, 
        ci=None, 
        color='red', 
        label='hdim=32 Trend'
    )
    sns.regplot(
        data=df_64, 
        x='num_params', 
        y='VAF', 
        scatter=False, 
        ci=None, 
        color='green', 
        label='hdim=64 Trend'
    )
    
    plt.title("VAF vs. num_params (Hidden Dim = 32 vs. 64)")
    plt.xlabel("Number of Parameters")
    plt.ylabel("VAF")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
