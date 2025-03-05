import torch
import torch.nn as nn
import pickle
# --- 1) Define your model code (same as in training) ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def count_gru_params(input_size, hidden_size, num_layers=1):
    model = GRUModel(input_size, hidden_size, num_layers)
    return sum(p.numel() for p in model.parameters())
file_path = 'experiment_results_gru_hidden_long.pkl'
with open(file_path, 'rb') as f:
            results = pickle.load(f)
# --- 2) Suppose this is your list of runs (already done) ---
runs = results

NUM_LAYERS = 1  # or however many you used

# --- 3) Augment each run with 'num_params' ---
for run in runs:
    # Your actual formula for 'input_size' depends on how you fed data into GRU.
    # If it was just run['N']:
    input_size = run['N']
    # If it was run['N'] * run['k'], then do that instead:
    # input_size = run['N'] * run['k']
    # etc.

    hidden_size = run['hidden_dim']

    run['num_params'] = count_gru_params(input_size, hidden_size, NUM_LAYERS)
output_file = 'experiment_results_gru_hidden_long_2.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results, f)
# Now each dictionary in 'runs' has 'num_params' set
print(runs)
