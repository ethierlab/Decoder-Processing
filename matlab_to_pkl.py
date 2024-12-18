import scipy.io
import pickle
import os

import scipy.io
import pickle
import os

# Nom du fichier .mat à lire
mat_file = 'Jango_2013-12-09_WF_001_bin.mat'

# Chargement du fichier .mat
data = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

# On suppose que les données sont organisées de la même manière que dans binnedData standard
# binnedData.spikeratedata -> (N x M) : N = nombre d'échantillons (ex: 24000), M = nombre de canaux
# binnedData.trialtable -> table d'essai, première colonne = "Event time"
# binnedData.forcedatabin -> (N x 2) dimension, avec la première colonne = force x, deuxième = force y
binnedData = data['binnedData']

# Récupération des données de spikes, trialtable et force
spikeratedata = binnedData.spikeratedata
trialtable = binnedData.trialtable
forcedatabin = binnedData.forcedatabin

# Construction du dictionnaire pour les données de spikes
# Format souhaité:
# {
#    "data": {
#       "Channel01": {"ID_ch1#2": array_of_values}, 
#       "Channel02": {"ID_ch2#2": array_of_values},
#       ...
#    },
#    "Event time": array_des_event_times (colonne 0 du trialtable)
# }

spike_dict = {}
spike_dict['data'] = {}
num_channels = spikeratedata.shape[1]

for i in range(num_channels):
    channel_name = f"Channel{i+1:02d}"  # Channel01, Channel02, ...
    id_name = f"ID_ch{i+1}#2"
    # Extraire la colonne i des spikerates
    spike_values = spikeratedata[:, i]
    spike_dict['data'][channel_name] = {id_name: spike_values}
print(spike_dict['data'])
# Ajout de la clé "Event time" à partir de la première colonne du trialtable
event_time = trialtable[:, 0]
spike_dict['Event time'] = event_time
print(len(event_time))

# Construction du dictionnaire pour les données de force
# Format souhaité:
# {
#    "Force": {
#       "x": force_x (N long),
#       "y": force_y (N long)
#    }
# }

force_x = forcedatabin[:, 0]
force_y = forcedatabin[:, 1]

force_dict = {
    "Force": {
        "x": force_x,
        "y": force_y
    }
}
print(len(force_x))
# Sauvegarde du dictionnaire force dans un fichier pkl
with open('force_data.pkl', 'wb') as f:
    pickle.dump(force_dict, f)

# Sauvegarde du dictionnaire spikerate dans un fichier pkl
with open('spikerate_data.pkl', 'wb') as f:
    pickle.dump(spike_dict, f)
