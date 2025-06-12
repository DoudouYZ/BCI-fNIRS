from itertools import compress
import numpy as np
import mne
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import h5py

# Yuxuan : 1
# Oscar : 2
# Nikolaj : 3

# Load the data
file_path = '2_hand.snirf'


with h5py.File(file_path, 'r') as f:
    stim1_data = None
    control_data = None

    for key in f['nirs']:
        group = f['nirs'][key]
        if 'name' in group:
            label = group['name'][()].decode()
            if label == '1':
                stim1_data = group['data'][()]
            elif label == 'control':
                control_data = group['data'][()]

# Plot both stimuli
plt.figure(figsize=(12, 2))

# Plot stim1 in orange
if stim1_data is not None:
    onsets = stim1_data[:, 0]
    durations = stim1_data[:, 1]
    for onset, duration in zip(onsets, durations):
        plt.axvspan(onset, onset + duration, color='orange', alpha=0.5, label='activation')

# Plot control in lightblue
if control_data is not None:
    onsets = control_data[:, 0]
    durations = control_data[:, 1]
    for onset, duration in zip(onsets, durations):
        plt.axvspan(onset, onset + duration, color='lightblue', alpha=0.5, label='control')

# Prevent duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())

plt.xlabel('Time (s)')
plt.title('Stimulus 1 and Control Events')
plt.yticks([])
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()