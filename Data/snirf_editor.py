import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mne

file_path = 'Data/1_tongue.snirf'

# Delete existing 'control' group if it exists
with h5py.File(file_path, 'r+') as f:
    # ✅ Now it's safe to use `f`
    for key in list(f['nirs']):
        if key.startswith('stim'):
            name = f['nirs'][key]['name'][()].decode()
            if name == 'control':
                del f['nirs'][key]
                print(f"Deleted existing '{key}' group with name 'control'.")

# Print all existing events
with h5py.File(file_path, 'r') as f:
    for key in f['nirs']:
        if key.startswith('stim'):
            stim_group = f['nirs'][key]
            name = stim_group['name'][()].decode()
            data = stim_group['data'][()]
            print(f"\nStimulus: {name}")
            for i, (onset, duration, amplitude) in enumerate(data):
                end_time = onset + duration
                print(f"  Event {i+1}: Start = {onset}, End = {end_time}, Duration = {duration}, Amplitude = {amplitude}")

# Stimulus 1 event times
starts = [
    122.38848, 165.44563200000002, 208.207872, 251.16672, 294.22387200000003,
    337.18272, 453.18144, 496.23859200000004, 539.099136,
    582.057984, 625.115136, 741.0155520000001,
    783.9744000000001, 827.031552, 869.892096, 912.850944
]

ends = [
    132.38848000000002, 175.44563200000002, 218.207872, 261.16672, 304.22387200000003,
    420.222592, 463.18144, 506.23859200000004, 549.099136,
    592.057984,  708.1550080000001, 751.0155520000001,
    793.9744000000001, 837.031552, 879.892096, 922.850944
]

# Calculate control events centered in gaps
control_events = []
for i in range(len(ends) - 1):
    center = (ends[i] + starts[i + 1]) / 2
    onset = center - 5.0  # 10s duration centered on gap
    duration = 10.0
    control_events.append([onset, duration, 1.0])

control_events = np.array(control_events)

# Add to SNIRF file
with h5py.File(file_path, 'r+') as f:
    # Remove existing "control" group if needed
    for key in list(f['nirs']):
        if key.startswith('stim'):
            name = f['nirs'][key]['name'][()].decode()
            if name == 'control':
                del f['nirs'][key]
                print(f"Deleted existing '{key}' group with name 'control'.")

    # Create new control group
    stim_keys = [k for k in f['nirs'].keys() if k.startswith('stim')]
    next_index = max([int(k[4:]) for k in stim_keys]) + 1
    stim_name = f'stim{next_index}'

    stim_control = f['nirs'].create_group(stim_name)
    stim_control.create_dataset('name', data=np.string_('control'))
    stim_control.create_dataset('data', data=control_events)

print(f"✅ Created '{stim_name}' with label 'control', centered in gaps, all 10s long.")

# Load the data

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
        plt.axvspan(onset, onset + duration, color='orange', alpha=0.5, label='stim 1')

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