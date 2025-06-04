import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mne

file_path = 'Data/2_hand.snirf'

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
    119.046144, 162.398208, 205.94688, 249.495552, 292.847616, 336.19968,
    453.18144, 496.63180800000004, 540.377088, 583.92576,
    627.376128, 744.259584, 788.004864, 831.65184,
    875.003904, 918.4542720000001
]

ends = [
    129.046144, 172.398208, 215.94688, 259.495552, 302.847616,
    419.731072, 463.18144, 506.63180800000004, 550.377088, 593.92576,
    710.90752, 754.259584, 798.004864, 841.65184,
    885.003904, 928.4542720000001
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

# # Load the data
# with h5py.File(file_path, 'r') as f:
#     stim_data = f['nirs/stim_gap/data'][()]
#     onsets = stim_data[:, 0]
#     durations = stim_data[:, 1]
#     ends = onsets + durations

# # Plot events
# plt.figure(figsize=(10, 2))
# for onset, end in zip(onsets, ends):
#     plt.axvspan(onset, end, color='lightblue', alpha=0.5)

# plt.xlabel('Time (s)')
# plt.title('Gap Events from stim_gap')
# plt.yticks([])  # optional: hide y-axis
# plt.grid(True, axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()