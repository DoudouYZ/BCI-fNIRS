import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Data/2_dummy_hand.snirf'

# Delete existing 'control' group if it exists
with h5py.File(file_path, 'r+') as f:
    for key in list(f['nirs']):
        if not key.startswith('stim'):
            continue
        grp = f['nirs'][key]
        # only proceed if this group really has a 'name' dataset
        if 'name' not in grp:
            continue
        name = grp['name'][()].decode()
        if name == 'control':
            del f['nirs'][key]
            print(f"Deleted existing '{key}' group named 'control'.")

# Print all existing events
with h5py.File(file_path, 'r') as f:
    for key in f['nirs']:
        if not key.startswith('stim'):
            continue
        grp = f['nirs'][key]
        # skip any stim* group that doesn’t have both 'name' and 'data'
        if 'name' not in grp or 'data' not in grp:
            print(f"Skipping {key!r}: missing 'name' or 'data'")
            continue

        label = grp['name'][()].decode()
        data  = grp['data'][()]
        print(f"\nStimulus: {label!r}  (group {key})")
        for i, (onset, duration, amp) in enumerate(data, 1):
            end = onset + duration
            print(f"  Event {i}: start={onset:.3f}s, dur={duration:.3f}s, amp={amp:.3f}")

# Stimulus 1 event times
starts = [
    119.046144, 162.398208, 205.94688, 249.495552, 292.847616,
    336.19968, 453.18144, 496.63180800000004,
    540.377088, 583.92576, 627.376128,
    744.259584, 788.004864, 831.65184, 875.003904,
    918.4542720000001
]

ends = [
    129.046144, 172.398208, 215.94688, 259.495552, 302.847616,
    419.731072, 463.18144, 506.63180800000004,
    550.377088, 593.92576, 710.90752,
    754.259584, 798.004864, 841.65184, 885.003904,
    928.4542720000001
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
        plt.axvspan(onset, onset + duration, color='tomato', alpha=0.5, label='activation')

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
plt.title('Activation and Shortened Control Events')
plt.yticks([])
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ''' UNDER THIS IS FIRST ITERATION OF THE SNIRF EDITOR SCRIPT'''
# # 1. Rename the old “1” block → “activation”
# with h5py.File(file_path, 'r+') as f:
#     for key in f['nirs'].keys():
#         grp = f['nirs'][key]
#         if 'name' in grp:
#             label = grp['name'][()].decode()
#             if label == '1':
#                 grp['name'][()] = np.string_('activation')
#                 print(f"Renamed /nirs/{key}’s 'name' from '1' → 'activation'.")

# # 2. Delete any existing group whose name is “control”
# with h5py.File(file_path, 'r+') as f:
#     for key in list(f['nirs'].keys()):
#         grp = f['nirs'][key]
#         if 'name' in grp and grp['name'][()].decode() == 'control':
#             del f['nirs'][key]
#             print(f"Deleted existing /nirs/{key} because name=='control'.")

# # (Recompute control_events here as you already have in your script.)
# starts = [
#     119.046144, 162.398208, 205.94688, 249.495552, 292.847616,
#     336.19968, 409.731072, 453.18144, 496.63180800000004,
#     540.377088, 583.92576, 627.376128, 700.90752,
#     744.259584, 788.004864, 831.65184, 875.003904,
#     918.4542720000001
# ]

# ends = [
#     129.046144, 172.398208, 215.94688, 259.495552, 302.847616,
#     346.19968, 419.731072, 463.18144, 506.63180800000004,
#     550.377088, 593.92576, 637.376128, 710.90752,
#     754.259584, 798.004864, 841.65184, 885.003904,
#     928.4542720000001
# ]

# control_events = []
# for i in range(len(ends) - 1):
#     control_start = ends[i]
#     control_end   = starts[i+1]
#     duration      = control_end - control_start
#     control_events.append([control_start, duration, 1.0])

# control_events = np.array(control_events)

# # 3. Now delete whatever group is still named “1” (if it’s left over)
# with h5py.File(file_path, 'r+') as f:
#     to_delete = []
#     for key in f['nirs'].keys():
#         grp = f['nirs'][key]
#         if 'name' in grp and grp['name'][()].decode() == '1':
#             to_delete.append(key)

#     for key in to_delete:
#         del f['nirs'][key]
#         print(f"Deleted /nirs/{key} because name=='1' (old activation).")

#     # 4. Create the new “stim_gap” → “control” block
#     if 'stim_gap' in f['nirs']:
#         del f['nirs']['stim_gap']        # remove any stale copy from prior runs
#         print("Removed old /nirs/stim_gap so we can recreate it cleanly.")

#     stim_gap = f['nirs'].create_group('stim_gap')
#     stim_gap.create_dataset('name', data=np.string_('control'))
#     stim_gap.create_dataset('data', data=control_events)
#     print(f"Created /nirs/stim_gap (name='control') with {len(control_events)} events.")

# # Activation events (onset times and durations)
# activation_onsets = [
#     119.046144, 162.398208, 205.94688, 249.495552, 292.847616,
#     336.19968, 409.731072, 453.18144, 496.63180800000004,
#     540.377088, 583.92576, 627.376128, 700.90752,
#     744.259584, 788.004864, 831.65184, 875.003904,
#     918.4542720000001
# ]
# activation_durations = [10.0] * len(activation_onsets)

# # Gap (control) events (onset times and durations)
# gap_onsets = [
#     129.046144, 172.398208, 215.94688, 259.495552, 302.847616,
#     419.731072, 463.18144, 506.63180800000004,
#     550.377088, 593.92576, 710.90752,
#     754.259584, 798.004864, 841.65184, 885.003904
# ]
# gap_durations = [
#     33.35206400000001, 33.54867199999998, 33.54867200000001,
#     33.35206400000004, 33.352063999999984, 33.450368000000026,
#     33.450368000000026, 33.74527999999992, 33.54867200000001,
#     33.450368000000026, 33.35206400000004, 33.74527999999998,
#     33.646975999999995, 33.35206400000004, 33.450368000000026
# ]

# plt.figure(figsize=(12, 2))

# # Plot activation events in orange
# for onset, duration in zip(activation_onsets, activation_durations):
#     plt.axvspan(onset, onset + duration, color='tomato', alpha=0.5, label='activation')

# # Plot gap events in light blue
# for onset, duration in zip(gap_onsets, gap_durations):
#     plt.axvspan(onset, onset + duration, color='lightblue', alpha=0.5, label='control')

# # Prevent duplicate legend entries
# handles, labels = plt.gca().get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# plt.legend(unique.values(), unique.keys())

# plt.xlabel('Time (s)')
# plt.title('Activation and Control Events')
# plt.yticks([])
# plt.grid(True, axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()