import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt

"""

# Copying a .snirf document

"""

shutil.copy('Data/2_hand.snirf', 'Data/2_dummy_hand.snirf')



file_path = 'Data/2_dummy_hand.snirf'


# Print start and end times of the original events

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


# Original stimulus start and end times
starts = [
    119.046144, 162.398208, 205.94688, 249.495552, 292.847616, 336.19968,
    453.18144, 496.63180800000004, 540.377088, 583.92576,
    627.376128,  744.259584, 788.004864, 831.65184,
    875.003904, 918.4542720000001
]

ends = [
    129.046144, 172.398208, 215.94688, 259.495552, 302.847616,
    419.731072, 463.18144, 506.63180800000004, 550.377088, 593.92576,
    710.90752, 754.259584, 798.004864, 841.65184,
    885.003904, 928.4542720000001
]

# Generate gap events
gap_events = []
for i in range(len(ends) - 1):
    gap_start = ends[i]
    gap_end = starts[i+1]
    duration = gap_end - gap_start
    gap_events.append([gap_start, duration, 1.0])  # amplitude = 1.0

gap_events = np.array(gap_events)

# Edit the .snirf file
with h5py.File(file_path, 'r+') as f:
    # Delete stimulus 1
    del f['nirs']['stim1']

    # Create a new stimulus group for the gaps
    stim_gap = f['nirs'].create_group('stim_gap')
    stim_gap.create_dataset('name', data=np.string_('gap_period'))
    stim_gap.create_dataset('data', data=gap_events)

print("Stimulus 1 deleted and gap events created as 'stim_gap'.")

# Load the data
with h5py.File(file_path, 'r') as f:
    stim_data = f['nirs/stim_gap/data'][()]
    onsets = stim_data[:, 0]
    durations = stim_data[:, 1]
    ends = onsets + durations

# Plot events
plt.figure(figsize=(10, 2))
for onset, end in zip(onsets, ends):
    plt.axvspan(onset, end, color='lightblue', alpha=0.5)

plt.xlabel('Time (s)')
plt.title('Gap Events from stim_gap')
plt.yticks([])  # optional: hide y-axis
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()