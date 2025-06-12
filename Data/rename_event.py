import h5py

file_path = 'Data/2_hand.snirf'

with h5py.File(file_path, 'r+') as f:
    for key in f['nirs']:
        group = f['nirs'][key]
        if 'name' in group:
            label = group['name'][()].decode()
            if label == 'activation':
                group['name'][...] = b'1'
                print(f"✅ Renamed stimulus group '{key}' from 'activation' to '1'.")


print("\nCurrent stimulus labels:")
for key in f['nirs']:
    name = f['nirs'][key]['name'][()].decode()
    print(f"{key}: {name}")