"""
VS-Code run-button script for the new pipeline.
"""
import sys
import torch, numpy as np
from pathlib import Path
import mne
from torch.utils.data import DataLoader
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from SWAE_preprocessing import preprocess_epochs
from SWAE_model import EpochDataset, Encoder1D, Decoder1D, \
                      GMMPrior, sliced_wasserstein
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data   

# ───────────────────────────────────  HYPER-PARAMS  ───────────────────────────────────
SUBJECT_INDEX        = 0
USE_LEFT_HAND        = False
CONTROL_SANITY_CHECK = True          # Control_A vs Control_B if True
N_AUG                = 40              # augmented copies per epoch
NOISE_FACTOR         = 100.0
BATCH_SIZE           = 16
EPOCHS               = 500
Z_DIM                = 8
LR                  = 1e-3
LAMBDA_SWD           = 25
WARMUP_EPOCHS       = 50
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0); np.random.seed(0)
print("device:", DEVICE)

# ───────────────────────────────────  LOAD EPOCHS  ───────────────────────────────────
full = get_raw_subject_data(SUBJECT_INDEX)

sfreq = full.info['sfreq']
if CONTROL_SANITY_CHECK:
    # Load 60-second epochs (as mne.Epochs or numpy arrays)
    full_epochs = get_raw_subject_data(SUBJECT_INDEX, tmin=-30, tmax=30)
    control_epochs = full_epochs["Control"]
    new_control_epochs = []  # will contain two epochs per original epoch
    for epoch in control_epochs:
        # If epoch is a NumPy array of shape (n_channels, n_times) then slice manually:
        # Calculate sample indices (assuming time axis corresponds to the second dimension)
        # For first 20 seconds: from -30 to -10 seconds
        epoch_first = epoch[:, :int(20 * sfreq)]
        # For last 20 seconds: from 10 to 30 seconds (i.e. last 20 seconds)
        epoch_last  = epoch[:, int(40 * sfreq): int(60 * sfreq)]
        new_control_epochs.extend([epoch_first, epoch_last])
    
    # Randomly split the doubled epochs into two groups
    idx = np.random.permutation(len(new_control_epochs))
    split = len(idx) // 2
    control_A_list = [new_control_epochs[i] for i in idx[:split]]
    control_B_list = [new_control_epochs[i] for i in idx[split:]]
    # If needed, convert lists of arrays back to mne.Epochs using mne.EpochsArray
    # Here we assume the info is available from full_epochs["Control"]
    info = full_epochs["Control"].info.copy()
    control_A = mne.EpochsArray(np.stack(control_A_list), info, tmin=-30)
    control_B = mne.EpochsArray(np.stack(control_B_list), info, tmin=-30)
    epochs_dict = {"Control_A": control_A, "Control_B": control_B}
    class_labels = ["Control_A", "Control_B"]
else:
    tap = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
    epochs_dict = {tap: full[tap], "Control": full["Control"]}


# if CONTROL_SANITY_CHECK:
#     ctl = full["Control"]
#     idx = np.random.permutation(len(ctl))
#     A,B = np.array_split(idx, 2)
#     epochs_dict = {"Control_A": ctl[A], "Control_B": ctl[B]}
# else:
#     tap = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
#     epochs_dict = {tap: full[tap], "Control": full["Control"]}

labels = list(epochs_dict)
print({l: len(epochs_dict[l]) for l in labels})

# ───────────────────────────────────  PRE-PROCESS  ───────────────────────────────────
arrays = preprocess_epochs(epochs_dict, n_aug=N_AUG, rng_seed=0)
for k,v in arrays.items(): print(f"{k}: {len(v)} samples")

all_samples = arrays[labels[0]] + arrays[labels[1]]
ds = EpochDataset(all_samples); dl = DataLoader(ds,BATCH_SIZE,shuffle=True,drop_last=True)
print(f"Input shape  C={ds.C}, T={ds.T}, total {len(ds)} samples")

# ───────────────────────────────────  MODEL  ───────────────────────────────────
enc = Encoder1D(ds.C, Z_DIM).to(DEVICE)
dec = Decoder1D(ds.C, ds.T, Z_DIM).to(DEVICE)
pri = GMMPrior(Z_DIM).to(DEVICE)

opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+list(pri.parameters()), lr=LR)
mse = torch.nn.MSELoss()

# ───────────────────────────────────  TRAIN  ───────────────────────────────────
print("\n=== Training ===")
for ep in range(1,EPOCHS+1):

    # Linearly increase lambda during the warm-up period
    if ep < 10:
        current_lambda = 0
    elif ep < WARMUP_EPOCHS:
        current_lambda = LAMBDA_SWD * ((ep-10) / (WARMUP_EPOCHS-10))
    else:
        current_lambda = LAMBDA_SWD
    rec_tot,swd_tot=0.,0.
    for xb in dl:
        xb = xb.to(DEVICE)
        mu, logv = enc(xb)
        z = mu + NOISE_FACTOR * torch.exp(0.5 * logv) * torch.randn_like(mu)
        xh = dec(z)
        loss_rec = mse(xh, xb)
        loss_swd = sliced_wasserstein(z, pri.sample(len(z)), n_proj=400)
        loss = loss_rec + current_lambda*loss_swd
        opt.zero_grad(); loss.backward(); opt.step()
        rec_tot += loss_rec.item(); swd_tot += loss_swd.item()
    if ep%20==0 or ep==1:
        print(f"Ep {ep:3d}: recon {rec_tot/len(dl):.4e}  swd {swd_tot/len(dl):.4e}")

# ───────────────────────────────────  CLUSTER TEST  ───────────────────────────────────
def embed(arrs):
    enc.eval(); out=[]
    with torch.no_grad():
        for a in DataLoader(EpochDataset(arrs),batch_size=256):
            out.append(enc(a.to(DEVICE))[0].cpu())
    return torch.cat(out)

lat_A = embed(arrays[labels[0]])
lat_B = embed(arrays[labels[1]])

def assign(z):
    device = z.device
    mu  = pri.mu.detach().to(device).squeeze()
    var = torch.exp(pri.logvar.detach().to(device).squeeze())
    d0 = ((z - mu[0])**2 / var[0]).sum(1)
    d1 = ((z - mu[1])**2 / var[1]).sum(1)
    return (d1 < d0).long()

cont = np.zeros((2,2),int)
cont[0] += np.bincount(assign(lat_A), minlength=2)
cont[1] += np.bincount(assign(lat_B), minlength=2)
p_val = fisher_exact(cont)[1]
print("\nContingency\n", cont, "\nFisher p =", p_val)

# ───────────────────────────────────  PLOT  ───────────────────────────────────
# pcs = PCA(2).fit_transform(torch.cat([lat_A,lat_B]).numpy())
# lbls = np.array([labels[0]]*len(lat_A) + [labels[1]]*len(lat_B))
# cols = {labels[0]:"tab:blue", labels[1]:"tab:orange"}
# plt.scatter(pcs[:,0], pcs[:,1], c=[cols[l] for l in lbls], alpha=.7, edgecolor='k', linewidth=.3)
# for l in labels: plt.scatter([],[],c=cols[l],label=l)
# plt.title(f"Latent PCs)"); plt.legend(); plt.show()

data = torch.cat([lat_A, lat_B]).numpy()
pca = PCA(2)
pcs = pca.fit_transform(data)
# Transform the prior means to the same PCA space
pcs_pri = pca.transform(pri.mu.cpu().detach().numpy())

lbls = np.array([labels[0]]*len(lat_A) + [labels[1]]*len(lat_B))
cols = {labels[0]:"tab:blue", labels[1]:"tab:orange"}

plt.figure(figsize=(8,6))
plt.scatter(pcs[:,0], pcs[:,1], c=[cols[l] for l in lbls], alpha=.7, edgecolor='k', linewidth=.3)
for l in labels: 
    plt.scatter([], [], c=cols[l], label=l)
# Plot the two prior means
plt.scatter(pcs_pri[:,0], pcs_pri[:,1], c='red', marker='X', s=200, label='Prior Means')
plt.title("Latent PCs and Prior Means")
plt.legend()
plt.show()