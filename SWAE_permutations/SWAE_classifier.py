"""
WAE training runner
-------------------
Usage (default hyper-params):
    python train_wae.py
or custom:
    python train_wae.py --subject 0 --n_perm 4 --epochs 300 --lr 5e-4
"""
import sys
import argparse, torch, numpy as np
from pathlib import Path
from WAE_model import *
from WAE_preprocessing import *

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
#  HYPER-PARAMETERS  ---------------------------------------------------------
# ---------------------------------------------------------------------------
SUBJECT_INDEX      = 0          # which person in the group dataset
USE_LEFT_HAND      = True       # True → Tapping_Left + Control
                                # False → Tapping_Right + Control
N_PERM             = 2          # epochs per permutation cube
MAX_PERM           = 5_000       # cap number of cubes per label (None = all)
BATCH_SIZE         = 32
EPOCHS             = 150
Z_DIM              = 8
LEARNING_RATE      = 1e-3
LAMBDA_SWD         = 25.0
N_PROJ             = 100
DEVICE             = ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
#  1. Load & slice epochs ----------------------------------------------------
# ---------------------------------------------------------------------------
print("→ loading subject", SUBJECT_INDEX)
full_epochs = get_raw_subject_data(subject=SUBJECT_INDEX)          # MNE.Epochs

label_tap = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
label_ctl = "Control"

# Turn them into the dict format expected by `_channel_order`
epochs_dict = {
    label_tap: full_epochs[label_tap],      # MNE.Epochs objects
    label_ctl: full_epochs[label_ctl],
}

# ---------------------------------------------------------------------------
#  2. Build permutation cubes -----------------------------------------------
# ---------------------------------------------------------------------------
cubes_per_label = preprocess_for_wae_cube(
    epochs_dict,
    class_labels=[label_tap, label_ctl],
    n_perm=N_PERM,
    max_perm=MAX_PERM,
    rng=0,
)

# Merge the two classes into one list for unsupervised training
cubes_all = cubes_per_label[label_tap] + cubes_per_label[label_ctl]
print(f"   {len(cubes_per_label[label_tap])} cubes from {label_tap}")
print(f"   {len(cubes_per_label[label_ctl])} cubes from {label_ctl}")

# ---------------------------------------------------------------------------
#  3. Dataset / DataLoader ---------------------------------------------------
# ---------------------------------------------------------------------------
ds = CubeDataset(cubes_all, zscore=True)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)

print(f"Dataset cube shape: (n_perm={ds.D}, C={ds.C}, T={ds.T})")

# ---------------------------------------------------------------------------
#  4. Build model ------------------------------------------------------------
# ---------------------------------------------------------------------------
enc = CubeEncoder(ds.D, ds.C, z_dim=Z_DIM).to(DEVICE)
dec = CubeDecoder(ds.D, ds.C, ds.T, z_dim=Z_DIM).to(DEVICE)
pri = GMMPrior(Z_DIM).to(DEVICE)

opt = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()) + list(pri.parameters()),
    lr=LEARNING_RATE
)
mse = torch.nn.MSELoss()

# ---------------------------------------------------------------------------
#  5. Train loop with live loss print ----------------------------------------
# ---------------------------------------------------------------------------
print("\n===== Training SWAE =====")
loss_hist = {"recon": [], "swd": []}

for ep in range(1, EPOCHS + 1):
    recon_acc, swd_acc = 0.0, 0.0
    for x in dl:
        x = x.to(DEVICE)                           # (B, 1, D, C, T)

        mu, logv = enc(x)
        z   = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
        x_hat   = dec(z)

        loss_rec = mse(x_hat, x)
        loss_swd = sliced_wasserstein(
            z,
            pri.sample(x.size(0)),
            n_proj=N_PROJ
        )
        loss = loss_rec + LAMBDA_SWD * loss_swd

        opt.zero_grad()
        loss.backward()
        opt.step()

        recon_acc += loss_rec.item()
        swd_acc   += loss_swd.item()

    loss_hist["recon"].append(recon_acc / len(dl))
    loss_hist["swd"].append(swd_acc / len(dl))
    print(f"Epoch {ep:3d} | recon {loss_hist['recon'][-1]:.5f} "
          f"| swd {loss_hist['swd'][-1]:.5f}")

# ---------------------------------------------------------------------------
#  6. Save checkpoint --------------------------------------------------------
# ---------------------------------------------------------------------------
ckpt_dir  = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / (
    f"swae_sub{SUBJECT_INDEX}_{'left' if USE_LEFT_HAND else 'right'}.pt"
)

torch.save({
    "encoder": enc.state_dict(),
    "decoder": dec.state_dict(),
    "prior":   pri.state_dict(),
    "params":  {
        "subject": SUBJECT_INDEX,
        "hand":    "left" if USE_LEFT_HAND else "right",
        "n_perm":  N_PERM,
        "epochs":  EPOCHS,
        "λ":       LAMBDA_SWD,
        "z_dim":   Z_DIM,
    },
    "loss": loss_hist,
}, ckpt_path)

print("\n Training complete — checkpoint saved to", ckpt_path)
