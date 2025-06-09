import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)


from Classifier.AE_models import ReconstructionAutoencoder, ClassificationAutoencoder, MixtureVAE, train_reconstruction_autoencoder, train_classification_autoencoder, train_mixture_vae, create_sliding_windows
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import your preprocessing pipeline (make sure Preprocessing.py is in your PYTHONPATH)
from Preprocessing import get_group_epochs_subtracting_short, get_continuous_subject_data
import random
import itertools
import pandas as pd
from tqdm import tqdm

seed_ = 42
np.random.seed(seed_)
random.seed(seed_)
torch.manual_seed(seed_)
torch.cuda.manual_seed(seed_)
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_AE(
    device,
    test_participant,
    *,
    mode="classification",         # <-- new: either "classification" or "reconstruction"
    latent_dim=2,
    num_classes=3,
    epochs_num=25,
    verbose=True,
    transform_data=True,
    noise_std=0.01,
    subtract_max=0.01,
    scale_max=0.01,
    shift_max=0,
    data=None,
    labels=None,
):
    if data is None or labels is None:
        # -------------------------
        # Data Preparation
        # -------------------------
        # Load preprocessed epochs
        epochs = get_continuous_subject_data(subject=test_participant)


        # Extract data in a given time window (e.g., 0 to 10 seconds)
        # MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
        data = []
        for i in range(len(epochs)):
            data.append(epochs[i].copy().crop(tmin=0, tmax=14).get_data())
        # For PyTorch, we want the input shape to be (batch, channels, time)

        for X in data:
            n_epochs, n_channels, n_times = X.shape

            for ch in range(n_channels):
                channel_mean = np.mean(X[:, ch, :])
                channel_std = np.std(X[:, ch, :])
                X[:, ch, :] = (X[:, ch, :]-channel_mean) / channel_std


        # Extract labels (assumed to be in epochs.events[:, -1])
        # Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
        # To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
        labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]  # subtract one to have classes 0,1,2


    # Split into train and test sets
    X_train, X_test, y_train, y_test =  data[:test_participant] + data[test_participant+1:], data[test_participant], labels[:test_participant] + labels[test_participant+1:], labels[test_participant]

    X_train_np, y_train_np = create_sliding_windows(X_train, y_train, 32)
    X_test_np, y_test_np = create_sliding_windows([X_test], [y_test], 32)
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_train = torch.from_numpy(y_train_np)
    y_test = torch.from_numpy(y_test_np)
    # Create TensorDatasets and DataLoaders
    batch_size = 16
    train_dataset_recon = TensorDataset(X_train)  # for reconstruction, input=target
    val_dataset_recon = TensorDataset(X_test)
    train_loader_recon = DataLoader(train_dataset_recon, batch_size=batch_size, shuffle=True)
    val_loader_recon = DataLoader(val_dataset_recon, batch_size=batch_size, shuffle=False)


    train_dataset_class = TensorDataset(X_train, y_train)
    val_dataset_class = TensorDataset(X_test, y_test)
    train_loader_class = DataLoader(train_dataset_class, batch_size=batch_size, shuffle=True)
    val_loader_class = DataLoader(val_dataset_class, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Hyperparameters
    # -------------------------

    # After you have train_loader_recon, val_loader_recon, train_loader_class, val_loader_class:

    input_channels = X_train.shape[1]
    input_length   = X_train.shape[2]

    # ─────────────── classification branch ───────────────
    if mode == "classification":
        model = ClassificationAutoencoder(input_channels, input_length,
                                        latent_dim, num_classes)
        hist = train_classification_autoencoder(
            model, train_loader_class, val_loader_class,
            epochs_num, device,
            verbose        = verbose,
            transform_data = transform_data,
            noise_std      = noise_std,
            subtract_max   = subtract_max,
            scale_max      = scale_max,
            shift_max      = shift_max,
        )

        model.eval()
        with torch.no_grad():
            latent_test  = model.get_latent(X_test.to(device)).cpu().numpy()
            latent_train = model.get_latent(X_train.to(device)).cpu().numpy()

        # accuracy on the validation set
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader_class:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        acc = correct / total

        return {
            "mode":           "classification",
            "latent_test":    latent_test,
            "y_test":         y_test_np,
            "latent_train":   latent_train,
            "y_train":        y_train_np,
            "accuracy":       acc,
            "history":        hist,
            "recon_test":     None,   # keeps keys aligned
            "recon_train":    None,
        }

    # ─────────────── reconstruction branch ───────────────
    elif mode == "reconstruction":
        model = ReconstructionAutoencoder(input_channels, input_length, latent_dim)
        hist  = train_reconstruction_autoencoder(
            model, train_loader_recon, val_loader_recon,
            epochs_num, device,
            verbose = verbose
        )

        model.eval()
        with torch.no_grad():
            # encodings  (== want the same as classification)
            latent_test  = model.get_latent(X_test.to(device)).cpu().numpy()
            latent_train = model.get_latent(X_train.to(device)).cpu().numpy()
            # full reconstructions (extra info)
            recon_test   = model(X_test.to(device)).cpu().numpy()
            recon_train  = model(X_train.to(device)).cpu().numpy()

        # average reconstruction error on the test set
        mse_test = np.mean((recon_test - X_test_np) ** 2)

        return {
            "mode":           "reconstruction",
            "latent_test":    latent_test,
            "y_test":         y_test_np,
            "latent_train":   latent_train,
            "y_train":        y_train_np,
            "accuracy":       mse_test,      # same field name, different meaning
            "history":        hist,
            "recon_test":     recon_test,    # only present for reconstruction
            "recon_train":    recon_train,
        }
    
    
    else:
        raise ValueError("mode must be 'classification' or 'reconstruction'")

def run_mixture_vae_single_subject(
        device,
        participant_idx,
        *,
        latent_dim   = 2,
        epochs_num   = 25,
        beta         = 5.0,
        means        = 1,
        logvar       = 0,
        verbose      = True):

    # ---------- load & prepare ONE subject ----------
    # -------------------------
    # Data Preparation
    # -------------------------
    # Load preprocessed epochs
    epochs = get_group_epochs(tmin=-5, add_hbr=False, hbr_multiplier=5.0, hbr_shift=1.0)
    labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]  # subtract one to have classes 0,1,2
    labels = labels[participant_idx]
    X = epochs[participant_idx]
    # Extract data in a given time window (e.g., 0 to 10 seconds)
    # MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
    
    X = X.copy().crop(tmin=-4.5, tmax=14).get_data()
    # For PyTorch, we want the input shape to be (batch, channels, time)

    n_epochs, n_channels, n_times = X.shape

    for ch in range(n_channels):
        channel_mean = np.mean(X[:, ch, :])
        channel_std = np.std(X[:, ch, :])
        X[:, ch, :] = (X[:, ch, :]-channel_mean) / channel_std


    # Extract labels (assumed to be in epochs.events[:, -1])
    # Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
    # To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
    
    n_ep, n_ch, n_t = X.shape

    # z‑score per channel
    for ch in range(n_ch):
        m, s = X[:, ch, :].mean(), X[:, ch, :].std()
        X[:, ch, :] = (X[:, ch, :] - m) / s

    # sliding‑window → (N,C,Twin)
    X, labels = create_sliding_windows([X], [labels], 32)
    X            = torch.from_numpy(X)                       # float32
    labels       = [1 if x >= 1 else 0 for x in labels]

    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

    # ---------- model ----------
    model = MixtureVAE(n_ch, X.shape[2], latent_dim).to(device)

    # prior: two Gaussians near zero
    prior_means   = torch.ones(size=(2, latent_dim), device=device)/np.sqrt(latent_dim) * means
    prior_means[0]*= -1
    prior_logvars = torch.ones_like(prior_means)/np.sqrt(latent_dim) * logvar  
    pi_mix        = torch.tensor([0.5, 0.5], device=device)

    # ---------- train ----------
    hist = train_mixture_vae(model, loader, loader,          # same data for “val”
                             prior_means, prior_logvars, pi_mix,
                             epochs_num, device,
                             beta=beta, verbose=verbose)

    # ---------- latent + clustering ----------
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X.to(device))                  # [N,2]
        latent = mu.cpu().numpy()

    # classify by nearest prior mean
    d0 = ((latent - prior_means[0].cpu().numpy())**2).sum(1)
    d1 = ((latent - prior_means[1].cpu().numpy())**2).sum(1)
    pred = np.where(d0 < d1, 0, 1)                          # 0/1 cluster label

    acc = np.mean(np.abs(pred-np.array(labels)))
    if acc < 0.5: acc = 1-acc
    return {
        "latent":       latent,
        "pred_labels":  pred,
        "labels":       labels,
        "prior_means":  prior_means.cpu().numpy(),
        "history":      hist,
        "acc" :         acc,
    }


# ───────────────────────── 2. grid_search_AE  ────────────────────────────────
def grid_search_AE(device,
                   *,
                   latent_dim        = 2,
                   num_classes       = 3,
                   epochs_num        = 25,
                   seeds             = (42, 43, 44),
                   noise_std_list    = (0.001, 0.005, 0.01),
                   subtract_max_list = (0.001, 0.005, 0.01),
                   scale_max_list    = (0.005, 0.01),
                   shift_max_list    = (0, 1),
                   summary_csv       = "grid_search_summary.csv",
                   epoch_csv         = "grid_search_epoch_log.csv"):
    # how many participants?
    n_participants = len(get_group_epochs(tmin=-5,
                                          add_hbr=False,
                                          hbr_multiplier=5.0,
                                          hbr_shift=1.0))

    combos = list(itertools.product(noise_std_list,
                                    subtract_max_list,
                                    scale_max_list,
                                    shift_max_list))
    # -------------------------
    # Data Preparation
    # -------------------------
    # Load preprocessed epochs
    epochs = get_group_epochs(tmin=-5, add_hbr=False, hbr_multiplier=5.0, hbr_shift=1.0)


    # Extract data in a given time window (e.g., 0 to 10 seconds)
    # MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
    data = []
    for i in range(len(epochs)):
        data.append(epochs[i].copy().crop(tmin=0, tmax=14).get_data())
    # For PyTorch, we want the input shape to be (batch, channels, time)

    for X in data:
        n_epochs, n_channels, n_times = X.shape

        for ch in range(n_channels):
            channel_mean = np.mean(X[:, ch, :])
            channel_std = np.std(X[:, ch, :])
            X[:, ch, :] = (X[:, ch, :]-channel_mean) / channel_std


    # Extract labels (assumed to be in epochs.events[:, -1])
    # Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
    # To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
    labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]  # subtract one to have classes 0,1,2
    summary_rows = []
    epoch_rows   = []
    best_acc     = -float("inf")
    best_params  = None

    total_runs = len(combos) * len(seeds) * n_participants
    pbar = tqdm(total=total_runs, desc="grid-search")
    for ns, sub, sc, sh in combos:
        acc_sum = 0.0
        runs    = 0

        for seed in seeds:
            for participant in range(n_participants):
                # unpack run_AE
                pbar.update(1)
                _, _, _, _, acc, hist = run_AE(
                    device,
                    participant,
                    latent_dim     = latent_dim,
                    num_classes    = num_classes,
                    epochs_num     = epochs_num,
                    verbose        = False,
                    transform_data = True,
                    noise_std      = ns,
                    subtract_max   = sub,
                    scale_max      = sc,
                    shift_max      = sh,
                    data=data,
                    labels=labels
                )

                # accumulate for summary
                acc_sum += acc
                runs    += 1

                # log every epoch too
                for ep, (tr_l, tr_a, v_l, v_a) in enumerate(zip(
                        hist['train_loss'], hist['train_acc'],
                        hist['val_loss'],   hist['val_acc']), start=1):
                    epoch_rows.append({
                        'noise_std':    ns,
                        'subtract_max': sub,
                        'scale_max':    sc,
                        'shift_max':    sh,
                        'seed':         seed,
                        'participant':  participant,
                        'epoch':        ep,
                        'train_loss':   tr_l,
                        'train_acc':    tr_a,
                        'val_loss':     v_l,
                        'val_acc':      v_a
                    })

                

        mean_acc = acc_sum / runs

        summary_rows.append({
            'noise_std':     ns,
            'subtract_max':  sub,
            'scale_max':     sc,
            'shift_max':     sh,
            'mean_test_acc': mean_acc
        })

        # **pick best by accuracy**
        if mean_acc > best_acc:
            best_acc    = mean_acc
            best_params = {
                'noise_std':    ns,
                'subtract_max': sub,
                'scale_max':    sc,
                'shift_max':    sh
            }

    pbar.close()

    # write out CSVs
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(epoch_rows  ).to_csv(epoch_csv,   index=False)
    print(f"✓ summary → {summary_csv}")
    print(f"✓ epochs  → {epoch_csv}")

    # return best_params *and* best_acc so you can inspect it
    return best_params, best_acc

"""
best_params, best_score = grid_search_AE(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    latent_dim        = 2,
    num_classes       = 3,
    epochs_num        = 20,
    seeds             = [42],
    noise_std_list    = (0.001, 0.01),
    subtract_max_list = (0.001, 0.01),
    scale_max_list    = (0.001, 0.01),
    shift_max_list    = (0, 1),
    summary_csv       = "Classifier/grid_summary.csv",
    epoch_csv         = "Classifier/grid_epoch_log.csv"
)

print("Best parameters:", best_params)
print("Best mean test accuracy:", best_score)
quit()
"""


results = run_AE(device=device, 
                 test_participant=4, 
                 mode="reconstruction", 
                 epochs_num=25, 
                 verbose=True, 
                 latent_dim=2, 
                 transform_data=True)


# unpack
mode         = results["mode"]
latent_test  = results["latent_test"]
y_test       = results["y_test"]
latent_train = results["latent_train"]
y_train      = results["y_train"]
score        = results["accuracy"]      # accuracy or MSE depending on mode

# common legend setup
legend_labels = {0: "Control", 1: "Tapping/Left", 2: "Tapping/Right"}
handles = [
    mpatches.Patch(color=plt.cm.viridis(i/2.0), label=legend_labels[i])
    for i in sorted(legend_labels)
]


# test set scatter
plt.figure(figsize=(7, 5))
plt.scatter(latent_test[:, 0], latent_test[:, 1],
            c=y_test, cmap='viridis', s=15, alpha=0.2)
plt.title(f'{mode.capitalize()} AE test  (score: {score:.2f})')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()

# train set scatter
plt.figure(figsize=(7, 5))
plt.scatter(latent_train[:, 0], latent_train[:, 1],
            c=y_train, cmap='viridis', s=8, alpha=0.2)
plt.title(f'{mode.capitalize()} AE train')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()