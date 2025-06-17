"""
Train SWAE with sliding-window sparse matrices.
Produces p-value & PC scatter.
"""
import sys, os
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact, ranksums
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mne
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

PLOT = False

# ─── hyper-params ───────────────────────────────────────────────────────────
SUBJECTS          = (1,4)
USE_LEFT_HAND     = True 
CONTROL_SPLIT     = True
CONTROL_FRAC      = 0.0   # new: fraction of tapping epochs to replace with control epochs
SEEDS             = (4,)
# SEEDS           = range(10)

WINDOW_SIZE      = 100
WINDOW_STEP      = 20
BATCH_SIZE       = 64
EPOCHS           = 140
Z_DIM            = 11
LR               = 5e-4
LAMBDA_SWD       = 0.7
WARMUP_EPOCHS    = 50
FIRST_EPOCHS     = 50
N_PROJ           = 465
PRIOR_MEAN       = 1.0
PRIOR_LOGVAR     = 0.0
LEARNABLE_PRIOR  = True

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE          = False
# ────────────────────────────────────────────────────────────────────────────

from SWAE_preprocessing      import preprocess_for_wae_windows
from SWAE_model              import (WindowDataset, build_dataloader,
                                    Encoder2D, Decoder2D,
                                    GMMPrior, sliced_wasserstein)

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data   

# If CONTROL_FRAC is used, import the replacement function.
if CONTROL_FRAC > 0:
    from PA_classifier.PA_tests.pa_classification_run_no_overlap import replace_fraction_with_control_no_overlap_both

# Helper function to save latent space plot at intermediate epochs
def save_latent_space_plot(enc, pri, labels, mats, subject, epoch):
    # Switch to evaluation mode
    enc.eval()
    lat, lab_vals = [], []
    # Build latent representation for each label
    for lbl in labels:
        dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
        with torch.no_grad():
            for xb in dl_lbl:
                out = enc(xb.to(DEVICE))[0].cpu()
                lat.append(out)
                lab_vals.extend([lbl] * xb.size(0))
    lat = torch.cat(lat).numpy()
    pca = PCA(2).fit(lat)
    pc  = pca.transform(lat)
    pc_prior = pca.transform(pri.mu.detach().cpu().numpy())  # shape (n_modes, 2)

    pal = dict(zip(labels, ["tab:blue", "tab:orange"]))
    plt.figure(figsize=(6,5))
    for lbl in labels:
        mask = np.array(lab_vals) == lbl
        plt.scatter(pc[mask,0], pc[mask,1], c=pal[lbl], label=lbl, s=8, alpha=0.7)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(lat)
    # centers = pca.transform(kmeans.cluster_centers_)
    # plt.scatter(centers[:, 0], centers[:, 1], c="green", marker="X", s=150, label="KMeans Centers")
    plt.scatter(pc_prior[:, 0], pc_prior[:, 1], c="red", marker="X", s=100, label="Prior Means")
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
    # plt.title(f"Subject {subject} - Epoch {epoch}")
    plt.title(f"Latent Space, Subject 4 - Control Test")
    plt.tight_layout()
    # Create folder based on subject name if it does not exist
    folder = f"subject{subject}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/latent_epoch_{epoch}.png")
    plt.close()
    # Return to training mode
    enc.train()

def centroid_mahalanobis(X, y):
    X1, X2 = X[y==0], X[y==1]
    S  = np.cov(np.vstack([X1, X2]).T)      # pooled covariance
    diff = X1.mean(0) - X2.mean(0)
    D2   = diff @ np.linalg.inv(S) @ diff   # squared distance
    return np.sqrt(D2)

def silhouette_null(lat, n_iter=500, rng=None):
    rng = np.random.default_rng(rng)
    from sklearn.metrics import silhouette_score
    null_scores = []
    for _ in range(n_iter):
        null_lbl = rng.permutation(label_int)
        null_scores.append(silhouette_score(lat, null_lbl))
    return np.percentile(null_scores, [95])   # 95th-percentile threshold

# torch.manual_seed(0); np.random.seed(0)
print("device:", DEVICE) if VERBOSE else None
for seed in SEEDS:
    for subj in SUBJECTS:

        print(f"\n Processing subject {subj}… \n")
        # 1 ── load epochs
        full_epochs = get_raw_subject_data(subj)
        tap_lbl = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
        ctl_lbl = "Control"

        # New: if CONTROL_FRAC > 0, replace some tapping epochs with control epochs.
        if CONTROL_FRAC > 0:
            tap_left_mixed, tap_right_mixed, new_control = replace_fraction_with_control_no_overlap_both(
                full_epochs["Tapping_Left"], full_epochs["Tapping_Right"], full_epochs["Control"], CONTROL_FRAC, seed=seed)
            if USE_LEFT_HAND:
                epochs_dict = {tap_lbl: tap_left_mixed, ctl_lbl: new_control}
            else:
                epochs_dict = {tap_lbl: tap_right_mixed, ctl_lbl: new_control}
            labels = [tap_lbl, ctl_lbl]
        elif CONTROL_SPLIT:
            ctl = full_epochs["Control"]
            idx = np.random.permutation(len(ctl))
            half = len(ctl) // 2
            epochs_dict = {"Control_A": ctl[idx[:half]], "Control_B": ctl[idx[half:]]}
            labels = ["Control_A", "Control_B"]
        else:
            epochs_dict = {tap_lbl: full_epochs[tap_lbl], ctl_lbl: full_epochs[ctl_lbl]}
            labels      = [tap_lbl, ctl_lbl]

        sfreq = full_epochs.info['sfreq']
        if CONTROL_SPLIT:
            full_epochs = get_raw_subject_data(subj, tmin=-30, tmax=30)
            control_epochs = full_epochs["Control"]
            new_control_epochs = []
            for epoch in control_epochs:
                epoch_first = epoch[:, :int(20 * sfreq)]
                epoch_last  = epoch[:, int(40 * sfreq): int(60 * sfreq)]
                new_control_epochs.extend([epoch_first, epoch_last])
            
            idx = np.random.permutation(len(new_control_epochs))
            split = len(idx) // 2
            control_A_list = [new_control_epochs[i] for i in idx[:split]]
            control_B_list = [new_control_epochs[i] for i in idx[split:]]
            info = full_epochs["Control"].info.copy()
            control_A = mne.EpochsArray(np.stack(control_A_list), info, tmin=-30)
            control_B = mne.EpochsArray(np.stack(control_B_list), info, tmin=-30)
            epochs_dict = {"Control_A": control_A, "Control_B": control_B}
            labels = ["Control_A", "Control_B"]

        # 2 ── window → matrices
        mats = preprocess_for_wae_windows(epochs_dict,
                                        class_labels=labels,
                                        window_size=WINDOW_SIZE,
                                        window_step=WINDOW_STEP)
        for k,v in mats.items():
            print(f"{k}: {len(v)} samples") if VERBOSE else None
        if BATCH_SIZE > len(mats[labels[0]]):
            BATCH_SIZE = len(mats[labels[0]])

        all_mats = mats[labels[0]] + mats[labels[1]]
        ds       = WindowDataset(all_mats)
        C, T     = ds.C, ds.T

        # 3 ── model + optimizer
        enc = Encoder2D(C, Z_DIM).to(DEVICE)
        dec = Decoder2D(C, T, Z_DIM).to(DEVICE)
        pri = GMMPrior(Z_DIM, mu=PRIOR_MEAN, logvar=PRIOR_LOGVAR, learnable=LEARNABLE_PRIOR).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+list(pri.parameters()), lr=LR)
        mse = torch.nn.MSELoss()
        print(f"batch size: {BATCH_SIZE}, C={C}, T={T}, Z_DIM={Z_DIM}") if VERBOSE else None
        dl = build_dataloader(all_mats, BATCH_SIZE, shuffle=True, drop_last=True)

        # 4 ── train
        print("training…") if VERBOSE else None
        for ep in range(1, EPOCHS+1):
            if ep < FIRST_EPOCHS:
                current_lambda = 0.05
            elif ep < WARMUP_EPOCHS:
                current_lambda = LAMBDA_SWD * ((ep - FIRST_EPOCHS) / (WARMUP_EPOCHS - FIRST_EPOCHS))
            else:
                current_lambda = LAMBDA_SWD
            rec, swd = 0., 0.
            for xb in dl:
                xb = xb.to(DEVICE)
                mu_enc, logv_enc = enc(xb)
                z  = mu_enc + torch.exp(0.5*logv_enc) * torch.randn_like(mu_enc)
                xb_hat = dec(z)
                l_rec  = mse(xb_hat, xb)
                l_swd  = sliced_wasserstein(z, pri.sample(xb.size(0)), n_proj=N_PROJ)
                loss   = l_rec + current_lambda*l_swd

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Clamp the learnable prior parameters to avoid collapsing
                pri.clamp_parameters(mu_min=-2.0, mu_max=2.0,
                                     logvar_min=-5.0, logvar_max=5.0)

                rec += l_rec.item()
                swd += l_swd.item()
            if (ep % 20 == 0 or ep == 1) and VERBOSE:
                print(f"epoch {ep:3d} | recon {rec/(len(dl)+1e-7):.4f} | swd {swd/(len(dl)+1e-6):.4f}")
            # Save latent space plot: every epoch for first 20, then every 10 epochs
            if PLOT and (ep <= 20 or ep % 10 == 0):
                save_latent_space_plot(enc, pri, labels, mats, subj, ep)

        # 5 ── contingency + p-value
        def assign(z):   # Mahalanobis closest mode
            diff = z.unsqueeze(1) - pri.mu
            d2 = (diff**2 / torch.exp(pri.logvar).unsqueeze(0)).sum(dim=2)
            return torch.argmin(d2, dim=1).cpu().numpy()

        table = np.zeros((2,2), int)
        enc.eval()
        for r,lbl in enumerate(labels):
            dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
            with torch.no_grad():
                for xb in dl_lbl:
                    cl = assign(enc(xb.to(DEVICE))[0])
                    table[r] += np.bincount(cl, minlength=2)

        p_val = fisher_exact(table)[1]
        print(f"SUBJECT: {subj}: P =", p_val)

        # 5b ── epoch-level contingency + p-value (one vote per epoch)
        wins_per_epoch = ((T - WINDOW_SIZE) // WINDOW_STEP) + 1  # windows in one epoch
        table_ep = np.zeros((2, 2), dtype=int)

        enc.eval()
        for r, lbl in enumerate(labels):
            dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
            all_cl = []
            with torch.no_grad():
                for xb in dl_lbl:
                    cl = assign(enc(xb.to(DEVICE))[0])
                    all_cl.extend(cl)
            all_cl = np.asarray(all_cl)
            n_epochs = len(epochs_dict[lbl])
            all_cl = all_cl.reshape(n_epochs, wins_per_epoch)
            epoch_vote = (all_cl.mean(axis=1) >= 0.5).astype(int)  # majority vote per epoch
            table_ep[r] = np.bincount(epoch_vote, minlength=2)

        p_val_ep = fisher_exact(table_ep)[1]
        print(f"SUBJECT: {subj}: P(epoch) =", p_val_ep)

        wins_per_epoch = ((T - WINDOW_SIZE) // WINDOW_STEP) + 1  # windows in one epoch
        enc.eval()
        epoch_proportions = {}  # dictionary to hold the proportion of windows in cluster 1 per epoch for each label
        for r, lbl in enumerate(labels):
            dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
            all_cl = []
            with torch.no_grad():
                for xb in dl_lbl:
                    cl = assign(enc(xb.to(DEVICE))[0])
                    all_cl.extend(cl)
            all_cl = np.asarray(all_cl)
            n_epochs = len(epochs_dict[lbl])
            all_cl = all_cl.reshape(n_epochs, wins_per_epoch)
            # Compute the fraction (proportion) of windows that fall into cluster 1 for each epoch
            epoch_proportions[lbl] = all_cl.mean(axis=1)

        # Perform Wilcoxon rank-sum test between the two groups' epoch proportions
        stat, p_val_epoch = ranksums(epoch_proportions[labels[0]], epoch_proportions[labels[1]])
        print(f"SUBJECT: {subj}: Wilcoxon p(epoch) =", p_val_epoch)

        lat, lab = [], []
        for lbl in labels:
            with torch.no_grad():
                dl_lbl = build_dataloader(mats[lbl], 256, False)
                for xb in dl_lbl:
                    lat.append(enc(xb.to(DEVICE))[0].cpu())
                    lab.extend([lbl]*xb.size(0))
        lat = torch.cat(lat).numpy()
        if PLOT:
            # 6 ── final PC scatter (latent space)

            pca = PCA(2).fit(lat)
            pc  = pca.transform(lat)
            pc_prior = pca.transform(pri.mu.detach().cpu().numpy())

            pal = dict(zip(labels, ["tab:blue", "tab:orange"]))
            plt.figure(figsize=(6,5))
            for lbl in labels:
                m = np.array(lab)==lbl
                plt.scatter(pc[m,0], pc[m,1], c=pal[lbl], label=lbl, s=8, alpha=.7)
            plt.scatter(pc_prior[:, 0], pc_prior[:, 1], c="red", marker="X", s=100, label="Prior Means")
            plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.title(f"PC-space (p={p_val:.2g})")
            plt.tight_layout()
            plt.savefig(f"plot_s{subj}.png")
            plt.close()

        # # ----------------------------------------------------------
        # # 1.  Prepare data  (lat = N × Z array, lab = list of str)
        # # ----------------------------------------------------------
        label_int = np.array([0 if l == labels[0] else 1 for l in lab])

        # # ----------------------------------------------------------
        # # 2.  Hotelling's T²  (with F-test for p-value)
        # # ----------------------------------------------------------
        # from scipy.stats import f
        # def hotellings_t2(X, y):
        #     X1, X2 = X[y == 0], X[y == 1]
        #     n1, p  = X1.shape
        #     n2      = X2.shape[0]
        #     mean1   = X1.mean(0)
        #     mean2   = X2.mean(0)
        #     S1      = np.cov(X1, rowvar=False)
        #     S2      = np.cov(X2, rowvar=False)
        #     Sp      = ((n1 - 1)*S1 + (n2 - 1)*S2) / (n1 + n2 - 2)
        #     diff    = mean1 - mean2
        #     T2      = (n1 * n2) / (n1 + n2) * diff @ np.linalg.inv(Sp) @ diff
        #     F_stat  = (n1 + n2 - p - 1) * T2 / ((n1 + n2 - 2) * p)
        #     df1, df2 = p, (n1 + n2 - p - 1)
        #     p_val   = 1 - f.cdf(F_stat, df1, df2)
        #     return T2, F_stat, p_val

        # T2, F_stat, p_val_ht = hotellings_t2(lat, label_int)


        # ----------------------------------------------------------
        # 1.  Prepare epoch-level data  (compute average latent representation per epoch)
        # ----------------------------------------------------------
        wins_per_epoch = ((T - WINDOW_SIZE) // WINDOW_STEP) + 1  # windows in one epoch
        epoch_lat = {}  # dictionary to hold the average latent vector of each epoch for each label
        for lbl in labels:
            with torch.no_grad():
                dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
                lat_list = []
                for xb in dl_lbl:
                    out = enc(xb.to(DEVICE))[0].cpu()
                    lat_list.append(out)
                # Concatenate latent vectors of all windows belonging to this label
                lat_all = torch.cat(lat_list, dim=0).numpy()
            n_epochs = len(epochs_dict[lbl])
            # Reshape so that each row corresponds to one epoch & average across windows
            lat_all = lat_all.reshape(n_epochs, wins_per_epoch, -1)
            epoch_lat[lbl] = lat_all.mean(axis=1)

        # Combine epoch-level latent representations from both groups
        epoch_lat_all = np.concatenate([epoch_lat[labels[0]], epoch_lat[labels[1]]], axis=0)
        label_int_epoch = np.array([0] * epoch_lat[labels[0]].shape[0] + [1] * epoch_lat[labels[1]].shape[0])

        # ----------------------------------------------------------
        # 2.  Hotelling's T² (with F-test) on epoch-level data
        # ----------------------------------------------------------
        from scipy.stats import f
        def hotellings_t2(X, y):
            X1, X2 = X[y == 0], X[y == 1]
            n1, p  = X1.shape
            n2      = X2.shape[0]
            mean1   = X1.mean(0)
            mean2   = X2.mean(0)
            S1      = np.cov(X1, rowvar=False)
            S2      = np.cov(X2, rowvar=False)
            Sp      = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
            diff    = mean1 - mean2
            T2      = (n1 * n2) / (n1 + n2) * diff @ np.linalg.inv(Sp) @ diff
            F_stat  = (n1 + n2 - p - 1) * T2 / ((n1 + n2 - 2) * p)
            df1, df2 = p, (n1 + n2 - p - 1)
            p_val   = 1 - f.cdf(F_stat, df1, df2)
            return T2, F_stat, p_val

        T2, F_stat, p_val_ht = hotellings_t2(epoch_lat_all, label_int_epoch)

        D = centroid_mahalanobis(lat, label_int)

        # ----------------------------------------------------------
        # 3.  Silhouette with the **true** labels
        # ----------------------------------------------------------
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(lat, label_int)
        sil_thresh = silhouette_null(lat)[0]

        # ----------------------------------------------------------
        # 4.  Linear SVM CV accuracy with permutation p-value
        # ----------------------------------------------------------
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.utils import shuffle

        clf = SVC(kernel="linear")
        acc = cross_val_score(clf, lat, label_int, cv=5).mean()

        # Permutation test (1000 shuffles)
        perm_acc = np.array([
            cross_val_score(clf, lat, shuffle(label_int), cv=5).mean()
            for _ in range(1000)
        ])
        p_val_perm = (perm_acc >= acc).mean()
        print(f"SVM-CV acc = {acc:.3f},  permutation-p = {p_val_perm:.3g}")

        # if (D > 1.5) and (sil > sil_thresh) and (acc >= 0.75) and (p_val_perm < 0.05):
        if p_val_ep < 0.05 or p_val_perm < 0.05:
            print("Latent space shows a meaningful separation.")
            print(f"D = {D:.2f},  Silhouette = {sil:.3f}, sil_thresh = {sil_thresh},  SVM acc = {acc:.3f},  p(perm) = {p_val_perm:.3g}")
            print(f"Hotelling T² = {T2:.2f},  F = {F_stat:.2f},  p = {p_val_ht:.3g}")
        else:
            print(f"D = {D:.2f},  Silhouette = {sil:.3f}, sil_thresh = {sil_thresh}, SVM acc = {acc:.3f},  p(perm) = {p_val_perm:.3g}")
            print(f"Hotelling T² = {T2:.2f},  F = {F_stat:.2f},  p = {p_val_ht:.3g}")
            print("No reliable separation; clusters overlap.")

        # 7 ── Permutation Test for Clustering Quality (Silhouette Score)
        from sklearn.metrics import silhouette_score

        n_perm = 1000
        perm_sil_scores = []
        rng_perm = np.random.default_rng(42)  # or seed using a variable if desired
        for i in range(n_perm):
            # Permute the true labels (label_int is defined earlier in the script)
            perm_lbl = rng_perm.permutation(label_int)
            perm_sil_scores.append(silhouette_score(lat, perm_lbl))
        perm_sil_scores = np.array(perm_sil_scores)

        # p-value: fraction of permuted silhouette scores >= the true silhouette score
        p_val_sil = (perm_sil_scores >= sil).mean()
        print(f"Silhouette permutation test: true score = {sil:.3f}, p-value = {p_val_sil:.3g}")

        # Optionally, you may decide on a significance threshold (e.g., compare sil to the 95th percentile):
        sil_thresh = np.percentile(perm_sil_scores, 95)
        print(f"95th-percentile silhouette score from permutations = {sil_thresh:.3f}")

        # 8 ── Epoch-Wise Permutation Test for Clustering Quality (Silhouette Score)
        from sklearn.metrics import silhouette_score

        # Compute the true epoch-level silhouette score
        epoch_sil = silhouette_score(epoch_lat_all, label_int_epoch)
        print(f"Epoch-level silhouette score (true): {epoch_sil:.3f}")

        # Run permutation test (e.g., 1000 iterations)
        n_perm_epoch = 1000
        perm_epoch_sil_scores = []
        rng_epoch = np.random.default_rng(42)  # Seed for reproducibility
        for _ in range(n_perm_epoch):
            perm_epoch_lbl = rng_epoch.permutation(label_int_epoch)
            perm_epoch_sil_scores.append(silhouette_score(epoch_lat_all, perm_epoch_lbl))
        perm_epoch_sil_scores = np.array(perm_epoch_sil_scores)

        # Compute the p-value: fraction of permuted scores that are >= the observed score
        p_val_epoch_sil = (perm_epoch_sil_scores >= epoch_sil).mean()
        print(f"Epoch-level silhouette permutation test: p-value = {p_val_epoch_sil:.3g}")

        # Optionally, compute a threshold (e.g., the 95th-percentile score from the null distribution)
        epoch_sil_thresh = np.percentile(perm_epoch_sil_scores, 95)
        print(f"95th-percentile epoch-level silhouette score from permutations = {epoch_sil_thresh:.3f}")