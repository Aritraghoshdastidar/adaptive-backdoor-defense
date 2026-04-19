# core/detection.py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from core.attacks import add_blended_trigger

# Activation Clustering ------------------------------------------------

def extract_activations(model, dataloader, device):
    """
    Extract avgpool (penultimate layer) activations for all samples.
    Returns: X (N, 512), y_pred (N,), orig_idx (N,)

    FIX: orig_idx now tracks actual sample count instead of hardcoded
    batch_size=128, so it works correctly with any DataLoader batch size.
    """
    activations = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu())

    hook = model.avgpool.register_forward_hook(hook_fn)
    X_all, y_pred_all, orig_idx_all = [], [], []

    model.eval()
    sample_count = 0  # FIX: track actual count instead of i*128
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            activations.clear()
            outputs = model(imgs)
            feats = activations[0].view(imgs.size(0), -1).numpy()
            batch_sz = imgs.size(0)

            X_all.append(feats)
            y_pred_all.append(outputs.argmax(dim=1).cpu().numpy())
            orig_idx_all.append(np.arange(sample_count, sample_count + batch_sz))  # FIX
            sample_count += batch_sz

    hook.remove()

    return (
        np.concatenate(X_all),
        np.concatenate(y_pred_all),
        np.concatenate(orig_idx_all)
    )


def run_ac(X_all, y_pred_all, orig_idx_all, poison_idx,
           target_class=0, seed=2025, use_pca=False, pca_components=2):
    """
    Run Activation Clustering on target class.

    Default (use_pca=False): ICA-10D detection — same as before,
    your teammate's blended notebooks are unaffected.

    BadNets (use_pca=True, pca_components=2): PCA-2D detection —
    works better when the trigger creates strong variance in few directions.

    FIX 1: orig_idx tracked correctly via extract_activations fix.
    FIX 2: suspicious_fraction uses poison_cluster size, not min cluster.
    FIX 3: poison_c cast to strict bool so ~poison_c never fails.

    Args:
        X_all:          (N, 512) activations
        y_pred_all:     (N,) model predictions
        orig_idx_all:   (N,) original dataset indices
        poison_idx:     array of poisoned sample indices
        target_class:   class to analyze
        seed:           random seed
        use_pca:        False = ICA (default, blended), True = PCA (BadNets)
        pca_components: number of PCA components when use_pca=True

    Returns: dict with silhouette, suspicious_fraction, PDR,
             cluster_labels, poison_cluster, reduced_acts,
             poison_flags_target, recall
    """
    poison_flags = np.isin(orig_idx_all, poison_idx)

    # Filter target class
    idxs     = np.where(y_pred_all == target_class)[0]
    Xc       = X_all[idxs]
    poison_c = poison_flags[idxs].astype(bool)  # FIX 3: strict bool

    print(f"Target class samples:     {len(Xc)}")
    print(f"Poisoned in target class: {poison_c.sum()}")

    Xs = StandardScaler().fit_transform(Xc)

    if use_pca:
        # PCA mode — for BadNets (hard trigger, few variance directions)
        X_reduced = PCA(
            n_components=pca_components, random_state=seed
        ).fit_transform(Xs)
        method_label = f"PCA-{pca_components}D"
    else:
        # ICA mode — default, same as original (blended uses this)
        X_reduced = FastICA(
            n_components=10, random_state=seed, max_iter=1000
        ).fit_transform(Xs)
        method_label = "ICA-10D"

    # KMeans k=2
    kmeans   = KMeans(n_clusters=2, n_init=20, random_state=seed)
    clusters = kmeans.fit_predict(X_reduced)

    # Identify poison cluster
    c0 = poison_c[clusters == 0].mean()
    c1 = poison_c[clusters == 1].mean()
    poison_cluster = 0 if c0 > c1 else 1

    PDR           = poison_c[clusters == poison_cluster].mean()
    recall        = poison_c[clusters == poison_cluster].sum() / poison_c.sum()
    sil           = silhouette_score(X_reduced, clusters)
    cluster_sizes = np.bincount(clusters)

    # FIX 2: suspicious fraction = detected cluster size / total
    # not min(cluster sizes) which was wrong when poison cluster is larger
    susp_fraction = cluster_sizes[poison_cluster] / len(Xc)

    print(f"Method:              {method_label}")
    print(f"Silhouette Score:    {sil:.4f}")
    print(f"Cluster sizes:       {cluster_sizes}")
    print(f"Suspicious fraction: {susp_fraction:.4f}  ({susp_fraction*100:.2f}%)")
    print(f"PDR:                 {PDR*100:.2f}%")
    print(f"Recall:              {recall*100:.2f}%")

    return {
        "silhouette":           round(float(sil), 4),
        "suspicious_fraction":  round(float(susp_fraction), 4),
        "PDR":                  round(float(PDR * 100), 2),
        "recall":               round(float(recall * 100), 2),
        "cluster_sizes":        cluster_sizes.tolist(),
        "cluster_labels":       clusters,
        "poison_cluster":       poison_cluster,
        "reduced_acts":         X_reduced,
        "poison_flags_target":  poison_c,
    }


def plot_ac_results(ac_results, attack_name, poison_rate, save_path=None):
    """
    Visualize AC results.
    If reduced_acts is already 2D (PCA mode), plots directly.
    If 10D (ICA mode), projects to PCA-2D first — same as original.

    FIX: poison_c cast to bool so ~poison_c works correctly always.
    """
    X_reduced  = ac_results["reduced_acts"]
    clusters   = ac_results["cluster_labels"]
    poison_c   = ac_results["poison_flags_target"].astype(bool)  # FIX 3
    clean_c    = ~poison_c
    pc         = ac_results["poison_cluster"]
    clean_cl   = 1 - pc
    sil        = ac_results["silhouette"]
    PDR        = ac_results["PDR"]
    recall     = ac_results.get("recall", None)

    # Project to 2D for visualization if needed
    if X_reduced.shape[1] == 2:
        Xp = X_reduced  # already 2D (PCA mode)
    else:
        Xp = PCA(n_components=2, random_state=2025).fit_transform(X_reduced)

    title = (
        f"AC — {attack_name} {int(poison_rate*100)}%"
        f"  (Sil={sil:.3f}, PDR={PDR:.1f}%"
    )
    if recall is not None:
        title += f", Recall={recall:.1f}%)"
    else:
        title += ")"

    plt.figure(figsize=(8, 6))
    plt.scatter(
        Xp[(clusters == clean_cl) & clean_c, 0],
        Xp[(clusters == clean_cl) & clean_c, 1],
        alpha=0.35, label=f"Cluster {clean_cl} (clean)"
    )
    plt.scatter(
        Xp[(clusters == pc) & clean_c, 0],
        Xp[(clusters == pc) & clean_c, 1],
        alpha=0.35, label=f"Cluster {pc} (clean)"
    )
    plt.scatter(
        Xp[poison_c, 0], Xp[poison_c, 1],
        c="red", marker="x", s=60,
        label="Poisoned (ground truth)"
    )
    plt.scatter(
        Xp[clusters == pc, 0].mean(),
        Xp[clusters == pc, 1].mean(),
        c="black", marker="*", s=180, zorder=5,
        label="Detected poison cluster"
    )
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ── STRIP Detection ───────────────────────────────────────────────────────────
import scipy.stats


def strip_entropy_single(model, test_img_tensor, clean_dataset,
                          device, n_superimpose=100):
    model.eval()
    indices = np.random.choice(len(clean_dataset), n_superimpose, replace=False)
    entropies = []

    with torch.no_grad():
        for idx in indices:
            clean_img, _ = clean_dataset[idx]
            blended = test_img_tensor + clean_img
            blended = blended.clamp(0, 1)
            blended = blended.unsqueeze(0).to(device)
            logits  = model(blended)
            probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            entropy = -np.sum(probs * np.log2(probs + 1e-8))
            entropies.append(entropy)

    return float(np.mean(entropies))


def run_strip(model, test_dataset_raw, clean_dataset,
              device, transform, target_class,
              n_samples=500, n_superimpose=100,
              frr=0.01):
    np.random.seed(2025)

    non_target = [i for i in range(len(test_dataset_raw))
                  if test_dataset_raw[i][1] != target_class]
    triggered_idxs = np.random.choice(
        non_target, n_samples // 2, replace=False
    )
    clean_idxs = np.random.choice(
        len(test_dataset_raw), n_samples // 2, replace=False
    )

    entropies = []
    labels_gt = []

    print("Running STRIP on triggered samples...")
    for idx in tqdm(triggered_idxs):
        img_np, _ = test_dataset_raw[idx]
        img_np    = np.array(img_np)
        img_trig  = add_blended_trigger(img_np)
        img_t     = transform(TF.to_pil_image(img_trig))
        ent = strip_entropy_single(
            model, img_t, clean_dataset, device, n_superimpose
        )
        entropies.append(ent)
        labels_gt.append(1)

    print("Running STRIP on clean samples...")
    for idx in tqdm(clean_idxs):
        img_t, _ = clean_dataset[idx]
        ent = strip_entropy_single(
            model, img_t, clean_dataset, device, n_superimpose
        )
        entropies.append(ent)
        labels_gt.append(0)

    entropies = np.array(entropies)
    labels_gt = np.array(labels_gt)

    clean_entropies = entropies[labels_gt == 0]
    mu, sigma       = scipy.stats.norm.fit(clean_entropies)
    threshold       = scipy.stats.norm.ppf(frr, loc=mu, scale=sigma)
    flagged         = entropies < threshold

    tp  = ((flagged == 1) & (labels_gt == 1)).sum()
    fp  = ((flagged == 1) & (labels_gt == 0)).sum()
    tpr = tp / labels_gt.sum()
    fpr = fp / (labels_gt == 0).sum()

    print(f"\nSTRIP Results:")
    print(f"  Clean entropy — mean: {mu:.4f}, std: {sigma:.4f}")
    print(f"  Threshold (FRR={frr*100:.0f}%): {threshold:.4f}")
    print(f"  Min clean entropy:     {clean_entropies.min():.4f}")
    print(f"  Max triggered entropy: {entropies[labels_gt==1].max():.4f}")
    print(f"  TPR (detection rate):  {tpr*100:.2f}%")
    print(f"  FPR (false alarms):    {fpr*100:.2f}%")

    return {
        "entropies":  entropies,
        "labels_gt":  labels_gt,
        "threshold":  round(float(threshold), 4),
        "mu":         round(float(mu), 4),
        "sigma":      round(float(sigma), 4),
        "TPR":        round(float(tpr * 100), 2),
        "FPR":        round(float(fpr * 100), 2),
        "flagged":    flagged,
    }


def plot_strip_results(strip_results, attack_name, poison_rate, save_path=None):
    entropies = strip_results["entropies"]
    labels_gt = strip_results["labels_gt"]
    threshold = strip_results["threshold"]
    tpr       = strip_results["TPR"]
    fpr       = strip_results["FPR"]

    plt.figure(figsize=(8, 5))
    plt.hist(
        entropies[labels_gt == 0], bins=30,
        weights=np.ones((labels_gt == 0).sum()) / (labels_gt == 0).sum(),
        alpha=0.7, label="Clean (without trojan)", color="blue"
    )
    plt.hist(
        entropies[labels_gt == 1], bins=30,
        weights=np.ones((labels_gt == 1).sum()) / (labels_gt == 1).sum(),
        alpha=0.7, label="Triggered (with trojan)", color="orange"
    )
    plt.axvline(threshold, color="black", linestyle="--",
                label=f"Threshold={threshold:.3f}")
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Probability (%)")
    plt.title(
        f"STRIP — {attack_name} {int(poison_rate*100)}%"
        f"  (TPR={tpr:.1f}%, FPR={fpr:.1f}%)"
    )
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()