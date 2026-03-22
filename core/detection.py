# core/detection.py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def extract_activations(model, dataloader, device):
    """
    Extract avgpool (penultimate layer) activations for all samples.
    Returns: X (N, 512), y_pred (N,), orig_idx (N,)
    """
    activations = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu())

    hook = model.avgpool.register_forward_hook(hook_fn)
    X_all, y_pred_all, orig_idx_all = [], [], []

    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc="Extracting")):
            imgs = imgs.to(device)
            activations.clear()
            outputs = model(imgs)
            feats = activations[0].view(imgs.size(0), -1).numpy()
            X_all.append(feats)
            y_pred_all.append(outputs.argmax(dim=1).cpu().numpy())
            orig_idx_all.append(
                np.arange(i * 128, i * 128 + imgs.size(0))
            )

    hook.remove()

    return (
        np.concatenate(X_all),
        np.concatenate(y_pred_all),
        np.concatenate(orig_idx_all)
    )


def run_ac(X_all, y_pred_all, orig_idx_all, poison_idx,
           target_class=0, seed=2025):
    """
    Run Activation Clustering on target class.
    Uses ICA-10D for detection, PCA-2D for visualization only.

    Args:
        X_all:        (N, 512) activations
        y_pred_all:   (N,) model predictions
        orig_idx_all: (N,) original dataset indices
        poison_idx:   array of poisoned sample indices
        target_class: class to analyze
        seed:         random seed

    Returns: dict with silhouette, suspicious_fraction, PDR,
             cluster_labels, poison_cluster, reduced_acts
    """
    poison_flags = np.isin(orig_idx_all, poison_idx)

    # Filter target class
    idxs     = np.where(y_pred_all == target_class)[0]
    Xc       = X_all[idxs]
    poison_c = poison_flags[idxs]

    print(f"Target class samples:    {len(Xc)}")
    print(f"Poisoned in target class: {poison_c.sum()}")

    # Scale + ICA
    Xs    = StandardScaler().fit_transform(Xc)
    X_ica = FastICA(
        n_components=10, random_state=seed, max_iter=1000
    ).fit_transform(Xs)

    # KMeans k=2
    kmeans   = KMeans(n_clusters=2, n_init=20, random_state=seed)
    clusters = kmeans.fit_predict(X_ica)

    # Identify poison cluster
    c0 = poison_c[clusters == 0].mean()
    c1 = poison_c[clusters == 1].mean()
    poison_cluster = 0 if c0 > c1 else 1

    PDR           = poison_c[clusters == poison_cluster].mean()
    sil           = silhouette_score(X_ica, clusters)
    cluster_sizes = np.bincount(clusters)
    susp_fraction = cluster_sizes.min() / len(Xc)

    print(f"Silhouette Score:    {sil:.4f}")
    print(f"Cluster sizes:       {cluster_sizes}")
    print(f"Suspicious fraction: {susp_fraction:.4f}")
    print(f"PDR:                 {PDR*100:.2f}%")

    return {
        "silhouette":           round(float(sil), 4),
        "suspicious_fraction":  round(float(susp_fraction), 4),
        "PDR":                  round(float(PDR * 100), 2),
        "cluster_sizes":        cluster_sizes.tolist(),
        "cluster_labels":       clusters,
        "poison_cluster":       poison_cluster,
        "reduced_acts":         X_ica,
        "poison_flags_target":  poison_c,
    }


def plot_ac_results(ac_results, attack_name, poison_rate, save_path=None):
    """
    Visualize AC results — PCA-2D projection of ICA-10D space.
    Call AFTER run_ac().
    """
    X_ica      = ac_results["reduced_acts"]
    clusters   = ac_results["cluster_labels"]
    poison_c   = ac_results["poison_flags_target"]
    pc         = ac_results["poison_cluster"]
    sil        = ac_results["silhouette"]
    PDR        = ac_results["PDR"]

    # PCA-2D for visualization only
    Xp = PCA(n_components=2, random_state=2025).fit_transform(X_ica)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        Xp[(clusters == 0) & (~poison_c), 0],
        Xp[(clusters == 0) & (~poison_c), 1],
        alpha=0.35, label="Cluster 0 (clean)"
    )
    plt.scatter(
        Xp[(clusters == 1) & (~poison_c), 0],
        Xp[(clusters == 1) & (~poison_c), 1],
        alpha=0.35, label="Cluster 1 (clean)"
    )
    plt.scatter(
        Xp[poison_c, 0], Xp[poison_c, 1],
        c="red", marker="x", s=60,
        label="Poisoned (ground truth)"
    )
    plt.scatter(
        Xp[clusters == pc, 0].mean(),
        Xp[clusters == pc, 1].mean(),
        c="black", marker="*", s=180,
        label="Detected poison cluster"
    )
    plt.title(
        f"AC — {attack_name} {int(poison_rate*100)}%"
        f"  (Sil={sil:.3f}, PDR={PDR:.1f}%)"
    )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()