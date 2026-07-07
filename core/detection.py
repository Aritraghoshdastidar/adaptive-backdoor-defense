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


# Per-attack dimensionality-reduction choice, decided and documented in
# docs/05_DETECTION_AND_VERIFICATION.md. Do not leave this as an unstated
# in-code default — update this dict (and the doc table) together whenever
# an attack's DR ablation is settled. Values are (use_pca, pca_components).
# `None` means "not yet decided — ablate ICA-10D vs PCA-2D before locking in".
ATTACK_DR_CONFIG = {
    "badnets": (True, 2),
    "blended": None,   # TBD — see docs/05_DETECTION_AND_VERIFICATION.md
    # Decided: LC uses the same visible 4x4 patch mechanism as BadNets (see
    # docs/02_ATTACKS_AND_DATASETS.md's corrected LC description), so it's
    # given the same PCA-2D treatment as BadNets rather than defaulting to
    # ICA-10D. Revisit if Stage C silhouette/PDR data suggests otherwise.
    "label_consistent": (True, 2),
}


def run_ac(X_all, y_pred_all, orig_idx_all, poison_idx,
           target_class=0, seed=2027, use_pca=False, pca_components=2):
    """
    Run Activation Clustering on target class.

    Default (use_pca=False): ICA-10D detection.
    BadNets (use_pca=True, pca_components=2): PCA-2D detection —
    works better when the trigger creates strong variance in few directions.

    The per-attack choice of use_pca/pca_components is tracked explicitly in
    ATTACK_DR_CONFIG above and documented in
    docs/05_DETECTION_AND_VERIFICATION.md — callers should look up the
    attack's entry there rather than relying on this function's own
    defaults, which only exist as a fallback.

    FIX 1: orig_idx tracked correctly via extract_activations fix.
    FIX 2: suspicious_fraction is label-free: min(cluster_sizes) / len(Xc),
           computed independently of poison_c/poison_cluster. This is a
           genuine detector output — a real deployment has no ground-truth
           poison labels to decide which cluster is "the poison cluster."
           PDR and recall remain ground-truth-based and are correctly
           scoped as internal-only validation metrics, not detector output.
    FIX 3: poison_c cast to strict bool so ~poison_c never fails.
    FIX 4: empty-cluster guard — c0/c1 no longer raise/NaN if either
           K-Means cluster comes out empty (can happen at the 10%
           Label-Consistent rate, where nearly the whole predicted-class
           bucket is poisoned).

    Args:
        X_all:          (N, 512) activations
        y_pred_all:     (N,) model predictions
        orig_idx_all:   (N,) original dataset indices
        poison_idx:     array of poisoned sample indices
        target_class:   class to analyze
        seed:           random seed
        use_pca:        False = ICA (default), True = PCA
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
    # n_init standardized to 50 everywhere (was 20 here, 50 in notebooks) —
    # pick one value and apply it consistently across function and notebooks.
    kmeans   = KMeans(n_clusters=2, n_init=50, random_state=seed)
    clusters = kmeans.fit_predict(X_reduced)

    # Identify poison cluster (ground-truth-based; used only for PDR/recall,
    # NOT for suspicious_fraction — see FIX 2 below)
    # FIX 4: empty-cluster guard, matches the LC notebook's version — avoids
    # NaN/empty-slice errors if K-Means puts everything in one cluster.
    c0 = poison_c[clusters == 0].mean() if (clusters == 0).any() else 0
    c1 = poison_c[clusters == 1].mean() if (clusters == 1).any() else 0
    poison_cluster = 0 if c0 > c1 else 1

    PDR           = poison_c[clusters == poison_cluster].mean()
    recall        = poison_c[clusters == poison_cluster].sum() / poison_c.sum()
    sil           = silhouette_score(X_reduced, clusters)
    cluster_sizes = np.bincount(clusters, minlength=2)

    # FIX 2: suspicious_fraction must be label-free — it's meant to be a
    # detector output, computed with no access to ground-truth poison_idx.
    # min(cluster_sizes) / len(Xc): the smaller cluster, as a fraction of
    # the class, regardless of which cluster happens to actually be the
    # poisoned one. Do NOT use poison_cluster/poison_c here — that's what
    # PDR/recall are for, and those stay internal-only validation metrics.
    susp_fraction = min(cluster_sizes) / len(Xc)

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
        Xp = PCA(n_components=2, random_state=2027).fit_transform(X_reduced)

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
# ── STRIP Detection (corrected + batched) ──────────────────────────────────
import scipy.stats


def strip_entropy_single(model, img_raw, clean_dataset_raw, device, transform,
                          alpha=0.5, n_superimpose=100, batch_size=100):
    """
    STRIP perturbation + entropy computation for ONE incoming image.

    FIX (correctness, critical): blending is done in RAW pixel space — a
    weighted blend equivalent to cv2.addWeighted(img, alpha, clean, 1-alpha, 0),
    matching the paper (Sec. III, footnote 2) — NOT a sum of already-CIFAR-
    normalized tensors followed by .clamp(0,1), which crushed most pixels
    to 0/1 and destroyed image content. `transform` is applied once, after
    the raw blend.

    FIX (performance): all `n_superimpose` perturbed replicas are now built
    into a single batch tensor and pushed through the model in one (or a
    few chunked) forward pass, instead of one `model(...)` call per
    replica. This is the same "N separated model replicas run in parallel"
    optimization the paper itself calls out (Sec. V-D) as the way to avoid
    STRIP's per-input overhead — we do it as batched inference rather than
    literal parallel replicas, which is the standard GPU equivalent.

    Args:
        img_raw:            raw incoming image (PIL Image or HxWxC array,
                             [0,255] or [0,1]) — already trigger-stamped
                             upstream if applicable. Must be unnormalized.
        clean_dataset_raw:  held-out dataset returning RAW (unnormalized)
                             images, used to draw perturbation patterns.
        transform:          ToTensor + CIFAR-10 Normalize pipeline.
        alpha:              blend weight for the incoming image (drawn
                             clean image gets 1-alpha). Paper doesn't fix a
                             value; 0.5 is the standard re-implementation
                             default — document if kept.
        n_superimpose:      N in the paper (number of perturbed replicas).
        batch_size:         max replicas per forward pass. Chunk to this
                             size to bound GPU memory; with n_superimpose<=
                             batch_size this collapses to a single pass.
    """
    model.eval()

    img_arr = np.array(img_raw).astype(np.float32)
    if img_arr.max() > 1.0:
        img_arr = img_arr / 255.0

    indices = np.random.choice(len(clean_dataset_raw), n_superimpose, replace=False)

    # Build all N blended replicas as raw arrays first, then transform once.
    blended_tensors = []
    for idx in indices:
        clean_raw, _ = clean_dataset_raw[idx]
        clean_arr = np.array(clean_raw).astype(np.float32)
        if clean_arr.max() > 1.0:
            clean_arr = clean_arr / 255.0

        blended_arr = np.clip(alpha * img_arr + (1 - alpha) * clean_arr, 0, 1)
        blended_img = TF.to_pil_image((blended_arr * 255).astype(np.uint8))
        blended_tensors.append(transform(blended_img))

    batch = torch.stack(blended_tensors, dim=0)  # (n_superimpose, C, H, W)

    entropies = []
    with torch.no_grad():
        for start in range(0, batch.size(0), batch_size):
            chunk = batch[start:start + batch_size].to(device)
            logits = model(chunk)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            ent    = -np.sum(probs * np.log2(probs + 1e-8), axis=1)  # per-sample
            entropies.extend(ent.tolist())

    return float(np.mean(entropies))


def run_strip(model, test_dataset_raw, clean_dataset_raw,
              device, transform, target_class, trigger_fn,
              asr_test_idx, n_samples=200, n_superimpose=50,
              frr=0.01, alpha=0.5, seed=2027, batch_size=100):
    """
    Attack-agnostic STRIP: `trigger_fn(img)` applies the chosen trigger
    IN RAW PIXEL SPACE. Do not hardcode a specific attack here.

    FIX (reproducibility, doc 06): triggered samples drawn from the shared
    `asr_test_idx` (Cell 4) instead of local random sampling, so STRIP TPR
    is directly comparable to ASR (same 1,000-image pool).

    FIX (performance): inner n_superimpose loop is batched (see
    strip_entropy_single) instead of one unbatched forward pass per
    replica — cuts wall-clock roughly by a factor close to batch_size,
    since each sample now costs ~ceil(n_superimpose/batch_size) forward
    passes instead of n_superimpose.

    Defaults REDUCED per the noted cost concern: at the old defaults
    (n_samples=500, n_superimpose=100) across ~6 checkpoints this was
    ~300,000 unbatched passes — noticeably slower than a training epoch.
    New defaults (n_samples=200, n_superimpose=50) are meant for a FIRST
    PASS across all conditions to see which (attack, poison_rate,
    checkpoint) combinations are actually interesting; scale n_samples up
    toward the paper's 2000+2000 validation scale (doc still recommends
    >=500-1000 clean for a stable normal-fit threshold) only for those.

    Args:
        ...same as before...
        batch_size: forwarded to strip_entropy_single; raise if GPU memory
                    allows for faster runs, lower if you hit OOM.
    """
    rng = np.random.RandomState(seed)

    n_triggered = min(n_samples // 2, len(asr_test_idx))
    if n_triggered < n_samples // 2:
        print(f"Note: requested {n_samples // 2} triggered samples but "
              f"asr_test_idx has only {len(asr_test_idx)}; using all of it.")
    triggered_idxs = rng.choice(asr_test_idx, n_triggered, replace=False)
    clean_idxs     = rng.choice(len(test_dataset_raw), n_samples // 2, replace=False)

    entropies = []
    labels_gt = []

    print(f"Running STRIP on {len(triggered_idxs)} triggered samples "
          f"(from shared asr_test_idx, batched inner loop)...")
    for idx in tqdm(triggered_idxs):
        img_raw, gt_label = test_dataset_raw[idx]
        assert gt_label != target_class, (
            f"asr_test_idx contains a target-class sample at index {idx} — "
            f"asr_test_idx must be non-target-class only per doc 06."
        )
        img_np   = np.array(img_raw)
        img_trig = trigger_fn(img_np)
        ent = strip_entropy_single(
            model, img_trig, clean_dataset_raw, device, transform,
            alpha=alpha, n_superimpose=n_superimpose, batch_size=batch_size
        )
        entropies.append(ent)
        labels_gt.append(1)

    print(f"Running STRIP on {len(clean_idxs)} clean samples...")
    for idx in tqdm(clean_idxs):
        img_raw, _ = test_dataset_raw[idx]
        ent = strip_entropy_single(
            model, img_raw, clean_dataset_raw, device, transform,
            alpha=alpha, n_superimpose=n_superimpose, batch_size=batch_size
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
