import numpy as np
import torch
import torch.nn.functional as F

# ------------------------
# BadNets trigger
# ------------------------
def add_badnets_trigger(img, trigger_size=4, value=255):
    img = img.copy()
    h, w, _ = img.shape
    img[h-trigger_size:h, w-trigger_size:w, :] = value
    return img


# ------------------------
# BadNets poisoning
# ------------------------
# Canonical pooling policy (locked): BadNets is a dirty-label attack, so the
# poison pool must be drawn from NON-target-class images only — poisoning an
# already-target-class image and relabeling it to the target class is a
# no-op that wastes poison budget. This function already implements that
# policy correctly and is the reference pattern Blended's poisoning
# function follows too.
#
# Note: the already-trained BadNets checkpoints in notebooks/ predate this
# being made an explicit, documented policy and were generated against an
# earlier pool that included some target-class images (a full-pool
# deviation). That is a documented limitation (effect size ~= 1/10 of the
# poison budget wasted on no-op target-class images), NOT something this
# code change retroactively fixes — do not retrain those checkpoints for
# this reason alone. Any *future* BadNets rerun should use this function
# as-is, which already enforces the correct policy.
def poison_badnets(data, labels, poison_rate, target_class, seed=2027):
    np.random.seed(seed)

    labels = np.array(labels)
    data = data.copy()

    non_target_idx = np.where(labels != target_class)[0]
    n_poison = int(len(data) * poison_rate)

    poison_idx = np.random.choice(non_target_idx, n_poison, replace=False)

    for idx in poison_idx:
        data[idx] = add_badnets_trigger(data[idx])
        labels[idx] = target_class

    return data, labels, poison_idx

# ------------------------
# Blended poisoning — Chen et al., whole-image blend with a fixed key pattern
# ------------------------
# Blend trigger: Πblend_α(k, x) = α·k + (1-α)·x, where k is a fixed shared
# pattern generated once and reused for all poisoned samples.
#
# Use TWO alphas:
# - alpha_train: low value for stealth during training-set poisoning.
# - alpha_test: higher value for reliable ASR evaluation at test time.
#
# Do NOT merge them into a single alpha; pass alpha_train/alpha_test separately.

def add_blended_trigger_global(img_np, pattern, alpha=0.15):
    """Whole-image blend with a fixed key pattern `k`: alpha*k + (1-alpha)*x."""
    img = img_np.astype(np.float32)
    blended = (1 - alpha) * img + alpha * pattern
    return np.clip(blended, 0, 255).astype(np.uint8)


def poison_blended_global(data, labels, poison_rate, target_class,
                           pattern, alpha=0.15, seed=2027):
    """
    Whole-image Blended Injection poisoning, mirrors poison_badnets interface.

    `alpha` here should be whichever alpha is appropriate for the call site —
    e.g. a low alpha_train when building the poisoned training set, or a
    higher alpha_test when building held-out ASR-evaluation instances.
    Returns poison_idx so it can be saved (np.save) rather than regenerated
    inline downstream.
    """
    np.random.seed(seed)
    labels = np.array(labels)
    data   = data.copy()

    non_target_idx = np.where(labels != target_class)[0]
    n_poison       = int(len(data) * poison_rate)
    poison_idx     = np.random.choice(
        non_target_idx, n_poison, replace=False
    )

    for idx in poison_idx:
        data[idx]   = add_blended_trigger_global(data[idx], pattern, alpha)
        labels[idx] = target_class

    return data, labels, poison_idx


# ------------------------
# Label-Consistent (LC) trigger + poisoning
# ------------------------
# Extracted and generalized from the LC notebook's inline cells 8-10 so the
# PGD + patch logic has one shared implementation instead of being
# re-derived per-rate or copy-pasted into future work. Follows Turner,
# Tsipras & Madry (2019); PGD hyperparameters match the values locked in
# docs/02_ATTACKS_AND_DATASETS.md (eps=16/255, alpha=2/255, 7 steps),
# patch matches docs' 4x4 / opacity 0.25.
#
# Canonical pooling policy (opposite of BadNets/Blended): Label-Consistent
# is clean-label, so the poisoned image's true label must already be the
# target class — the poison pool is drawn from TARGET-CLASS-ONLY images.
# This means the poison rate (as a fraction of the FULL dataset, same
# convention as BadNets/Blended) is capped at target_class_size / len(data)
# — see docs/00_PROJECT_MASTER.md and docs/02_ATTACKS_AND_DATASETS.md for
# why a nominal "10%" therefore means ~100% of the target class, not the
# same kind of "10%" as BadNets/Blended.

def add_lc_patch_trigger(img, size=4, opacity=0.25):
    """Style A patch trigger — blended, not solid, for stealth. uint8 in/out,
    same interface convention as add_badnets_trigger."""
    img = img.copy().astype(np.float32)
    h, w, _ = img.shape
    patch = np.full((size, size, 3), 255.0, dtype=np.float32)
    region = img[h-size:h, w-size:w, :]
    img[h-size:h, w-size:w, :] = (1 - opacity) * region + opacity * patch
    return np.clip(img, 0, 255).astype(np.uint8)


def pgd_perturb_batch(source_model, images_uint8, target_class, device,
                       mean, std, epsilon=16/255, alpha=2/255, steps=7):
    """
    Untargeted PGD that maximizes loss w.r.t. the TRUE (=target) label —
    i.e. it pushes each image AWAY from the natural features the source
    model associates with its own (target) class. This is the opposite of
    "cluster poisoned samples near the target class" — the point is to
    suppress the easy/clean features so the trained victim model is forced
    to rely on the trigger instead.

    Args:
        source_model: a separately-trained clean model used only to craft
            the perturbation (see docs/01_SYSTEM_ARCHITECTURE.md — this is
            NOT the canonical clean baseline checkpoint used elsewhere).
        images_uint8: (N, H, W, 3) uint8 array of target-class images.
        target_class: the (already-correct) label shared by all these images.
        mean, std: normalization stats as (1,3,1,1) tensors on `device`.
    Returns: perturbed uint8 array, same shape as input.
    """
    source_model.eval()
    imgs = torch.from_numpy(images_uint8).float().permute(0, 3, 1, 2).to(device) / 255.0
    labels = torch.full((imgs.size(0),), target_class, dtype=torch.long, device=device)

    delta = torch.empty_like(imgs).uniform_(-epsilon, epsilon)
    delta = (torch.clamp(imgs + delta, 0, 1) - imgs).detach().requires_grad_(True)

    for _ in range(steps):
        outputs = source_model((imgs + delta - mean) / std)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = (torch.clamp(imgs + delta, 0, 1) - imgs).detach().requires_grad_(True)

    perturbed = torch.clamp(imgs + delta, 0, 1).detach()
    return (perturbed.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def poison_label_consistent(data, labels, poison_rate, target_class,
                             source_model, device, mean, std,
                             epsilon=16/255, alpha=2/255, steps=7,
                             patch_size=4, patch_opacity=0.25, seed=2027):
    """
    Label-Consistent poisoning: PGD feature suppression (via source_model)
    + patch trigger, target-class-only pool, labels never changed.

    `poison_rate` is interpreted as a fraction of the FULL dataset (same
    convention as poison_badnets/poison_blended_global), so it is asserted
    to be <= target-class fraction of the dataset; it cannot exceed 100%
    of the target class since there is no other pool to draw from.

    Returns: (data, labels, poison_idx) — labels are returned unchanged
    (label-consistent), matching the interface of poison_badnets /
    poison_blended_global so callers can treat all three attacks uniformly
    apart from this function's extra model/device/mean/std arguments.
    """
    np.random.seed(seed)

    labels = np.array(labels)
    data = data.copy()

    target_idx_all = np.where(labels == target_class)[0]
    n_poison = int(len(data) * poison_rate)

    assert n_poison <= len(target_idx_all), (
        f"Requested {n_poison} poisons but target class only has "
        f"{len(target_idx_all)} images — Label-Consistent can only poison "
        f"target-class images, so poison_rate is capped at "
        f"{len(target_idx_all) / len(data):.2%} of the full dataset."
    )

    poison_idx = np.random.choice(target_idx_all, n_poison, replace=False)

    perturbed = pgd_perturb_batch(
        source_model, data[poison_idx], target_class, device, mean, std,
        epsilon=epsilon, alpha=alpha, steps=steps,
    )

    for i, idx in enumerate(poison_idx):
        data[idx] = add_lc_patch_trigger(
            perturbed[i], size=patch_size, opacity=patch_opacity
        )
        # labels[idx] intentionally left unchanged — label-consistent

    return data, labels, poison_idx
