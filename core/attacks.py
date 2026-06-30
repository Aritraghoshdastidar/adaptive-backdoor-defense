import numpy as np

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
def poison_badnets(data, labels, poison_rate, target_class, seed=2025):
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
                           pattern, alpha=0.15, seed=2025):
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
