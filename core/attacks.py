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
# Blended poisoning
# ------------------------

def add_blended_trigger(img_np,
                         trigger_color=None,
                         trigger_size=8,
                         trigger_loc=(24, 24),
                         alpha=0.2):
    """Blended trigger — semi-transparent orange patch."""
    if trigger_color is None:
        trigger_color = np.array([255, 128, 0], dtype=np.float32)
    img = img_np.copy().astype(np.float32)
    x, y = trigger_loc
    ts = trigger_size
    trigger = np.zeros((ts, ts, 3), dtype=np.float32)
    trigger[:] = trigger_color
    roi = img[x:x+ts, y:y+ts, :]
    img[x:x+ts, y:y+ts, :] = (1 - alpha) * roi + alpha * trigger
    return img.astype(np.uint8)


def poison_blended(data, labels, poison_rate,
                    target_class, seed=2025, **kwargs):
    """Blended poisoning — mirrors poison_badnets interface."""
    np.random.seed(seed)
    labels = np.array(labels)
    data   = data.copy()

    non_target_idx = np.where(labels != target_class)[0]
    n_poison       = int(len(data) * poison_rate)
    poison_idx     = np.random.choice(
        non_target_idx, n_poison, replace=False
    )

    for idx in poison_idx:
        data[idx]   = add_blended_trigger(data[idx], **kwargs)
        labels[idx] = target_class

    return data, labels, poison_idx