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
