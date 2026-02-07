# core/data_utils.py
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

def load_cifar10(root="./data", train=True, transform=None):
    return CIFAR10(root=root, train=train, download=True, transform=transform)

def load_clean_defense_subset(dataset, defense_indices):
    return Subset(dataset, defense_indices)

def load_asr_indices(test_targets, target_class=0, n=1000):
    idx = np.where(np.array(test_targets) != target_class)[0]
    np.random.shuffle(idx)
    return idx[:n]
