import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL import Image


class CIFARPoisoned(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_cifar10(train=True, transform=None):
    return CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )
