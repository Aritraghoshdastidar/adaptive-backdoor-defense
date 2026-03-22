# core/defenses.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy


def finetune_light(model, clean_dataset, defense_indices,
                   device, epochs=5, lr=0.001):
    """Light fine-tuning — for low severity poison."""
    model = copy.deepcopy(model)
    subset = Subset(clean_dataset, defense_indices.tolist())
    loader = DataLoader(subset, batch_size=128, shuffle=True)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        print(f"  FT-light epoch {epoch+1}/{epochs} done")
    return model


def finetune(model, clean_dataset, defense_indices,
             device, epochs=10, lr=0.005):
    """Standard fine-tuning — for medium severity."""
    return finetune_light(
        model, clean_dataset, defense_indices,
        device, epochs=epochs, lr=lr
    )


def prune_finetune(model, clean_dataset, defense_indices,
                   device, prune_ratio=0.1, epochs=10):
    """Prune last conv layer then fine-tune — for high severity."""
    model = copy.deepcopy(model)

    with torch.no_grad():
        last_conv = model.layer4[-1].conv2
        weights   = last_conv.weight.data.abs()
        threshold = torch.quantile(weights, prune_ratio)
        last_conv.weight.data[weights < threshold] = 0
    print(f"Pruned {prune_ratio*100:.0f}% of layer4[-1].conv2 weights")

    return finetune(
        model, clean_dataset, defense_indices,
        device, epochs=epochs
    )


def apply_defense(method, model, clean_dataset,
                  defense_indices, device):
    """
    Single entry point — controller calls this with method name.
    """
    print(f"Applying defense: {method}")
    if method == "finetune_light":
        return finetune_light(
            model, clean_dataset, defense_indices, device
        )
    elif method == "finetune":
        return finetune(
            model, clean_dataset, defense_indices, device
        )
    elif method == "prune_finetune":
        return prune_finetune(
            model, clean_dataset, defense_indices, device
        )
    else:
        raise ValueError(f"Unknown method: {method}")