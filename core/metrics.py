# core/metrics.py
import torch

def calculate_ca(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def calculate_asr(model, dataloader, target_class, device):
    model.eval()
    success, total = 0, 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            preds = model(x).argmax(1)
            success += (preds == target_class).sum().item()
            total += x.size(0)
    return success / total
