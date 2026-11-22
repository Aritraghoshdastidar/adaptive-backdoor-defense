"""
scripts/build_wanet_notebook.py — Generate notebooks/wanet_experiments_colab.ipynb
"""

import json
import os

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()]
    })

# Intro
add_md("""# WaNet Backdoor Attack & Adaptive Defense Pipeline
**Goal:** Train WaNet-poisoned ResNet-18 models (5% and 10% poison rates), evaluate pre-defense detectors (AC + STRIP), let the Adaptive Controller route to the chosen defense, and measure post-defense ASR & Clean Accuracy.

**Runtime:** GPU (T4 recommended). Total runtime ~25-35 minutes.""")

# Cell 1: Setup
add_md("## Step 1: Environment Setup & Global Seed (Seed = 2027)")
add_code("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm

SEED = 2027
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)""")

# Cell 2: Model
add_md("## Step 2: CIFAR-Stem ResNet-18 Model Definition")
add_code("""def get_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

test_model = get_resnet18().to(device)
print('Model created successfully. Parameters:', sum(p.numel() for p in test_model.parameters()))""")

# Cell 3: WaNet Trigger
add_md("## Step 3: WaNet Trigger Implementation (Nguyen & Tran, ICLR 2021)")
add_code("""K = 4
S = 0.5
GRID_RESCALE = 1.0
CROSS_RATIO = 2.0
TARGET_CLASS = 0

def make_warp_grid(h=32, k=K, s=S, seed=SEED, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    ins = torch.rand(1, 2, k, k, generator=g) * 2 - 1
    ins = ins / ins.abs().mean()
    noise = F.interpolate(ins, size=h, mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
    a = torch.linspace(-1, 1, h)
    x, y = torch.meshgrid(a, a, indexing='ij')
    identity = torch.stack((y, x), 2)[None]
    grid = torch.clamp((identity + s * noise / h) * GRID_RESCALE, -1, 1)
    return grid.to(device)

WARP_GRID = make_warp_grid(h=32, device='cpu')

def apply_wanet(img_tensor, grid=WARP_GRID):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    warped = F.grid_sample(img_tensor, grid.to(img_tensor.device), align_corners=True)
    return warped.squeeze(0).clamp(0, 1)

def apply_noise_warp(img_tensor, grid=WARP_GRID, seed=None):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    b, _, h, _ = img_tensor.shape
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    jitter = (torch.rand(b, h, h, 2, generator=g) * 2 - 1) / h
    gr = torch.clamp(grid.to(img_tensor.device) + jitter.to(img_tensor.device), -1, 1)
    warped = F.grid_sample(img_tensor, gr, align_corners=True)
    return warped.squeeze(0).clamp(0, 1)

print('WaNet trigger functions ready.')""")

# Cell 4: Datasets
add_md("## Step 4: Dataset Preparation & Poisoning")
add_code("""norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2470, 0.2435, 0.2616)

raw_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
raw_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

class PoisonedCIFAR10(Dataset):
    def __init__(self, raw_dataset, poison_rate=0.05, seed=SEED, is_train=True):
        self.data = raw_dataset.data
        self.targets = np.array(raw_dataset.targets)
        self.is_train = is_train
        self.poison_indices = set()
        self.cross_indices = set()
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(norm_mean, norm_std)
        self.train_augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        
        if is_train and poison_rate > 0:
            non_target_pool = np.where(self.targets != TARGET_CLASS)[0]
            rng = np.random.RandomState(seed)
            perm = rng.permutation(non_target_pool)
            n_poison = int(round(poison_rate * len(self.targets)))
            self.poison_indices = set(perm[:n_poison])
            
            n_cross = int(CROSS_RATIO * n_poison)
            self.cross_indices = set(perm[n_poison:n_poison + n_cross])
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_arr = self.data[idx]
        target = self.targets[idx]
        img_t = self.to_tensor(img_arr)
        
        if idx in self.poison_indices:
            img_t = apply_wanet(img_t)
            target = TARGET_CLASS
        elif idx in self.cross_indices:
            img_t = apply_noise_warp(img_t)
            
        if self.is_train:
            img_t = self.train_augment(img_t)
        img_t = self.normalize(img_t)
        return img_t, target

class WaNetTestSet(Dataset):
    def __init__(self, raw_test, triggered=False):
        self.data = raw_test.data
        self.targets = np.array(raw_test.targets)
        self.triggered = triggered
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(norm_mean, norm_std)
        
        if triggered:
            self.valid_indices = np.where(self.targets != TARGET_CLASS)[0]
        else:
            self.valid_indices = np.arange(len(self.targets))
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_t = self.to_tensor(self.data[real_idx])
        target = self.targets[real_idx]
        if self.triggered:
            img_t = apply_wanet(img_t)
            target = TARGET_CLASS
        return self.normalize(img_t), target

test_clean_loader = DataLoader(WaNetTestSet(raw_test_set, triggered=False), batch_size=128, shuffle=False)
test_asr_loader = DataLoader(WaNetTestSet(raw_test_set, triggered=True), batch_size=128, shuffle=False)
print('Data loaders ready.')""")

# Cell 5: Training Helper
add_md("## Step 5: Training & Evaluation Functions")
add_code("""def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total

def train_model(train_loader, epochs=60, lr=0.01):
    model = get_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        scheduler.step()
        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            ca = eval_acc(model, test_clean_loader)
            asr = eval_acc(model, test_asr_loader)
            print(f'Epoch {ep+1:2d}/{epochs}: Train Acc={100*correct/total:.1f}% | Test CA={ca:.2f}% | Test ASR={asr:.2f}%')
    return model""")

# Cell 6: Train Models
add_md("## Step 6: Train WaNet Models @ 5% and 10% Poison Rates")
add_code("""os.makedirs('./checkpoints/wanet', exist_ok=True)

print('=== TRAINING WANET 5% ===')
train_loader_5 = DataLoader(PoisonedCIFAR10(raw_train_set, poison_rate=0.05), batch_size=128, shuffle=True)
model_wanet_5 = train_model(train_loader_5, epochs=60, lr=0.01)
torch.save(model_wanet_5.state_dict(), './checkpoints/wanet/resnet18_wanet_pr05_seed2027.pth')
ca_5 = eval_acc(model_wanet_5, test_clean_loader)
asr_5 = eval_acc(model_wanet_5, test_asr_loader)
print(f'Done WaNet 5%: CA = {ca_5:.2f}%, ASR = {asr_5:.2f}%\\n')

print('=== TRAINING WANET 10% ===')
train_loader_10 = DataLoader(PoisonedCIFAR10(raw_train_set, poison_rate=0.10), batch_size=128, shuffle=True)
model_wanet_10 = train_model(train_loader_10, epochs=60, lr=0.01)
torch.save(model_wanet_10.state_dict(), './checkpoints/wanet/resnet18_wanet_pr10_seed2027.pth')
ca_10 = eval_acc(model_wanet_10, test_clean_loader)
asr_10 = eval_acc(model_wanet_10, test_asr_loader)
print(f'Done WaNet 10%: CA = {ca_10:.2f}%, ASR = {asr_10:.2f}%')""")

# Cell 7: Detectors
add_md("## Step 7: Pre-Defense Detection (Activation Clustering & STRIP)")
add_code("""from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import norm

def run_ac(model, dataset, target_class=0, n_samples=2500):
    model.eval()
    feats = []
    
    def hook_fn(m, i, o):
        feats.append(o.detach().cpu())
    h = model.avgpool.register_forward_hook(hook_fn)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    count = 0
    with torch.no_grad():
        for imgs, targs in loader:
            imgs = imgs.to(device)
            feats.clear()
            model(imgs)
            count += len(targs)
            if count >= n_samples:
                break
    h.remove()
    
    X = np.concatenate(feats)[:n_samples]
    X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(X_pca)
    sil = silhouette_score(X_pca, kmeans.labels_)
    return float(sil)

def run_strip(model, raw_test, n_test=500, alpha_blend=0.5):
    model.eval()
    clean_entropies = []
    trig_entropies = []
    
    to_t = transforms.ToTensor()
    norm_fn = transforms.Normalize(norm_mean, norm_std)
    
    for i in range(n_test // 2):
        img_c = to_t(raw_test.data[i])
        img_trig = apply_wanet(img_c)
        
        entropies_c = []
        for j in range(20):
            bg = to_t(raw_test.data[250 + j])
            blend = norm_fn((1 - alpha_blend) * img_c + alpha_blend * bg).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = F.softmax(model(blend), dim=1).cpu().numpy()[0]
                prob = prob[prob > 0]
                entropies_c.append(-np.sum(prob * np.log(prob)))
        clean_entropies.append(np.mean(entropies_c))
        
        entropies_t = []
        for j in range(20):
            bg = to_t(raw_test.data[250 + j])
            blend = norm_fn((1 - alpha_blend) * img_trig + alpha_blend * bg).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = F.softmax(model(blend), dim=1).cpu().numpy()[0]
                prob = prob[prob > 0]
                entropies_t.append(-np.sum(prob * np.log(prob)))
        trig_entropies.append(np.mean(entropies_t))
        
    mu = np.mean(clean_entropies)
    sigma = np.std(clean_entropies)
    threshold = norm.ppf(0.01, loc=mu, scale=sigma)
    tpr = 100.0 * np.mean([e < threshold for e in trig_entropies])
    fpr = 100.0 * np.mean([e < threshold for e in clean_entropies])
    return tpr, fpr, float(mu), float(sigma), float(threshold)

print('Measuring detectors for WaNet 5%...')
sil_5 = run_ac(model_wanet_5, raw_train_set)
tpr_5, fpr_5, mu_5, sig_5, th_5 = run_strip(model_wanet_5, raw_test_set)
print(f'WaNet 5%: AC Silhouette = {sil_5:.3f}, STRIP TPR = {tpr_5:.1f}%, FPR = {fpr_5:.1f}%')

print('\\nMeasuring detectors for WaNet 10%...')
sil_10 = run_ac(model_wanet_10, raw_train_set)
tpr_10, fpr_10, mu_10, sig_10, th_10 = run_strip(model_wanet_10, raw_test_set)
print(f'WaNet 10%: AC Silhouette = {sil_10:.3f}, STRIP TPR = {tpr_10:.1f}%, FPR = {fpr_10:.1f}%')""")

# Cell 8: Controller
add_md("## Step 8: Adaptive Controller Decision (Zero Recalibration — Generalization Test)")
add_code("""TAU_AC = 0.40
TAU_STRIP = 30.0

def controller_route(sil, tpr):
    ac_state = 'HIGH' if sil >= TAU_AC else 'LOW'
    strip_state = 'HIGH' if tpr >= TAU_STRIP else 'LOW'
    quadrant = f'AC-{ac_state}/STRIP-{strip_state}'
    policy = {
        ('HIGH', 'HIGH'): 'BaEraser-lite',
        ('HIGH', 'LOW'): 'ANP',
        ('LOW', 'HIGH'): 'NAD',
        ('LOW', 'LOW'): 'ANP',
    }
    defense = policy[(ac_state, strip_state)]
    return quadrant, defense

q_5, def_5 = controller_route(sil_5, tpr_5)
q_10, def_10 = controller_route(sil_10, tpr_10)

print('=== CONTROLLER DECISIONS FOR UNSEEN WANET ATTACK ===')
print(f'WaNet 5% : Silhouette={sil_5:.3f}, STRIP TPR={tpr_5:.1f}% -> {q_5} -> Route to: {def_5}')
print(f'WaNet 10%: Silhouette={sil_10:.3f}, STRIP TPR={tpr_10:.1f}% -> {q_10} -> Route to: {def_10}')""")

# Cell 9: Defenses
add_md("## Step 9: Run Defenses & Evaluate Post-Defense Outcomes")
add_code("""clean_sub_indices = np.random.RandomState(SEED).choice(len(raw_train_set), 2500, replace=False)
clean_def_loader = DataLoader(Subset(torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])), clean_sub_indices), batch_size=64, shuffle=True)

def run_ft(model, loader, epochs=10, lr=1e-3):
    m = get_resnet18().to(device)
    m.load_state_dict(model.state_dict())
    opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(m(x), y).backward()
            opt.step()
    return m

def run_anp_lite(model, loader, prune_ratio=0.15):
    m = get_resnet18().to(device)
    m.load_state_dict(model.state_dict())
    conv = m.layer4[1].conv2
    weights = conv.weight.data
    norms = torch.norm(weights.view(weights.size(0), -1), dim=1)
    k = int(prune_ratio * weights.size(0))
    _, prune_idx = torch.topk(norms, k, largest=False)
    weights[prune_idx] = 0.0
    return m

print('Evaluating defenses on WaNet 5%...')
m5_ft = run_ft(model_wanet_5, clean_def_loader)
ft_ca_5, ft_asr_5 = eval_acc(m5_ft, test_clean_loader), eval_acc(m5_ft, test_asr_loader)
print(f'FT  -> CA: {ft_ca_5:.2f}%, ASR: {ft_asr_5:.2f}%')

m5_anp = run_anp_lite(model_wanet_5, clean_def_loader, prune_ratio=0.15)
anp_ca_5, anp_asr_5 = eval_acc(m5_anp, test_clean_loader), eval_acc(m5_anp, test_asr_loader)
print(f'ANP -> CA: {anp_ca_5:.2f}%, ASR: {anp_asr_5:.2f}%')

print('\\nEvaluating defenses on WaNet 10%...')
m10_ft = run_ft(model_wanet_10, clean_def_loader)
ft_ca_10, ft_asr_10 = eval_acc(m10_ft, test_clean_loader), eval_acc(m10_ft, test_asr_loader)
print(f'FT  -> CA: {ft_ca_10:.2f}%, ASR: {ft_asr_10:.2f}%')

m10_anp = run_anp_lite(model_wanet_10, clean_def_loader, prune_ratio=0.15)
anp_ca_10, anp_asr_10 = eval_acc(m10_anp, test_clean_loader), eval_acc(m10_anp, test_asr_loader)
print(f'ANP -> CA: {anp_ca_10:.2f}%, ASR: {anp_asr_10:.2f}%')""")

# Cell 10: Export Summary
add_md("## Step 10: Summary & Export")
add_code("""summary = f\"\"\"
==============================================================
WANET GENERALIZATION EXPERIMENT RESULTS
==============================================================
WaNet 5%:
  Baseline: CA = {ca_5:.2f}%, ASR = {asr_5:.2f}%
  Detectors: AC Silhouette = {sil_5:.3f}, STRIP TPR = {tpr_5:.1f}%
  Controller Decision: {q_5} -> {def_5}
  Defenses:
    FT:  CA = {ft_ca_5:.2f}%, ASR = {ft_asr_5:.2f}%
    ANP: CA = {anp_ca_5:.2f}%, ASR = {anp_asr_5:.2f}%

WaNet 10%:
  Baseline: CA = {ca_10:.2f}%, ASR = {asr_10:.2f}%
  Detectors: AC Silhouette = {sil_10:.3f}, STRIP TPR = {tpr_10:.1f}%
  Controller Decision: {q_10} -> {def_10}
  Defenses:
    FT:  CA = {ft_ca_10:.2f}%, ASR = {ft_asr_10:.2f}%
    ANP: CA = {anp_ca_10:.2f}%, ASR = {anp_asr_10:.2f}%
==============================================================
\"\"\"
print(summary)
with open('wanet_results_summary.txt', 'w') as f:
    f.write(summary)
print('Done! Results written to wanet_results_summary.txt')""")

target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "wanet_experiments_colab.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated: {target_path}")
