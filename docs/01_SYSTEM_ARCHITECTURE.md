# System Architecture
## Adaptive Hybrid Machine Unlearning for Backdoor Defense

---

## Full Pipeline Overview

```
==============================================================================================================================
                                               CORE EXECUTION PIPELINE
==============================================================================================================================

  [ Attack Input Module ]
  (BadNets / Blended / LC)
         │
         ▼
  [ ResNet-18 Poisoned Checkpoint ]
         │
         ▼
  [ Detection: Activation Clustering ]       ────────► Penultimate-layer activations per class
  (K-Means k=2, PCA reduction)                        Silhouette score, cluster sizes,
         │                                            suspicious_fraction → severity score d
         ▼
  [ Adaptive Hybrid Controller ]             ────────► Maps (d, STRIP entropy H) → defense action
  (2×2 Decision Matrix)                               via rule-based calibrated thresholds
         │
         ▼
  [ Selective Unlearning Engine ]
  (Fine-tune / Prune / NAD / BAERASER-lite)
         │
         ▼
  [ Evaluation & Verification Module ]       ────────► CA_after, ASR_after, STRIP entropy distributions,
  (STRIP + Grad-CAM + TAR)                            Grad-CAM heatmaps, Trigger Attention Ratio
         │
         ▼
  [ Post-Deployment Monitoring ]             ────────► Stream STL-10 + Custom dataset
  (Continuous STRIP + suspicious buffer)              Flag low-entropy inputs
         │                                            Periodic fine-tune update
         ▼
  [ Audit Report Output ]                    ────────► JSON/CSV: all metrics, decision logs, checkpoints

==============================================================================================================================
                                             ADAPTIVE INTELLIGENCE LAYER
==============================================================================================================================

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  [ Attack Fingerprint Extractor ]                                            │
  │  Calculates silhouette scores, cluster sizes, entropy signatures             │
  │                                                                              │
  │  [ Fingerprint Database Registry ]                                           │
  │  Indexes profiles, runs cosine-similarity matches, serves cached defense     │
  │                                                                              │
  │  [ Continual Learner Module ]  (Phase-2)                                     │
  │  Incremental parameter updates, suspicious sample streaming buffers          │
  └──────────────────────────────────────────────────────────────────────────────┘

==============================================================================================================================
                                           SUPPORT SYSTEMS & INFRASTRUCTURE
==============================================================================================================================

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  CustomDatasetProcessor  → Load, center-crop, LANCZOS resize to 32×32       │
  │  CrossDatasetEvaluator   → Multi-domain evaluation (CIFAR / STL / Custom)   │
  │  IndexManager            → Lock seeds, poisoned sample indices               │
  │  MetricsStandardizer     → CA, ASR, TAR — single shared implementation      │
  │  Logger & GitManager     → Metadata tracking, strict repo compliance         │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module Technical Specifications

### 1. DataLoader Module
- **Inputs:** Raw CIFAR-10, STL-10, or custom images
- **Outputs:** Clean split, poisoned split, defense subset (2,500 fixed images)
- **Critical:** Must use `defense_indices.npy` for the clean budget. All members share this file.
- **Transforms:**
  ```python
  transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
  ])
  ```

### 2. Attack Module
- **BadNets:** Inject a 4×4 white pixel patch at bottom-right corner of target-class images. Simple, visually obvious.
- **Blended:** Alpha-blend a fixed pattern (e.g., "Hello Kitty" image or noise pattern) at α=0.1. Semi-transparent, affects global appearance.
- **Label-Consistent (LC):** Semantic trigger — no pixel modification. Achieve by generating adversarial perturbations on clean images so that they naturally cluster near a target class in representation space, but the label stays correct. Hardest to detect.
- **Output:** Poisoned dataset splits at 1%, 5%, 10% poison rates. Save as separate splits.
- **Checkpoint naming:** `badnets_5pct_seed2025_poisoned_train.pth`

### 3. Model Training Module
- **Architecture:** ResNet-18 only (fixed)
- **Dataset:** CIFAR-10 poisoned training split
- **Hyper-params (starting point):**
  - Epochs: 100 (full), 30 (quick iteration)
  - LR: 0.01, SGD with momentum 0.9, weight decay 1e-4
  - Batch size: 128
- **Outputs:** Model checkpoints + activation `.npy` files from penultimate layer
- **Critical:** Save clean baseline first (`resnet18_clean_seed2025.pth`), then per-attack poisoned checkpoints

### 4. Detection Module — Activation Clustering (AC)
- **Input:** Penultimate-layer activations per class (extracted from poisoned model on training data)
- **Process:**
  1. Extract activations from layer before final FC (`layer4` output for ResNet-18)
  2. Apply PCA to reduce to 10-50 components
  3. Run K-Means with `k=2` per target class
  4. Compute silhouette score and cluster sizes
- **Outputs (all logged to CSV):**
  - `silhouette_score` (float, key metric)
  - `suspicious_fraction` = min(cluster_size) / total_class_samples
  - `cluster_sizes` = [C0, C1] (size of each cluster)
  - `poison_detection_rate (PDR)` = fraction of true poisoned samples in smaller cluster
  - **Combined severity score d** (see Controller doc)
- **What AC CAN say:** "Structurally embedded poisoning exists / its approximate severity"
- **What AC CANNOT say:** Which attack it is, where the trigger is, inference-time behavior

### 5. Adaptive Hybrid Controller
- See `03_CONTROLLER_LOGIC.md` for full spec
- **One-line:** Rule-based, auditable, calibrated offline — not learned

### 6. Selective Unlearning Engine
- See `04_DEFENSE_METHODS.md` for full spec
- **Three methods (+ optional 4th):**
  1. Fine-tuning (light) — for low-severity
  2. Neuron Pruning + Fine-tuning — for medium-severity
  3. BAERASER-lite — for high-severity structural embed + behavioral dominance
  4. NAD (Neural Attention Distillation) — for stealthy/semantic triggers (AC-miss, STRIP-catch)

### 7. Evaluation & Verification Module
- **STRIP:** Run on sanitized model to verify backdoor behavior is gone
- **Grad-CAM:** Generate heatmaps on triggered images before/after defense
- **TAR (Trigger Attention Ratio):** Quantify how much attention model places on trigger region (should decrease post-defense)
- **Outputs:** `CA_after`, `ASR_after`, STRIP entropy histograms, Grad-CAM side-by-side images
- **CSV output:** All per-run results to `results/experiment_log.csv`

### 8. Deployment & Monitoring Module (Phase-2)
- **Input stream:** Shuffled mix of STL-10 + custom images (clean and triggered)
- **Per-image:** Run prediction + STRIP entropy
- **Flag:** If entropy < threshold → mark suspicious, add to `suspicious_buffer/`
- **Periodic update:** Every N images, fine-tune on `suspicious_buffer` + `clean_memory_buffer`
- **Track:** CA and ASR after each update cycle
- **Critical:** Keep a `clean_memory_buffer` (≈500 CIFAR images) to prevent catastrophic forgetting

---

## Data Preprocessing Pipeline (STL-10 and Custom)

```
[ Raw Input Image ]
       │
       ▼
[ Center Crop ] — scale down along longest dimension
       │
       ▼
[ RGB Normalization ] — force 3-channel (handle grayscale/RGBA)
       │
       ▼
[ LANCZOS Resize ] — resize to 32×32 using PIL LANCZOS
       │
       ▼
[ CIFAR-10 Normalization ] — μ = [0.4914, 0.4822, 0.4465], σ = [0.2023, 0.1994, 0.2010]
```

---

## Recommended Repository Structure

```
/
├── core/                            ← SHARED scripts (everyone imports from here)
│   ├── models.py                    ← ResNet-18 definition (single source)
│   ├── data_utils.py                ← get_poisoned_loader(), get_clean_subset()
│   ├── metrics.py                   ← calculate_asr(), calculate_ca() — single impl
│   └── seed_utils.py                ← set_seed(2025)
│
├── attacks/
│   ├── badnets.py
│   ├── blended.py
│   └── label_consistent.py
│
├── detection/
│   ├── ac_detect.py                 ← Activation Clustering + severity score
│   └── strip.py                     ← STRIP entropy evaluation
│
├── controller/
│   └── controller.py                ← Adaptive decision logic + threshold config
│
├── defenses/
│   ├── finetune.py
│   ├── pruning.py
│   ├── nad.py
│   └── baeraser_lite.py
│
├── verification/
│   ├── strip_verify.py
│   └── gradcam.py                   ← Grad-CAM + TAR computation
│
├── deployment/
│   └── monitor.py                   ← Post-deployment streaming STRIP monitor
│
├── scripts/
│   ├── scan_model.py                ← Main CLI entrypoint
│   ├── run_phase1.sh
│   └── run_one_condition.py         ← Smoke test (CI guardrail)
│
├── notebooks/
│   ├── 01_badnets_experiment.ipynb
│   ├── 02_blended_experiment.ipynb
│   └── 03_lc_experiment.ipynb
│
├── results/
│   └── experiment_log.csv
│
├── checkpoints/                     ← Hosted on Drive; linked in README
├── activations/                     ← Cached .npy penultimate layer activations
├── defense_indices.npy              ← SHARED — same 2500 images for everyone
├── poison_indices.npy               ← SHARED — same poison indices
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Deep learning | PyTorch + Torchvision |
| Clustering / PCA | scikit-learn |
| Array ops | NumPy |
| Visualization | Matplotlib |
| Explainability | Native PyTorch Grad-CAM (manual or `pytorch-grad-cam` lib) |
| Environment | Google Colab + GPU |
| Experiment tracking | Google Sheets (shared) + CSV logs |

---

## Activation Caching Strategy

Cache penultimate-layer activations immediately after training. Never re-run model forward passes just for AC analysis.

```python
# Save after training
acts = extract_penultimate(model, dataloader)  # shape: (N, feature_dim)
np.save(f"activations/{attack}_{poison_rate}_{seed}.npy", acts)

# Load for AC
acts = np.load(f"activations/{attack}_{poison_rate}_{seed}.npy")
```

Naming: `activations/{attack_type}_{poison_pct}pct_seed{seed}.npy`
