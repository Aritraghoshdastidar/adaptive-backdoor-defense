# Adaptive Hybrid Machine Unlearning for Backdoor Defense
## Project Master Context — PW26_NVP_01

**Team:** Archita (AM063), Aritra (AM064), Tarun (CS642), Bhargavi (AM077)
**Guide:** Prof. Nitin V Pujari | **Institution:** PES University
**Status:** Phase-1 experiments being redone. Proceed from clean implementation.

---

## What This Project Is

A **modular, adaptive defense pipeline** that detects and mitigates backdoor (Trojan) attacks in deep learning image classifiers — without requiring full model retraining.

The pipeline:
1. Detects whether a model is poisoned and estimates **how severely**
2. Makes an intelligent, auditable **decision about which defense to use**
3. Applies that defense (**unlearns** the backdoor)
4. **Verifies** that the backdoor has been removed

### The Core Research Claim (Memorize This)

> "We formulate backdoor remediation as a cost-constrained decision problem and empirically show severity regimes where lightweight defenses suffice, achieving comparable security at significantly reduced compute cost."

This is a **system/orchestration paper**, not an algorithm paper. The novelty is the **adaptive decision logic and integrated evaluation**, not any single new component.

---

## Why This Matters (Industry Framing)

Frame as: **"Pre-deployment Model Safety Scanner"**

Key use cases:
- Company downloads a pretrained CNN for traffic sign recognition → unknown if poisoned → run this pipeline
- MLaaS provider needs to guarantee hosted models are safe
- Autonomous driving / medical imaging: a sticker on a stop sign can cause a misclassification → real harm

In the current (2026) landscape:
- EU AI Act mandates explainability and remediability of model failures (Grad-CAM directly addresses this)
- Models trained on massive uncurated web data → rising data poisoning risk
- Retraining a large model costs millions → adaptive lightweight remediation saves cost

---

## The Paper Contributions (What to Claim)

1. **Formal decision formulation** — Backdoor remediation as constrained optimization over security, utility, and compute
2. **Data-calibrated policy controller** — Thresholds calibrated offline; logic fixed and auditable
3. **Severity regime discovery** — Empirical phase transitions across attacks and poison rates
4. **Cost-aware evaluation** — Runtime, GPU-hours, accuracy tradeoffs (almost no one reports this)
5. **End-to-end deployable pipeline** — Detection → remediation → verification → post-deployment monitoring

### What NOT to Claim

- ❌ "Fully removes all backdoors"
- ❌ "Works for all attack types"
- ❌ "Automatic for semantic triggers"
- ✅ "Adaptive mitigation works best for strong and medium backdoors"
- ✅ "Stealthy attacks remain challenging — we document these as failure modes"

---

## The Attacks (Fixed — Do Not Change)

| Attack | Trigger Type | Severity Regime | Controller Response |
|--------|-------------|-----------------|-------------------|
| **BadNets** | 4×4 pixel patch (visible) | Low–Medium | Fine-tuning |
| **Blended** | Global semi-transparent overlay (α=0.1) | Medium | Pruning / NAD |
| **Label-Consistent (LC)** | Semantic, no pixel modification | High / Stealthy | BAERASER / NAD |

**Poison rates to test:** 1%, 5%, 10%
**Target class:** Class 0 (Airplane) — fixed across all experiments
**Model:** ResNet-18 only

**Reviewer-safe sentence:**
> "We select BadNets, Blended, and Label-Consistent attacks to represent increasing trigger subtlety and semantic realism, enabling systematic evaluation of severity-aware remediation regimes."

---

## The Pipeline (One-Line Per Stage)

```
Attack Generation → Train Poisoned Model → Activation Clustering (Detection)
→ Adaptive Hybrid Controller (Severity Scoring) → Selective Unlearning Engine
→ STRIP + Grad-CAM Verification → Post-Deployment Monitoring (STL-10 + Custom)
```

---

## Key Distinctions (Lock These In)

| Concept | Role | When Used |
|---------|------|-----------|
| **AC (Activation Clustering)** | Severity estimator — "how structurally embedded is the poison?" | Training-time |
| **STRIP** | Behavioral verifier — "does backdoor behavior still exist?" | Inference-time, post-remediation |
| **Grad-CAM** | Visualization only — "show where the model is looking" | Paper figures, TAR metric |

**Gold sentence:**
> "Activation Clustering provides an attack-agnostic severity signal based on representation separation, while STRIP serves as a post-remediation behavioral verification to ensure the absence of shortcut-triggered predictions."

---

## Datasets (Fixed)

| Dataset | Role | Notes |
|---------|------|-------|
| **CIFAR-10** | Train + baseline evaluation | 50k train / 10k test, 10 classes, 32×32 |
| **STL-10** | Cross-dataset OOD evaluation | 9 overlapping classes (Frog→Monkey), 96×96 → resize to 32×32, 8k test |
| **Custom (~800 images)** | Real-world deployment simulation | Collected by team, 10 classes, used only for test/validation |
| **GTSRB** | Safety-critical optional | Only if time permits after CIFAR-10 + STL-10 complete |

**Never mix custom or STL-10 into training. Training = CIFAR-10 only.**

---

## Reproducibility Non-Negotiables

- Global seed: `2025` — everywhere, no exceptions
- Shared `defense_indices.npy` — same 2,500 images (5% of 50k) for all defenses
- Shared `poison_indices.npy` — same poison injection indices
- Target class: `0` (airplane)
- Clean test set: full CIFAR-10 test set (10,000 images)
- Checkpoint naming: `[attack]_[poison_rate]_[seed].pth`

---

## Publication Targets

**Tier 1 (Workshop — submit first for quick feedback):**
- AISec (ACM Workshop on AI & Security)
- NeurIPS / ICLR / ICML Workshops (Backdoor / Safety / "BUGS")

**Tier 2 (Security conferences):**
- USENIX Security, IEEE S&P (Oakland), ACM CCS, NDSS

**Tier 3 (ML/Vision):**
- NeurIPS, ICML, ICLR, CVPR (only if strong novel component added)

---

## Project Title

> **"Adaptive Hybrid Machine Unlearning for Backdoor Defense"**

Alternative:
> **"Model Safety Scanner: A Cost-Aware, Severity-Calibrated Pipeline for Automated Backdoor Detection and Remediation"**
