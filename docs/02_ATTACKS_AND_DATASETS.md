# Attacks & Datasets — Detailed Specification

---

## 1. Attacks (Tier-1, Locked Scope)

### BadNets
- **Trigger:** 4×4 white pixel patch, fixed location (bottom-right corner)
- **Mechanism:** Patch pasted onto a fraction of target-class training images; label forced to target class (class 0, airplane)
- **Expected behavior:** Strong, localized, easily separable in activation space → high AC silhouette
- **Severity regime:** Low–Medium
- **From observed runs (treat as illustrative only — experiments are being redone):**
  prior silhouette scores trended upward with poison rate (~0.50 at 1% poison, ~0.53 at 10%), and ASR rose sharply with poison rate. Do not reuse these numbers in the new paper; they are reference points only, not final results.

### Blended
- **Trigger:** Global alpha-blended overlay pattern (α ≈ 0.1) — e.g., a fixed noise pattern or image blended faintly across the whole frame
- **Mechanism:** Semi-transparent, not localized; affects global image statistics rather than a small region
- **Expected behavior:** Evades simple cluster separation — clean and poisoned activations tend to overlap, producing low silhouette scores even though ASR can be high. This is exactly why STRIP and the multi-signal controller matter.
- **Severity regime:** Medium
- **Prior observation (illustrative only, re-verify):** silhouette dropped substantially from BadNets levels at the same poison rate, while ASR still rose with poison rate. Confirms Blended is a stealthier trigger that AC alone under-detects.

### Label-Consistent (LC)
- **Trigger:** Visible 4×4 patch, same as BadNets, applied only to target-class images. What makes LC different from BadNets is not the trigger itself but a PGD suppression step applied to the underlying image *before* the patch is added.
- **Mechanism:** Following Turner, Tsipras & Madry's original Label-Consistent Backdoor Attacks formulation (and consistent with the "Silent Killer" paper's framing of clean-label crafting via gradient-based optimization — Lederer, Maimon & Rokach, 2023), PGD is used to perturb target-class images *away* from their own class's natural/easily-learned features (bounded by l∞ ≤ ε), not to pull them toward the target class. This is the opposite of "cluster near the target class." By making the clean features harder for the model to rely on, the model is forced to associate the patch itself with the (already-correct) target label, so the patch becomes a reliable trigger at inference time even though every poisoned sample keeps its true label. No label change is required, which is what makes LC stealthy against simple data audits — but the pixels themselves are modified in two ways: the patch (visible) and the PGD perturbation (subtle).
- **Expected behavior:** Open question, not an assumption. It is plausible that AC fails to find clean separation because feature suppression pushes activations together — but it's equally plausible that forcing strong reliance on a fixed, visible patch makes LC *more* separable in activation space than Blended (whose trigger is intentionally low-amplitude and global). Which of these holds should be settled empirically from Stage C data, not pre-committed to a "4th quadrant" narrative (see `03_CONTROLLER_LOGIC.md`).
- **Severity regime:** High (tentative — hardest to catch by manual inspection due to label-consistency, not necessarily because ASR or AC-evasion is highest; confirm from data)
- **Action item:** This attack's AC/STRIP signal pairing is the open empirical question. Confirm from Stage C results whether it lands in the "AC-low, STRIP-high" quadrant before deciding whether to add NAD as a 4th defense regime — do not assume this in advance.
- **Poison-rate note:** LC poisoning is restricted to target-class images only, so the poison budget is capped at 10% of the *full* training set — which corresponds to ~100% of the target class. The 1%/5%/10% rates therefore don't mean the same thing for LC as they do for BadNets/Blended (see `00_PROJECT_MASTER.md`).
- **PGD/patch hyperparameters (decided):** ε = 16/255, α = 2/255, 7 PGD steps, 4×4 patch, patch opacity 0.25.

### Optional Tier-2 (add at most one if time permits)
- **WaNet** — warping-based trigger, harder to detect, AC may struggle
- **Dynamic triggers** — expensive, weaker detection signal

### Explicitly Out of Scope
- TrojanNN-style reverse engineering attacks
- CLIP / multimodal backdoors
- NLP backdoors
- Adaptive white-box attacks (these are "reviewer traps" — high effort, low payoff for this project's scope)

### Phase-2 Advanced Threats (only after base pipeline works)
These were explored in Set 3 as responses to reviewer feedback about dataset/threat diversity, and are good candidates once the core 3-attack pipeline is solid:

- **Frequency-Domain Backdoor:** FFT/DCT transform → modify high-frequency coefficients → inverse transform. Invisible to spatial inspection and defeats patch-localization defenses.
- **Steganographic Backdoor:** Low-amplitude sinusoidal/noise watermark blended at very low opacity (≤2–3 intensity levels change). Visually indistinguishable from clean images; stress-tests STRIP and any Neural-Cleanse-style defense.

These two attacks are good "hard mode" additions for a stronger Phase-2 story, but should only be attempted after BadNets/Blended/LC pipeline + controller + defenses are fully validated and re-run cleanly.

---

## 2. Datasets

### CIFAR-10 (Training + Baseline)
- 50,000 train / 10,000 test images, 32×32 RGB, 10 balanced classes
- All training happens here exclusively
- Target class for poisoning: **Class 0 (airplane)** — fixed across the whole project

### STL-10 (Cross-Dataset OOD Evaluation)
- 96×96 RGB, downsampled to 32×32 for pipeline compatibility
- 9 overlapping classes with CIFAR-10 (Frog is replaced by Monkey — STL-10 has no frog class)
- Used **only** for evaluation of sanitized checkpoints — no retraining
- Purpose: directly answers the reviewer criticism "your method may only work on CIFAR" — this is a cross-dataset robustness claim, not a "newer photos" claim
- Measure: Clean Accuracy (CA) and STRIP false-alarm rate on STL-10 inputs run through CIFAR-trained (and sanitized) models

### Custom Dataset (~800 images, Real-World Validation)
- Team-collected real photographs across the 10 CIFAR classes
- Roughly 50–100 images per class is sufficient (don't over-collect; papers commonly use 50–300 total)
- **Never used for training** — only for:
  1. Out-of-distribution (OOD) backdoor trigger test (does a CIFAR-trained backdoor transfer to real photos?)
  2. STRIP verification on real images
  3. Grad-CAM visualization on real images
  4. Post-deployment monitoring stream simulation
- Collection tip: favor "hard" images (odd angles, varied lighting, motion blur, clutter) over clean centered shots — this is what demonstrates real-world fragility/robustness of triggers, which is the actually interesting result
- Preprocessing: center-crop → RGB normalize → LANCZOS resize to 32×32 → CIFAR-10 mean/std normalize

### GTSRB (Optional, Safety-Critical Bonus)
- German Traffic Sign Recognition Benchmark, 43 classes
- Add only if a clean week remains after CIFAR-10 + STL-10 + Custom are fully done
- If added: single transfer experiment only — train on CIFAR-10, run sanitized model on GTSRB, report CA and STRIP behavior (no full retraining loop needed)
- Strong because "sticker on stop sign → misclassified as speed limit" is the most-cited real-world backdoor example in the literature — reviewers recognize it instantly
- Preprocessing overhead (resizing, class imbalance) is roughly a day's work — budget accordingly

### Explicitly Skipped
- **CIFAR-100** — decided against. Reasoning: project is built around single-target-class attacks (class 0); CIFAR-100 has only 600 images/class, making ASR numbers noisy and hard to reproduce. STL-10 + Custom already cover cross-dataset and real-world needs at lower cost.

---

## 3. Dataset Pipeline Summary

```
Train          → CIFAR-10 only
Test baseline  → CIFAR-10 test set (10,000 images)
Cross-dataset  → STL-10 (sanitized checkpoints, no retraining)
Real-world     → Custom dataset (~800 images, OOD + STRIP + Grad-CAM + deployment stream)
Optional       → GTSRB (single transfer experiment, safety narrative)
Skipped        → CIFAR-100
```

### Paper framing sentence (ready to use)
> "Experiments were conducted across CIFAR-10 (training and baseline evaluation), STL-10 (cross-dataset robustness), and a custom real-world dataset collected by the authors for out-of-distribution and post-deployment validation, with GTSRB included as an optional safety-critical transfer case."
