# Detection & Verification — Activation Clustering, STRIP, Grad-CAM

---

## Mental Model (Lock This In)

> **AC analyzes representations → STRIP analyzes behavior → Grad-CAM visualizes attention.**

They do three different jobs. Don't try to make one substitute for another.

---

## 1. Activation Clustering (AC) — Severity Estimator

### Inputs
- Untrusted training dataset `D_train = {(x_i, y_i)}`
- Trained model `f_θ`
- Hidden-layer activations, usually the **last hidden layer** (penultimate, before final FC): `a_i = f_θ^(L-1)(x_i)`
- Activations grouped per class (predicted or labeled class)

AC is a **training-time** analysis only.

### Process
1. Extract penultimate-layer activations for all samples in the target class
2. Apply dimensionality reduction per attack (see "Per-attack DR choice" below)
3. Run K-Means with `k=2` per class
4. Compute silhouette score and cluster sizes

### Outputs

| Output | Meaning | Use |
|--------|---------|-----|
| Cluster labels | clean vs suspicious samples | estimate poisoned subset |
| Poison ratio (`suspicious_fraction`) | % of suspicious points in class | severity signal |
| Separation score (silhouette) | cluster tightness/separation | confidence of poisoning |
| Optional visual summary | averaged images / sprites | paper figure |

Compress into one number — the **detection severity score d**:
```
d = 0.12  (≈12% of class clustered as suspicious, silhouette = 0.18)
```

### `suspicious_fraction` — must be label-free

`suspicious_fraction` is meant to be a **detector output**, computed with no access to ground-truth poison labels:

```
suspicious_fraction = min(cluster_sizes) / len(Xc)
```

i.e. the size of the *smaller* of the two K-Means clusters, as a fraction of the total target-class sample count. This is deliberately independent of which cluster actually contains the poison — the detector has no way to know that at inference/audit time.

This is a **separate concept** from `PDR` (poison detection rate) and `recall`, which *are* computed using ground-truth `poison_idx` and are correctly scoped as internal-only validation metrics (used to check whether the detector's flagged cluster is actually right, not something the pipeline could compute without ground truth). Do not conflate the two: `suspicious_fraction` is what a deployed detector reports; `PDR`/`recall` are what we use internally to know if it's working.

### Per-attack DR choice (decide once, document here)

Doc and code have drifted: this doc previously said PCA with 10–50 components generically, while the shared `run_ac` implementation defaults to ICA-10D and takes a per-attack `use_pca` flag (PCA-2D for BadNets). Once the Blended/LC ablations referenced in `02_ATTACKS_AND_DATASETS.md` are run, state the final per-attack choice explicitly here instead of leaving it as an unstated default in code, e.g.:

| Attack | DR method | Components | Reasoning |
|--------|-----------|------------|-----------|
| BadNets | PCA | 2 | Hard, fixed-location patch → strong variance in few directions |
| Blended | *(TBD — ablate ICA-10D vs PCA-2D)* | — | Global low-amplitude blend — variance likely spread across more directions; needs empirical ablation, don't assume ICA by default |
| Label-Consistent | PCA | 2 | Decided: LC uses the same visible 4×4 patch mechanism as BadNets (see `02_ATTACKS_AND_DATASETS.md`), so treated the same way. Revisit if Stage C data (silhouette/PDR) suggests otherwise — this was previously silently drifted to PCA-2D without being a deliberate, documented choice; it now is one. |

### What AC does NOT output
- ❌ Attack identity (BadNets vs Blended vs LC)
- ❌ Trigger location
- ❌ Runtime/inference-time detection
- ❌ Guarantee of completeness

That's fine — don't claim those. The fact that AC is attack-agnostic is a **strength**, framed as:
> "In practice, defenders don't know the attack type" → **attack-agnostic severity-based remediation**

### Detection Scope (Be Honest About This)
- **AC will catch:** strong, fixed-pattern triggers (BadNets) — highly separable activations, high silhouette, clear suspicious fraction
- **AC will struggle with:** Blended, label-consistent, or semantic triggers — cluster separation drops; lean on auxiliary signals (confidence gap, Grad-CAM attention shift, STRIP entropy)

---

## 2. STRIP — Behavioral Verifier

**STRIP = STRong Intentional Perturbation**

An **inference-time entropy test.**

### How it works
For a given input image `x`:
1. Create many perturbed versions (overlay random images / noise from a held-out clean set)
2. Run the model on each perturbed version
3. Measure prediction entropy across the perturbed batch

### Behavior

| Input Type | Model Behavior | Entropy |
|------------|----------------|---------|
| Clean image | Predictions vary across perturbations | **High entropy** |
| Backdoored image | Prediction stays locked to target class | **Low entropy** |

Why: backdoors create shortcut features that dominate the prediction regardless of perturbation.

### Inputs
- Deployed/test model `f_θ`
- Input image `x` under test
- Random overlay images from a held-out set

### Outputs

| Output | Meaning | Use |
|--------|---------|-----|
| Entropy score H | prediction stability under perturbation | backdoor flag (H < threshold → suspicious) |
| Binary flag | backdoored / clean | per-sample decision |
| Entropy histogram | distribution over test set | paper figure |

### What STRIP gives that AC cannot
1. **Detects distributed/semantic triggers** — Blended patterns, semantic triggers, anything not isolated in one neuron group. AC may miss these; STRIP often won't.
2. **Verifies after unlearning** — AC cannot be rerun post-remediation in any meaningful training-time sense, but STRIP can be run directly on the final sanitized model. This answers: *"Did we actually remove the backdoor?"* This is system-level validation, not just detection.
3. **Catches AC false negatives** — builds the failure taxonomy:

| Case | AC | STRIP |
|------|----|----|
| Obvious patch (BadNets) | ✅ | ✅ |
| Blended | ⚠️ | ✅ |
| Label-Consistent | ❌ sometimes | ✅ often |

This taxonomy is genuinely useful experimental content for the paper.

---

## 3. Grad-CAM — Visualization & TAR

### What it does
Produces class-activation heatmaps showing which image regions the model relies on for a prediction.

### Use in this pipeline
1. **Qualitative figures:** show the model's attention on a triggered image before defense (focused on trigger) vs after defense (focused on actual object)
2. **Trigger Attention Ratio (TAR):** a quantitative metric measuring how much attention the model places on the known trigger region — should decrease significantly after successful remediation

### Procedure
- Pick a small set of test images (clean + triggered) per attack/poison-rate combination
- Generate Grad-CAM heatmaps pre- and post-defense
- For BadNets and Label-Consistent, the trigger location is known (both use the same fixed 4×4 patch position) — directly compute TAR as the fraction of heatmap energy inside the patch bounding box for both attacks
- For Blended, TAR is harder to define precisely since there's no fixed patch location — consider reporting qualitative comparisons only, or defining TAR relative to the blend mask if available

---

## Combined Detection Pipeline Diagram

```
AC → severity score (d) → controller → unlearning
                                            ↓
                                    STRIP verification
                                            ↓
                                    Grad-CAM / TAR (figures)
```

---

## Verification Reviewer-Safe Sentence

> "Activation Clustering provides an attack-agnostic severity signal based on representation separation, while STRIP serves as a post-remediation behavioral verification to ensure the absence of shortcut-triggered predictions."

---

## What to Log for Every Run (Detection + Verification Stage)

- `silhouette_score`
- `suspicious_fraction`
- `cluster_sizes` [C0, C1]
- `poison_detection_rate` (if ground-truth poison indices available, for internal validation only — not claimed as a detector output)
- `STRIP_entropy_before`, `STRIP_entropy_after`
- `TAR_before`, `TAR_after` (where defined)
- Grad-CAM image pairs saved to `gradcam/{attack}_{poison_pct}_{before|after}.png`
