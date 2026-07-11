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
| Blended | *(still TBD — see Stage C ablation results below)* | — | Global low-amplitude blend — variance likely spread across more directions; ablation data now exists (`blended_dr_comparison.csv`), decision not yet locked |
| Label-Consistent | PCA | 2 | Decided: LC uses the same visible 4×4 patch mechanism as BadNets (see `02_ATTACKS_AND_DATASETS.md`), so treated the same way. Revisit if Stage C data (silhouette/PDR) suggests otherwise — this was previously silently drifted to PCA-2D without being a deliberate, documented choice; it now is one. |

**Blended DR ablation data (run, not yet decided — see `10_BLENDED_STAGE_C_STATUS.md`):**

| pr_tag | DR method | silhouette | suspicious_fraction | PDR | recall |
|---|---|---|---|---|---|
| pr01 | PCA-2D | 0.3778 | 26.69% | 11.16% | 90.00% |
| pr01 | ICA-10D | 0.1102 | 9.09% | 100.00% | 100.00% |
| pr05 | PCA-2D | 0.5408 | 33.37% | 99.76% | 99.88% |
| pr05 | ICA-10D | 0.0983 | 33.33% | 100.00% | 100.00% |
| pr10 | PCA-2D | 0.6380 | 50.00% | 99.96% | 99.98% |
| pr10 | ICA-10D | 0.1744 | 49.99% | 99.96% | 100.00% |

PCA-2D gives cleaner monotonic silhouette growth with poison rate but weaker
PDR at pr01; ICA-10D gives much better PDR at pr01 but a less clean silhouette
trend. Not resolved here on purpose — pick one and document the reasoning
before this feeds controller calibration (Stage D).

### What AC does NOT output
- ❌ Attack identity (BadNets vs Blended vs LC)
- ❌ Trigger location
- ❌ Runtime/inference-time detection
- ❌ Guarantee of completeness

That's fine — don't claim those. The fact that AC is attack-agnostic is a **strength**, framed as:
> "In practice, defenders don't know the attack type" → **attack-agnostic severity-based remediation**

### Detection Scope (Be Honest About This)
- **AC will catch:** strong, fixed-pattern triggers (BadNets) — highly separable activations, high silhouette, clear suspicious fraction. **Also confirmed to catch Blended strongly at 5%/10% poison rate** (silhouette 0.54–0.64, PDR ~100% — see ablation table above), which complicates the "Blended evades AC" framing below; that framing still holds at low poison rates (1%) but not at 5–10%.
- **AC will struggle with:** label-consistent or semantic triggers, and Blended specifically **at low poison rates** — cluster separation drops; lean on auxiliary signals (confidence gap, Grad-CAM attention shift, STRIP entropy).

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
1. **Detects distributed/semantic triggers** — Blended patterns, semantic triggers, anything not isolated in one neuron group. AC may miss these; STRIP often won't. **(Confirmed true at 1%/5% poison rate for Blended — see taxonomy below. Not true at 10%.)**
2. **Verifies after unlearning** — AC cannot be rerun post-remediation in any meaningful training-time sense, but STRIP can be run directly on the final sanitized model. This answers: *"Did we actually remove the backdoor?"* This is system-level validation, not just detection.
3. **Catches AC false negatives** — builds the failure taxonomy:

| Case | AC | STRIP |
|------|----|----|
| Obvious patch (BadNets) | ✅ | ✅ |
| Blended (1%, 5% poison rate) | ⚠️ | ✅ |
| **Blended (10% poison rate)** | **✅✅ (silhouette 0.64, PDR 99.96%)** | **❌ (TPR 0.00%, confirmed — see note below)** |
| Label-Consistent | ❌ sometimes | ✅ often |

This taxonomy is genuinely useful experimental content for the paper.

### ⚠️ Confirmed finding: STRIP fails on Blended at 10% poison rate, while AC succeeds strongly

This was investigated directly (full detail in `10_BLENDED_STAGE_C_STATUS.md`)
and is confirmed, not an artifact. The verification chain:

1. **Ruled out mismatched conditions.** ASR and STRIP's TPR could have been
   apples-to-oranges (ASR measured at `alpha_test=0.5`, STRIP's trigger pool
   built at `alpha_test=0.1`). Re-evaluating ASR at the exact trigger strength
   STRIP tests (`alpha_test=0.1`) confirms **100% ASR under matched
   conditions** — ruling out "the pr10 backdoor happens to be weak here" as
   an explanation.
2. **Ruled out threshold-boundary noise.** Checked percentiles, not just
   min/max/mean, on the raw entropy arrays:

   | pr_tag | triggered min | triggered p10 | triggered median | % triggered below clean median |
   |---|---|---|---|---|
   | pr01 | 0.0329 | 0.0600 | 0.1272 | 94.0% |
   | pr05 | 0.0347 | 0.0654 | 0.1144 | 98.8% |
   | pr10 | **0.1392** | **0.4854** | **0.6806** | **9.6%** |

   At pr10, the *minimum* triggered-image entropy (0.1392) already sits above
   the detection threshold (0.1037) — no single sample could have been
   flagged. This is a full distributional inversion, not a few outliers
   dragging a mean.
3. **Ruled out general model drift.** Clean-image entropy is flat across
   pr01/05/10 (0.43 → 0.47 → 0.47) — the effect is isolated to the triggered
   condition, not the model's general calibration shifting.
4. **Ruled out AC also missing it.** AC on the same pr10 checkpoint gives
   silhouette=0.638, PDR=99.96% — AC is unambiguously strong here. This is
   specifically a STRIP failure mode, not "this checkpoint is hard to detect
   in general."
5. **Code-level check.** `run_strip`/`strip_entropy_single`/
   `add_blended_trigger_global` in `core/detection.py` and `core/attacks.py`
   inspected directly — no implementation bug, same `blend_pattern` object
   used throughout, correct threshold/TPR logic.

**This is not the previously-open "4th quadrant" question** (AC-low /
STRIP-low-entropy, see `03_CONTROLLER_LOGIC.md`) — it's a different pairing
(AC-high / STRIP-fails) that the existing 2×2 quadrant matrix already assigns
a defense to (`prune_finetune`, the "AC HIGH / STRIP HIGH-entropy" cell —
STRIP reads pr10 as if it were clean-or-weak, which happens to route to the
same practical action as if STRIP had correctly flagged it, purely because AC
alone is sufficient here). **Do not use this checkpoint's STRIP entropy as
verification of remediation success** — since STRIP is blind to this
condition pre-defense, it cannot be trusted to confirm removal post-defense
either without re-validating that STRIP's blind spot doesn't persist after
fine-tuning/pruning changes the model.

**Plausible (not yet confirmed) mechanism:** STRIP's own perturbation is
itself an alpha-blend of the same family as Blended's trigger. At high poison
rate the model may lock onto a narrow blend-intensity band; STRIP's own
overlay (default α=0.5) dilutes the baked-in trigger weight and, at pr10,
appears to break the shortcut outright rather than merely adding noise. This
mechanism is not required to state the finding above and is left as an
optional follow-up (see `10_BLENDED_STAGE_C_STATUS.md` §4) — do not write the
mechanism into the paper as fact until it's tested directly (e.g. a
`STRIP_ALPHA` sweep on the pr10 checkpoint).

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

**Addendum, reviewer-safe, for the Blended pr10 finding:**
> "We observe that STRIP's detection rate for the Blended attack is non-monotonic in poison rate: while STRIP reliably flags Blended backdoors at 1–5% poison rates, its true positive rate falls to 0% at 10% poison rate despite the underlying attack remaining fully effective (100% ASR) and strongly detected by Activation Clustering. We report this as a documented failure mode rather than a limitation of the overall pipeline, since Activation Clustering's severity signal is sufficient to route this condition correctly regardless."

---

## What to Log for Every Run (Detection + Verification Stage)

- `silhouette_score`
- `suspicious_fraction`
- `cluster_sizes` [C0, C1]
- `poison_detection_rate` (if ground-truth poison indices available, for internal validation only — not claimed as a detector output)
- `STRIP_entropy_before`, `STRIP_entropy_after`
- `TAR_before`, `TAR_after` (where defined)
- Grad-CAM image pairs saved to `gradcam/{attack}_{poison_pct}_{before|after}.png`

**Also log, per the Blended pr10 finding above:** the trigger-strength
(`alpha_test`) used to generate STRIP's triggered pool, and confirm it
matches the `alpha_test` used for the reported ASR figure it's being compared
against. Mismatched trigger strengths between the ASR baseline and the STRIP
input pool is an easy silent error — verify explicitly (see
`10_BLENDED_STAGE_C_STATUS.md` for the full check that caught this).
