# Adaptive Hybrid Controller — Decision Logic Specification

---

## Core Principle

> **Optimization-inspired, rule-based controller with empirically calibrated thresholds.**

This is the deliberate, locked design choice. Do not change it without strong reason.

| Property | Why It's the Right Choice |
|----------|---------------------------|
| Deterministic | Reviewers and interviewers trust it more than a black box |
| Auditable | Industry-friendly — every decision is explainable in one sentence |
| Calibrated, not learned | Avoids overfitting to a narrow threat distribution; avoids RL/neural-policy complexity that the project doesn't need |

**Explicitly do NOT use:** reinforcement learning, a learned neural policy, or online/continuous adaptation of the controller logic itself. (Continual *model* updates post-deployment are fine — that's a different, supported feature — but the *controller's decision rules* stay fixed and calibrated offline.)

---

## State Vector

```
s = (d, a, u, c)
```

- **d** — AC severity score (cluster size ratio + silhouette score, combined)
- **a** — estimated ASR (from trigger test / proxy)
- **u** — utility loss proxy (Δ clean accuracy after light fine-tune probe)
- **c** — compute cost (epochs × FLOPs, or wall-clock proxy)

## Objective (Formal, for the Paper)

```
min   a(s) + λ·u(s) + μ·c(s)
 π
```

You are **not learning π**. You are **calibrating operating points** (the thresholds) via offline validation.

**Reviewer-safe wording:**
> "Thresholds are empirically calibrated via offline validation to minimize ASR + utility loss under compute constraints."

---

## Two Versions of the Controller — Use the Richer One

### V1 — Single-Signal (AC only) — Simpler Starting Point

```python
if d < τ1:
    action = "fine_tune"
elif τ1 <= d < τ2:
    action = "prune_finetune"
else:
    action = "baeraser"
```

Mapping discovered in early planning:

| AC Severity (d) | Dominant Attack (example) | Best Defense |
|---|---|---|
| Low | BadNets | Fine-tuning |
| Medium | Blended | Pruning / FT |
| High | Label-Consistent | BAERASER |

### V2 — Two-Signal (AC + STRIP) — Stronger, Preferred Version

This is the more publishable version developed in Set 2 of the project docs, and should be the **target design** for the redone experiments.

**Why two signals matter:** AC tells you how structurally embedded the poison is in representation space. STRIP tells you how behaviorally dominant the backdoor is at inference time. These two signals **don't always agree**, and the disagreement itself is informative.

### 2×2 Decision Quadrant Matrix

| AC Severity \ STRIP Entropy | **LOW entropy** (strong behavioral dominance) | **HIGH entropy** (weak behavioral dominance) |
|---|---|---|
| **AC HIGH** (structurally embedded) | → **BAERASER** (full unlearning — both signals fire) | → **Prune / Fine-tune** (structural fix sufficient) |
| **AC LOW** (stealthy in representation space) | → **NAD / Distillation** (behavioral fix needed; representation-agnostic) | → **Fine-tune / flag clean** (weak or no poison signal) |

### Quadrant Reasoning

- **AC high + STRIP low-entropy (locked):** Full backdoor compromise — both diagnostics fire → heavy unlearning (BAERASER) required.
- **AC high + STRIP high-entropy:** Structurally embedded but not yet behaviorally dominant — neuron pruning targeting the malicious activation zones is sufficient.
- **AC low + STRIP low-entropy:** AC missed it (stealthy trigger), but STRIP caught the behavioral lock-in. Needs NAD/distillation — a behavioral, representation-agnostic fix — because pruning doesn't know which neurons to target if AC never found them. Which attack(s) actually land here is an open empirical question, not a pre-committed assumption — see the note below.
- **AC low + STRIP high-entropy:** Either a genuinely clean model or very weak poisoning — fine-tune lightly or flag as clean.

---

## IMPORTANT: This 4th Quadrant Must Be *Earned*, Not Assumed

Adding NAD as a fourth defense is only justified if your experiments **actually produce** the AC-low/STRIP-low-entropy quadrant. This is an open question, not an assumption. Label-Consistent was previously flagged as the most likely candidate for this quadrant, but that framing pre-dated a correction to LC's mechanism: LC's PGD step forces the model to rely heavily on a fixed, visible patch (the same patch type BadNets uses), which could plausibly make LC's activations *more* separable in representation space, not less — the opposite of what the "AC-low" prediction assumes. Do not pre-commit to the 4th-quadrant narrative or to LC being the quadrant's occupant; decide purely from the actual Stage C AC/STRIP data.

**Action item for the redo:**
1. Run AC + STRIP on all three attacks × all three poison rates
2. Plot AC severity (x-axis) vs STRIP entropy (y-axis) for every condition
3. If a natural cluster appears in the AC-low/STRIP-low-entropy region → confirm 4 regimes, add NAD
4. If it doesn't naturally appear → keep the simpler 3-regime story (V1 controller) and document the absence as a finding, not a failure. Do not force a quadrant that the data doesn't show.

---

## Threshold Calibration Reference Logic (Starter)

```python
# 1. Compute inputs
sil = AC_silhouette(target_class_acts)
sus_frac = min(cluster_size) / total_class_samples
conf_gap = mean_confidence(clean_preds) - mean_confidence(triggered_preds)

# 2. Score mapping (initial weights — recalibrate empirically)
poison_score = (
    0.5 * sigmoid((sil - 0.1) / 0.2)
    + 0.3 * sigmoid((sus_frac - 0.02) / 0.05)
    + 0.2 * sigmoid(conf_gap / 0.2)
)

# 3. Rule execution tiers
if poison_score >= 0.6:
    method = "BAERASER"        # Aggressive mitigation
elif poison_score >= 0.3:
    method = "prune_finetune"  # Medium mitigation (or NAD if AC-low/STRIP-high)
else:
    method = "finetune_clean"  # Light mitigation + data augmentation
```

Calibrate `τ1`, `τ2` (or the 0.3/0.6 cutoffs) on a small validation grid (severity sweep across poison rates) — document the calibration procedure explicitly in the methods section; this is what keeps the controller "reviewer-safe."

---

## Per-Run Logging Requirement (Non-Negotiable)

For every single experimental run, log:
- `poison_score` (or the raw detection signals used)
- The defense method chosen
- The exact reasoning, in 1–2 numeric terms (e.g., `"silhouette=0.31, suspicious_fraction=0.06 → prune_finetune"`)

**Rule:** If you cannot explain the controller's choice in one sentence, the logic is too vague — simplify it.

---

## Controller Paper-Framing (Use Verbatim)

> "Activation Clustering provides an attack-agnostic severity signal based on representation separation, while STRIP serves as a post-remediation behavioral verification to ensure the absence of shortcut-triggered predictions."

> "We formulate backdoor remediation as a cost-constrained decision problem and empirically show severity regimes where lightweight defenses suffice, achieving comparable security at significantly reduced compute cost."
