# Paper Framing, Reviewer-Safe Sentences & References

---

## Paper Identity

**Title:** "Adaptive Hybrid Machine Unlearning for Backdoor Defense"
**Alternative/product framing title:** "Model Safety Scanner: A Cost-Aware, Severity-Calibrated Pipeline for Automated Backdoor Detection and Remediation"

**Type of paper:** System / orchestration paper, not an algorithm paper. The novelty is in integration, decision logic, and thorough empirical evaluation — not in inventing a new detector, attack, or unlearning primitive.

---

## System Pipeline (One Sentence)

> Training data → Activation Clustering → severity score (d) → Decision Controller (policy π) → Unlearning action (FT / Prune / BAERASER-lite / NAD) → STRIP + Grad-CAM verification → Audit report → Post-deployment monitoring.

---

## The Five Claimed Contributions

1. **Formal decision formulation** — backdoor remediation as a constrained optimization over security, utility, and compute.
2. **Data-calibrated policy controller** — thresholds calibrated offline; logic fixed and auditable (not learned).
3. **Severity regime discovery** — empirical phase transitions across attacks and poison rates.
4. **Cost-aware evaluation** — runtime, GPU-hours, accuracy tradeoffs (rarely reported in prior work).
5. **End-to-end deployable pipeline** — detection → remediation → verification → post-deployment monitoring, framed as a full model-lifecycle defense.

---

## Why Reviewers Should Accept This

- Doesn't claim perfect detection
- Doesn't invent a new attack
- Does provide decision logic + evidence
- Does show tradeoffs engineers actually care about (cost, not just accuracy)

---

## Reviewer-Safe Sentences (Use Verbatim Where Applicable)

| Purpose | Sentence |
|---|---|
| Attack selection | "We select BadNets, Blended, and Label-Consistent to represent increasing trigger subtlety and semantic realism, enabling systematic evaluation of severity-aware remediation regimes." |
| AC + STRIP roles | "Activation Clustering provides an attack-agnostic severity signal based on representation separation, while STRIP serves as a post-remediation behavioral verification to ensure the absence of shortcut-triggered predictions." |
| Threshold calibration | "Thresholds are empirically calibrated via offline validation to minimize ASR + utility loss under compute constraints." |
| Continual learning | "Periodic safety re-evaluation under incremental data updates." |
| Overall contribution | "We formulate backdoor remediation as a cost-constrained decision problem and empirically show severity regimes where lightweight defenses suffice, achieving comparable security at significantly reduced compute cost." |
| Industry framing | "Pre-deployment model safety scanner." |
| Cross-dataset rationale | "Experiments were conducted across CIFAR-10 (training and baseline evaluation), STL-10 (cross-dataset robustness), and a custom real-world dataset collected by the authors, with GTSRB included as an optional safety-critical transfer case." |
| Reproducibility | "To ensure a fair comparison and eliminate data bias, all attacks and defenses were evaluated on a fixed validation subset and test set. We maintained a strict 5% clean data budget across all remediation strategies, using identical image indices for every experiment." |
| BAERASER-lite caveat | "We implement a lightweight surrogate of BAERASER-style unlearning due to compute constraints; full generative trigger reconstruction is left as future work." |

---

## What NOT to Claim, Anywhere

- "Fully removes all backdoors"
- "Works for all attack types"
- "Automatic defense against semantic triggers"
- "Online learning" or "adaptive defense" in the continual-update context (use "periodic safety re-evaluation" instead)
- "Enterprise-ready security product" (use "pre-deployment model safety scanner" instead)

---

## Suggested Paper Sections

1. Introduction
2. Threat model
3. System design (pipeline diagram)
4. Formal decision problem (controller objective)
5. Policy design (controller logic, quadrant matrix)
6. Implementation
7. Experiments (attacks × poison rates × defenses, ablations)
8. Cross-dataset / real-world case study (STL-10, Custom, optional GTSRB)
9. Post-deployment monitoring case study
10. Discussion / Failure taxonomy / Limitations
11. Appendix (full tables, hyperparameters)

---

## Required Experiment Tables (Must Appear Or Reviewers Will Reject)

- ASR (before/after) per attack/poison-rate/defense
- Clean Accuracy (before/after)
- Δ Utility (accuracy drop)
- Detection precision/recall (internal validation against ground-truth poison indices)
- Compute cost: epochs, approx GPU-hours, wall-clock time
- Cost-effectiveness ratio: ASR reduction / compute cost
- Decision boundary plot: AC severity (x) vs ASR after defense (y)
- AC-severity vs STRIP-entropy scatter (drives the 3- vs 4-regime controller decision)
- 4-row lifecycle table for post-deployment monitoring

---

## Industry / Real-World Relevance Summary

**Who cares (high relevance):**
- Autonomous vehicles / ADAS vendors (physical camera-input backdoors are a real risk)
- Automotive Tier-1s, robotics, industrial automation
- Healthcare imaging/diagnostic vendors (regulatory + safety pressure)
- Aerospace, defense, specialized industrial vision systems
- Any regulated industry with model certification/audit requirements

**Moderate relevance:**
- Large platforms with user-provided training data
- Fintech fraud models

**Concrete scenario (use in intro/motivation):**
A company downloads a third-party CNN for traffic-sign recognition. A backdoor causes the model to misread a stickered stop sign as a speed-limit sign — a real safety hazard. Activation Clustering on the (likely GTSRB-style) training data would surface a suspicious cluster on "speed limit" predictions; the controller sanitizes the model pre-deployment; STRIP at inference ensures no latent backdoor remains before the model ships.

**2026 relevance points:**
- AI safety regulation (evolved EU AI Act) mandates explainability and remediability of model failures — Grad-CAM directly supports this
- Models increasingly trained on massive uncurated web-scraped data → rising poisoning risk
- Retraining large 2026-era vision models costs millions — the adaptive controller's cheapest-sufficient-defense logic (pruning/fine-tuning vs full retrain) is a direct cost-saving argument

---

## Background Concepts Referenced in Project Docs (For Quick Recall)

- **Activation Clustering (AC, Chen et al. 2018):** clusters hidden-layer activations per class to flag potential poisoned subpopulations without needing a trusted clean dataset.
- **STRIP (Gao et al. 2019):** STRong Intentional Perturbation — inference-time entropy test distinguishing clean (high entropy under perturbation) from backdoored (low entropy, locked prediction) inputs.
- **BAERASER:** recovers the trigger pattern via a max-entropy generator, then performs targeted gradient-ascent "unlearning" of that trigger. Computationally heavy — this project uses a lightweight surrogate ("BAERASER-lite").
- **Fine-Pruning (Liu et al.):** prunes neurons highly responsive to backdoor triggers, then fine-tunes on clean data to recover accuracy.
- **NAD (Neural Attention Distillation, Li et al.):** transfers attention focus from a clean-trained teacher to the poisoned student model, removing reliance on trigger-driven shortcut features.
- **Grad-CAM:** gradient-weighted class activation mapping — visualizes which image regions drive a prediction; used here both for paper figures and as the basis for the Trigger Attention Ratio (TAR) metric.

Use these as background/related-work anchors; do not reproduce text from any source paper verbatim — describe each in your own words when writing the related work section.

---

## Publication Targets (Ranked)

**Tier 1 — Workshops (submit first):**
- AISec (ACM Workshop on AI & Security)
- NeurIPS / ICLR / ICML workshops on backdoor, safety, or "BUGS" themes

**Tier 2 — Top security conferences (once validated across multiple datasets):**
- USENIX Security, IEEE S&P (Oakland), ACM CCS, NDSS

**Tier 3 — ML/vision main venues (requires an added novel algorithmic component):**
- NeurIPS, ICML, ICLR, CVPR, ICCV (e.g., PolicyCleanse-style framing at ICCV 2023 is a relevant comparator)
