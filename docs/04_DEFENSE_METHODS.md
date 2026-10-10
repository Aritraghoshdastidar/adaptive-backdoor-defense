# Defense / Unlearning Methods — Detailed Specification

---

## Scope Discipline

**The defense suite consists of four methods mapped to severity & behavioral regimes:**

| Method | Role | Severity Tier |
|--------|------|---------------|
| **Fine-tuning (FT)** | Low-severity fix | Cheap, strong baseline |
| **ANP (Adversarial Neuron Pruning + FT)** | Mid-severity fix / Feature robustness | Adversarial mask optimization, fast & robust |
| **BAERASER-lite** | High-severity structural fix | Distillation + targeted unlearning |
| **NAD (Neural Attention Distillation)** | Stealthy / behavioral fix | Teacher-student attention distillation |

### Explicitly Out of Scope
- Exact unlearning (SISA) — infrastructure-heavy, not worth it for this scope
- VIBE — full retraining loop, too expensive
- Continual online machine unlearning — explodes scope; covered instead by the lighter "periodic re-evaluation" framing in `06_POST_DEPLOYMENT.md`
- Fisher-guided damping — complex, slow to tune

---

## 1. Fine-Tuning (Light)

- **What it does:** Continue training the poisoned model on the shared 5% clean budget (2,500 fixed CIFAR-10 images) at a low learning rate, with early stopping.
- **Why it works:** Overwrites shallow, low-severity backdoor associations without large architectural changes.
- **Implementation notes:**
  - Use the **exact same 2,500 images** (`defense_indices.npy`) across all team members and all attacks — this is mandatory for valid comparison
  - Low LR (e.g., 1e-4 to 1e-3), few epochs (5–15), monitor validation CA to avoid overfitting/catastrophic forgetting
- **Expected effect:** Strong ASR reduction on BadNets-level (low severity) poisoning; may be insufficient against Blended/SK.

---

## 2. ANP (Adversarial Neuron Pruning + Fine-Tuning)

- **What it does:** Uses Adversarial Neuron Pruning (Wu & Dong, 2021) to identify and mask vulnerable backdoor neurons by applying adversarial perturbations to neuron weights/activations on the clean budget, followed by light fine-tuning.
- **How it identifies backdoor neurons:**
  - Injects continuous perturbation noise to find neurons whose sensitivity is disproportionately exploited by backdoor shortcuts
  - Prunes/masks neurons with highest perturbation sensitivity
- **Implementation notes:**
  - Optimize mask via adversarial objective on the 5% clean budget (2,500 images)
  - Follow pruning with light fine-tuning to restore clean accuracy (CA)
- **Expected effect:** Robust mid-severity defense — highly effective at eliminating backdoor pathways with minimal CA drop.

---

## 3. BAERASER-Style Unlearning (Heavy)

- **Concept (from original BAERASER):** Recover the trigger via a max-entropy generator, then "unlearn" it via targeted gradient ascent on the recovered trigger pattern.
- **Scope decision (locked):** Full BAERASER (training a generative trigger-recovery model) is compute-heavy. We implement a **"BAERASER-lite" surrogate**:
  - Skip full generative trigger reconstruction
  - Use a simplified procedure: combine a distillation step (teacher = lightly fine-tuned clean model) with a masking/gradient-ascent step on samples flagged as highly suspicious by AC
  - Document explicitly in the paper: *"We implement a lightweight surrogate of BAERASER-style unlearning due to compute constraints; full generative trigger reconstruction is left as future work."*
- **When triggered:** High AC severity + behavioral signal present (high severity quadrant).
- **Implementation order:** High effectiveness against explicit localized shortcuts.

---

## 4. NAD (Neural Attention Distillation)

- **What it does:** Neural Attention Distillation (Li et al., 2021) — trains the poisoned ("student") model to match the intermediate attention maps of a teacher network fine-tuned on the clean budget, aligning feature representations.
- **Role in Suite:** Essential for stealthy or semantic triggers where AC representation clustering is weak but behavioral anomalies (STRIP) or distributed features are present.
- **Why it fits:** Operates directly on attention map alignment across intermediate residual groups without requiring explicit neuron localization.

---

## Defense Comparison / Ablation Matrix (Required for the Paper)

Run this sweep, varying one factor at a time:

| Factor | Variants |
|--------|----------|
| Attacks | BadNets, Blended, Silent Killer (or LC) |
| Poison rates | 1%, 5%, 10% |
| Trigger visibility | Small (4×4 patch), Subtle (low-α blended), Semantic/Distributed |
| Seeds | Single fixed seed: 2027 (no multi-seed ablation) |
| Defenses | (A) FT, (B) ANP, (C) BAERASER-lite, (D) NAD |

For every cell of this matrix, record:
- `CA_before`, `ASR_before`
- `CA_after`, `ASR_after`
- Compute cost (epochs, wall-clock minutes, approx GPU-hours)
- STRIP entropy distribution before/after
- Controller's chosen method + 1-sentence numeric reasoning

---

## Required Evaluation Tables/Plots

1. **ASR before/after** table, per attack × poison rate × defense
2. **Clean Accuracy before/after** table
3. **Δ Utility** (CA drop from defense)
4. **Cost–effectiveness ratio:**
   ```
   ASR reduction
   ──────────────
    compute cost
   ```
5. **Decision boundary plot:** x = AC severity, y = ASR after defense — directly supports the controller's core claim
6. **Compute cost table:** epochs, GPU-hours (approx), wall-clock time, per defense method

---

## Failure Case Handling Rules (Apply During Evaluation)

- **ASR remains high after defense:** Escalate to a heavier defense method, OR explicitly mark and document as a failure case for the failure taxonomy (reviewers like seeing this — it builds credibility, not weakness)
- **CA collapses after defense:** The defense was too aggressive — back off pruning fraction / reduce BAERASER-lite gradient ascent strength
- **Never silently discard a bad result** — log it, it's evidence for the limitations section

---

## What NOT to Overclaim

- ❌ "Fully removes all backdoors"
- ❌ "Works for all attack types"
- ❌ "Automatic for semantic triggers"

## Safe Framing to Use Instead

- ✅ "Adaptive mitigation works best for strong and medium backdoors"
- ✅ "Stealthy/semantic attacks remain challenging — we document these as failure modes and motivate the conditional NAD branch"
