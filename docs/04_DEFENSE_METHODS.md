# Defense / Unlearning Methods — Detailed Specification

---

## Scope Discipline

**Exactly three core methods. No more, unless the 4th (NAD) is empirically earned (see `03_CONTROLLER_LOGIC.md`).**

| Method | Role | Severity Tier |
|--------|------|---------------|
| **Fine-tuning** | Low-severity fix | Cheap, strong baseline |
| **Neuron Pruning (+ fine-tune)** | Mid-severity fix | Fast, interpretable |
| **BAERASER-style unlearning** | High-severity fix | Heavy hammer |
| **NAD / Distillation** *(conditional 4th)* | Stealthy/behavioral fix | Only if AC-low/STRIP-low-entropy regime is observed |

### Explicitly Out of Scope
- Exact unlearning (SISA) — infrastructure-heavy, not worth it for this scope
- VIBE — full retraining loop, too expensive
- Continual online machine unlearning — explodes scope; covered instead by the lighter "periodic re-evaluation" framing in `06_POST_DEPLOYMENT.md`
- Fisher-guided damping — complex, slow to tune, optional Phase-2 only if pruning underperforms

---

## 1. Fine-Tuning (Light)

- **What it does:** Continue training the poisoned model on the shared 5% clean budget (2,500 fixed CIFAR-10 images) at a low learning rate, with early stopping.
- **Why it works:** Overwrites shallow, low-severity backdoor associations without large architectural changes.
- **Implementation notes:**
  - Use the **exact same 2,500 images** (`defense_indices.npy`) across all team members and all attacks — this is mandatory for valid comparison
  - Low LR (e.g., 1e-4 to 1e-3), few epochs (5–15), monitor validation CA to avoid overfitting/catastrophic forgetting
- **Expected effect:** Strong ASR reduction on BadNets-level (low severity) poisoning; may be insufficient against Blended/LC.

---

## 2. Neuron Pruning + Fine-Tuning

- **What it does:** Identify neurons in the penultimate (or earlier convolutional) layer that are unusually responsive to the trigger pattern, prune them, then lightly fine-tune to recover clean accuracy.
- **How to identify trigger-responsive neurons:**
  - Use activation statistics from the AC stage — neurons most active in the "suspicious cluster" are pruning candidates
  - Alternatively use Grad-CAM attention maps on triggered images to localize which channels light up on the trigger region
- **Implementation notes:**
  - Prune a small fraction first (e.g., top 1–5% most trigger-responsive channels), evaluate, increase if ASR remains high
  - Always follow pruning with a short fine-tune pass on the clean budget to recover accuracy
- **Expected effect:** Good middle-ground — removes structural backdoor pathways with minimal CA drop, when AC localizes the poison cleanly (BadNets-like).

---

## 3. BAERASER-Style Unlearning (Heavy)

- **Concept (from original BAERASER):** Recover the trigger via a max-entropy generator, then "unlearn" it via targeted gradient ascent on the recovered trigger pattern.
- **Scope decision (locked):** Full BAERASER (training a generative trigger-recovery model) is likely too compute-heavy for the timeline. Implement a **"BAERASER-lite" surrogate**:
  - Skip full generative trigger reconstruction
  - Use a simplified procedure: combine a distillation step (teacher = lightly fine-tuned clean model) with a masking/gradient-ascent step on samples flagged as highly suspicious by AC
  - Document explicitly in the paper: *"We implement a lightweight surrogate of BAERASER-style unlearning due to compute constraints; full generative trigger reconstruction is left as future work."* This sentence is reviewer-safe and avoids overclaiming.
- **When triggered:** High AC severity + low STRIP entropy (both diagnostics fire) — see controller quadrant matrix.
- **Implementation order:** Build this last, after fine-tuning and pruning are stable and working — it's the highest difficulty/debug-risk item.

---

## 4. NAD / Knowledge Distillation (Conditional)

- **What it does:** Neural Attention Distillation — train the poisoned ("student") model to match the attention maps of a small teacher network trained/fine-tuned on clean data only, transferring benign feature focus rather than relying on representation surgery.
- **When to add:** Only if experiments show a genuine AC-low/STRIP-low-entropy quadrant (i.e., AC misses the poison but STRIP catches behavioral lock-in). Most likely candidate attack: Label-Consistent, possibly low-poison-rate Blended.
- **Why it fits that gap specifically:** Pruning needs AC to localize neurons — if AC found nothing, pruning has no target. NAD works from behavioral supervision instead, making it the natural fix when representation-space detection fails but behavior-space detection succeeds.
- **Do not implement preemptively** — only after the AC vs STRIP scatter plot (described in `03_CONTROLLER_LOGIC.md`) confirms the quadrant is real in your data.

---

## Defense Comparison / Ablation Matrix (Required for the Paper)

Run this sweep, varying one factor at a time:

| Factor | Variants |
|--------|----------|
| Attacks | BadNets, Blended, Label-Consistent |
| Poison rates | 1%, 5%, 10% |
| Trigger visibility (BadNets/Blended only) | Small (4×4 patch), Medium (8×8), Subtle (low-α blended) |
| Seeds | 2-3 random seeds (2025 + at least two others) |
| Defenses | (A) Fine-tune, (B) Prune+FT, (C) BAERASER-lite, (D) NAD — if earned |

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
