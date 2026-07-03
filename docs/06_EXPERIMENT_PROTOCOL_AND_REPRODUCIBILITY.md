# Experiment Protocol & Reproducibility Rules

These rules are **mandatory**, not suggestions. The prior experiment round (Set 3 results) is being discarded and redone — use this document to make sure the redo doesn't repeat the same coordination/reproducibility gaps.

---

## Core Experiment Mindset

You are testing a **pipeline**, not just an attack or a defense in isolation. Every single run must answer three questions:
1. Was poison present?
2. Did we detect it?
3. Did the chosen defense reduce ASR without killing Clean Accuracy?

**Log numbers first, visuals second.** Don't generate Grad-CAM figures before you have the numeric table — numbers are the ground truth, figures illustrate them.

---

## Fixed Inputs (Never Change More Than One Variable at a Time)

Before running anything, these must be fixed and recorded:
- **Attack type:** BadNets / Blended / Label-Consistent
- **Poison rate:** Start with 5%, then add 1% and 10%
- **Trigger size/strength:** small vs visible (document exact pixel/alpha values used)
- **Random seed:** `2027` (write it down for every run — a single fixed global seed, no multi-seed ablation)
- **Model + dataset version:** ResNet-18 + CIFAR-10 (pin library versions in `requirements.txt`)

---

## Reproducibility Infrastructure — "Seed & Index" Strategy

### Step 1 — Fixed Global Seed (top of every script, every team member)

```python
import torch
import numpy as np
import random

def set_seed(seed=2027):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # crucial for reproducibility

set_seed(2027)
```

### Step 2 — Shared Clean-Defense-Budget Indices (generate ONCE, share with whole team)

```python
all_indices = np.arange(50000)
np.random.shuffle(all_indices)
defense_indices = all_indices[:2500]  # 5% clean budget
np.save('defense_indices.npy', defense_indices)
```

- All team members use the **exact same 2,500 images** for fine-tuning/pruning/distillation defenses.
- Store the file in a shared Drive folder; everyone loads from there, no regenerating locally.

### Step 3 — Shared Poison Indices (Explicit Protocol)
- **One canonical `.npy` file per (attack, rate) pair.** Generated once by whoever owns that attack, then promoted to the shared team Drive folder — never regenerated locally by other teammates, even if they think they're using the "same" seed/logic.
- **Nest indices across rates within an attack where feasible:** the 5% poison-index set should be a subset of the 10% set, and 1% a subset of 5% (`10% ⊇ 5% ⊇ 1%`), so that increasing the poison rate doesn't introduce a disjoint population of poisoned images. This makes poison-rate sweeps cleaner to interpret.
- **Naming:** `{attack}_poison_idx_{pct}pct.npy` (e.g. `badnets_poison_idx_10pct.npy`).
- **Retroactive promotion for already-trained BadNets:** BadNets is already trained and should not be retrained. Its existing saved `badnets_poison_idx_{pct}pct.npy` files (per rate, from the BadNets notebook) should be promoted as-is to the shared folder as the official record for BadNets, rather than regenerated under the nested-nesting rule above. Document any deviation from the nesting property as a known limitation, not fixed retroactively.
- **Blended and Label-Consistent generate their poison-index files fresh**, using the corrected, attack-appropriate pooling rule (non-target-class-only for Blended, target-class-only for LC — see `02_ATTACKS_AND_DATASETS.md`), following the nesting property above from the start.

### Step 4 — Clean Test Set
- Use the **entire standard CIFAR-10 test set** (10,000 images), unshuffled or shuffled with the fixed seed only. This measures Clean Accuracy.

### Step 5 — Poisoned (ASR) Test Set
- All members target **Class 0 (Airplane)**.
- All members apply their attack's trigger to the **same non-airplane test images** (use a shared `.npy` index list, typically ~1,000 images).
- This lets the team directly compare which attack is "most invisible" to detection on a true apples-to-apples basis.

---

## Why This Matters (Methodology Sentence, Ready to Use)

> "To ensure a fair comparison and eliminate data bias, all attacks and defenses were evaluated on a fixed validation subset and test set. We maintained a strict 5% clean data budget across all remediation strategies, using identical image indices for every experiment."

This single sentence pre-empts the most common reviewer rejection reason: "poor experimental setup."

---

## "Central Brain, Local Muscle" Workflow Model

Working in independent Colab notebooks is fine for the first week, but **independent code is the single biggest risk to validity.** If two members calculate ASR slightly differently, the final comparison table is scientifically invalid.

### The "Central Brain" — GitHub repo of shared scripts
One person sets up the repo. Everyone's Colab notebooks import from it:
- `models.py` — the exact ResNet-18 architecture (one definition, everyone uses it)
- `data_utils.py` — clean-subset creation + poison application logic (everyone calls the same function)
- `metrics.py` — single source of truth for `calculate_asr()` and `calculate_ca()`

### The "Local Muscle" — individual Colabs
Every notebook should start like this:
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<team>/backdoor-project.git
%cd backdoor-project

from scripts.models import get_resnet18
from scripts.data_utils import get_poisoned_loader
from scripts.metrics import evaluate_model

# Now run YOUR specific attack — the heavy lifting is in the shared scripts
```

**Process notes:**
- Daily 15-minute stand-up to sync checkpoints and next steps
- Merge only via PR, to keep experiments reproducible
- Store checkpoints on shared Drive, named `[attack]_[poison_pct]_[epoch]_[seed].pth`

### Immediate "Quick Fix" Steps (Do This First, This Week)
1. **Standardize the eval function** — pick the best-written `evaluate_model()` from the team, everyone copies that exact function. Ensures everyone tests on the same 10,000 CIFAR test images with `target_class = 0`.
2. **Use a shared index file** — generate `defense_indices.npy` once, upload to shared Drive, everyone loads it.
3. **Shared Results Sheet** — Google Sheet, columns:
   ```
   Member | Attack | Poison % | Baseline ASR | Defended ASR | CA Drop | Remediation Time (mins) | Silhouette Score
   ```
   This keeps the team in sync without waiting for the paper draft.

---

## Per-Run Logging Checklist (Detection → Decision → Defense → Verification)

For every single experimental condition (attack × poison-rate × seed), record:

**Baseline (before any defense):**
- `CA_before`, `ASR_before`
- Training config: epochs, LR, batch size
- **Note:** if ASR isn't high after attack injection, the attack failed — stop and fix it before proceeding to detection

**Detection (AC):**
- Silhouette score
- Cluster sizes `[C0, C1]`
- `suspicious_fraction`
- Rule of thumb: high silhouette + small cluster → likely backdoor; low silhouette ≠ "clean," it may mean a stealthy attack

**Controller decision:**
- `poison_score` (or the raw signals used)
- Defense method chosen
- 1–2 sentence numeric justification

**Defense execution:**
- Save the sanitized checkpoint (never overwrite the original poisoned model)
- Compute cost (epochs / wall-clock time)

**Post-defense verification (non-negotiable):**
- `CA_after` — must not drop severely
- `ASR_after` — should drop significantly
- STRIP entropy before/after distributions
- Grad-CAM / TAR — verify trigger attention reduced

**Failure handling:**
- ASR still high → escalate defense OR explicitly log as failure case
- CA collapsed → defense was too aggressive, back off

---

## Poison-Rate Behavior Pattern (Expect This, Shapes Controller Design)

- **1% poison:** Detection is weak; a light defense usually works
- **5% poison:** Detection is reliable; most defense types work
- **10% poison:** Easy to detect, but significantly harder to fully clean

The controller should actively change its mitigation behavior across these tiers — this pattern is core evidence for the "severity-aware" framing of the whole project.

---

## Reviewer Focus Areas (Don't Miss These)

- **Reproducibility:** constant random seeds, matching configurations across team members
- **Ablations:** sweeps across varying poison rates (1%, 5%, 10%) at a single fixed seed (`2027`)
- **Clear reporting:** clean before/after comparison tables
- **Honesty:** open evaluation of failure cases — don't hide them, they build credibility

---

## One-Line Goal to Remember

> **Detect → decide → defend → verify. Prove each step with numbers.**

---

## Repository Hygiene

- `.gitignore` the `data/` and `data_poisoned/` folders — don't push raw datasets to GitHub
- Host `.pth` checkpoints on shared Drive, link from README
- Keep checkpoint and log filenames consistent across members so the comparative-analysis script can auto-read everyone's data
- `requirements.txt` pinned to exact library versions to prevent "version drift" between teammates
