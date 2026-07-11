# Blended Notebook — Stage C Status & Handoff
## Source: `blended_finalNB.ipynb`

**Purpose of this doc:** capture what's actually been run and confirmed in the
Blended notebook, where the outputs live, and the one significant finding
that came out of this pass — so the next person (or future-you) doesn't have
to re-derive any of it. This supplements `05_DETECTION_AND_VERIFICATION.md`
and feeds directly into Stage C / Stage D of `08_ROADMAP_AND_NEXT_STEPS.md`.

**Confidence key used below:**
- ✅ **Verified** — output inspected directly (from the uploaded `.ipynb` or
  pasted cell output) as part of this investigation.
- 🟡 **User-confirmed** — reported by the team as run, output pasted/described,
  but not independently inspected line-by-line.
- ⬜ **Present in code, not confirmed run** — the cell exists and is wired up
  correctly, but no output has been seen for it in this pass.

---

## 1. What's confirmed done

### Main sweep (pr01 / pr05 / pr10) — ✅ fully confirmed

| pr_tag | poison_rate | CA | ASR @ α_test=0.5 | ASR @ α_test=0.1 |
|---|---|---|---|---|
| pr01 | 1% | 94.83% | 100.0% | 99.90% |
| pr05 | 5% | 94.73% | 100.0% | 100.00% |
| pr10 | 10% | 94.39% | 100.0% | 100.00% |

`α_test=0.1` is the correct, matched-to-training number (`ALPHA_TRAIN=0.1`) and
is the one that should be used everywhere downstream — it's now confirmed run
(Cell "19" / Section J), not just proposed. **All three checkpoints are
confirmed 100%-ASR-equivalent backdoors under the trigger strength that
Detection/STRIP actually tests them at.** This matters because it rules out
"the pr10 backdoor is just weaker under 0.1" as an explanation for what
follows.

### Activation Clustering (AC) — ✅ confirmed (Cell 19 / "Cell 11", DR comparison)

| pr_tag | DR method | silhouette | suspicious_fraction | PDR | recall |
|---|---|---|---|---|---|
| pr01 | PCA-2D | 0.3778 | 26.69% | 11.16% | 90.00% |
| pr01 | ICA-10D | 0.1102 | 9.09% | 100.00% | 100.00% |
| pr05 | PCA-2D | 0.5408 | 33.37% | 99.76% | 99.88% |
| pr05 | ICA-10D | 0.0983 | 33.33% | 100.00% | 100.00% |
| pr10 | PCA-2D | 0.6380 | 50.00% | 99.96% | 99.98% |
| pr10 | ICA-10D | 0.1744 | 49.99% | 99.96% | 100.00% |

**Open decision (unchanged from doc05):** which DR method is "official" for
Blended is still not locked. PCA-2D gives cleaner monotonic silhouette growth
with poison rate; ICA-10D gives much better PDR at pr01. Don't pick one
implicitly via whichever CSV gets merged first — decide and document it in
doc05 §"Per-attack DR choice" before this feeds the controller.

### STRIP on main sweep — ✅ confirmed, and the pr10 result is a real finding

| pr_tag | clean entropy μ | threshold (FRR=1%) | TPR | FPR |
|---|---|---|---|---|
| pr01 | 0.4298 | 0.0885 | 28.0% | 2.4% |
| pr05 | 0.4662 | 0.1014 | 40.8% | 1.6% |
| pr10 | 0.4671 | 0.1037 | **0.0%** | 2.0% |

**Full verification chain for the pr10 finding** (each step ruled out a
possible artifact before treating this as a real result):

1. **Mismatched conditions, ruled out.** ASR and STRIP's TPR were at risk of
   being apples-to-oranges (ASR historically measured at `alpha_test=0.5`,
   STRIP's trigger pool built at `alpha_test=0.1`). Re-evaluating ASR at the
   exact trigger strength STRIP tests (Cell 29 re-eval) confirms **100% ASR
   under matched conditions** — same trigger strength on both sides of the
   comparison.
2. **Threshold-boundary noise, ruled out.** Distributional check (percentiles,
   not just min/max/mean) on the raw entropy arrays:

   | pr_tag | triggered min | triggered p10 | triggered median | % triggered below clean median |
   |---|---|---|---|---|
   | pr01 | 0.0329 | 0.0600 | 0.1272 | 94.0% |
   | pr05 | 0.0347 | 0.0654 | 0.1144 | 98.8% |
   | pr10 | **0.1392** | **0.4854** | **0.6806** | **9.6%** |

   At pr10, the *minimum* triggered-image entropy (0.1392) already sits above
   the detection threshold (0.1037) — no single sample could have been
   flagged. Not a few outliers dragging a mean — the entire distribution
   moved.
3. **General model drift, ruled out.** Clean-image entropy is flat across all
   three models (0.43 → 0.47 → 0.47), so this isn't the model's general
   calibration drifting; it's specific to the triggered condition.
4. **AC also missing it, ruled out.** AC on the same pr10 checkpoint gives
   silhouette=0.638, PDR=99.96% — AC is *not* the weak signal here; it's
   unambiguously strong. This is specifically a STRIP failure mode.
5. **Implementation bug, ruled out.** Read `run_strip`/`strip_entropy_single`/
   `add_blended_trigger_global` in `core/detection.py` and `core/attacks.py`
   directly — no bug found, same `blend_pattern` object used throughout,
   correct threshold/TPR logic.

### → The finding (write this into doc05, done — see file 2 below)

**At pr10, Blended is a confirmed 100%-ASR backdoor (matched trigger
strength) that STRIP's TPR fully misses (0%), while AC detects it very
strongly (silhouette 0.64, PDR 99.96%).** This is the *opposite* pairing from
what doc05's original taxonomy assumed for Blended ("AC ⚠️ / STRIP ✅") — that
pairing holds at 1% and 5%, but **inverts at 10%**. This is not the
previously-tracked "4th quadrant" question (AC-low/STRIP-low) — it's a new,
different failure mode (AC-high/STRIP-fails) that the existing 2×2 controller
matrix already has an answer for (`prune_finetune`), so it doesn't require a
new defense regime — just an honest note in the failure taxonomy and in the
STRIP entropy plot for the paper.

**Confidence level:** confirmed real, confirmed not an artifact of the
evaluation setup (mismatched alpha, threshold noise, general drift, or AC
also failing), confirmed isolated to STRIP specifically at pr10. This is
solid enough to state as fact in the taxonomy.

**Mechanism (plausible, not yet confirmed):** STRIP's own perturbation is
itself an alpha-blend (default α=0.5) — the same mechanism Blended's trigger
uses. At high poison rate the model may lock onto a narrow, specific blend
intensity; STRIP's own 50% overlay dilutes the baked-in trigger pattern
weight (0.1 → effectively ~0.05 in the final blended-then-perturbed image),
and at pr10 this dilution appears to break the shortcut rather than just add
noise. **This causal story is not required for doc05** (the observation alone
is enough per doc04/06's logging rules) — it's an optional deeper dive if
time allows later (see doc04 Scope Discipline; doc08 "if time permits"). This
is the one open piece: the *what* is fully verified, the *why* is a plausible
hypothesis only — don't write the mechanism into the paper as established
fact until it's tested directly (§4, item 1).

---

## 2. File inventory (Drive: `ps-capstone/`)

### Core / always needed
| File | Contents | Keyed by |
|---|---|---|
| `resnet18_blended_pr01/05/10.pth` | Checkpoints | pr_tag |
| `blended_poison_idx_pr01/05/10.npy` | Canonical poison indices (nested 1%⊂5%⊂10%) | pr_tag |
| `blended_pattern_seed777.npy` | Shared blend pattern, `RandomState(777)` — same object used everywhere, verified | — |

### Results tables
| File | Contents | Status |
|---|---|---|
| `blended_results.csv` | Main sweep CA/ASR @ α_test=0.5 (legacy/saturated — appendix only) | ✅ |
| `blended_results_alphatest_01.csv` | Main sweep CA/ASR @ α_test=0.1 (**primary, use this one**) | ✅ |
| `blended_alpha_test_sweep_pr01.csv` | Supplementary α_test sweep, pr01 checkpoint only (0.1/0.2/0.3/0.5) | ✅ |
| `blended_dr_comparison.csv` | AC severity, PCA-2D vs ICA-10D, all 3 rates | ✅ |
| `blended_main_strip_results.csv` | STRIP TPR/FPR/threshold, all 3 rates, α_test=0.1 | ✅ |
| `blended_main_strip_entropies_{pr_tag}.npy` + `_labels_{pr_tag}.npy` | Raw per-sample STRIP entropies + ground truth (1=triggered, 0=clean) | ✅ (used directly for the distributional check above) |

### Low-count / sub-1% arm (Section G/H/I)
| File | Contents | Status |
|---|---|---|
| `resnet18_blended_n{0001..0025}.pth` | Sub-1% poison-count checkpoints | ⬜ not re-confirmed this pass |
| `blended_lowrate_results.csv` / `blended_lowrate_results_01_alphatest.csv` | ASR at α_test=0.5 / 0.1 respectively | ⬜ |
| `blended_lowrate_ac_results.csv` | AC severity vs poison count | ⬜ (referenced in notebook, not inspected here) |
| `blended_lowrate_strip_results.csv` + `_entropies_/_labels_{tag}.npy` | STRIP vs poison count | ⬜ |

**If you're picking this up next:** the low-rate arm wasn't part of this
investigation — don't assume it's stale, just re-check its outputs before
citing numbers from it, the same way we just did for the main sweep.

### Activations (Stage C prerequisite)
| File | Contents |
|---|---|
| `activations/blended_{pr_tag}_seed2027.npy` (naming per `01_SYSTEM_ARCHITECTURE.md`) | Cached penultimate-layer activations, feeds Cell 19's AC comparison |

---

## 3. What to reference where

- **Trigger construction / attack mechanism:** `core/attacks.py` →
  `add_blended_trigger_global` — confirmed matches doc02's spec (raw pixel
  blend, non-target-class pool only).
- **AC + STRIP implementation:** `core/detection.py` → `run_ac`, `run_strip`,
  `strip_entropy_single`. Confirmed correct (raw-pixel STRIP blending,
  batched inner loop, threshold via `scipy.stats.norm.ppf` on clean entropy
  fit). One minor reproducibility note: `strip_entropy_single`'s overlay-image
  sampling uses the global NumPy RNG, not the seeded `rng` passed into
  `run_strip` — results are stable but not bit-for-bit reproducible run to
  run. Worth a footnote in `06_EXPERIMENT_PROTOCOL_AND_REPRODUCIBILITY.md` if
  exact reproducibility is ever audited; not a correctness bug.
- **Failure taxonomy:** `05_DETECTION_AND_VERIFICATION.md` (updated, see
  companion file).
- **Next roadmap step this unblocks:** `08_ROADMAP_AND_NEXT_STEPS.md` Stage C
  item 3 ("plot AC-severity vs STRIP-entropy scatter") — Blended's three
  points are now ready to plot; still need BadNets and LC run through the
  same AC+STRIP pass before the full scatter (and Stage D calibration) can
  happen.

---

## 4. Optional follow-up (not required for doc-of-record, do only if time allows)

Per doc04's Scope Discipline and doc08's "if time permits" framing, these are
**not** blockers for Stage C/D and should not delay moving on to LC/BadNets:

1. Sweep `STRIP_ALPHA` (STRIP's own overlay weight) at pr10 only, to test the
   dilution-collision hypothesis directly.
2. Check whether the inversion is a cliff or a slope between 5% and 10% (only
   if intermediate checkpoints are cheap to get).
3. Run the same matched-ASR + STRIP check on BadNets pr10, to see if the
   inversion is Blended-specific (patch trigger vs. blend trigger, different
   representation space than STRIP's own perturbation).
