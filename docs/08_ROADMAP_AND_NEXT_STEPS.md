# Roadmap & Next Steps — Resuming Implementation

**Status as of this handoff:** The Set 3 (May) experimental results in `latest_ppt_points.md` are **explicitly discarded**. The pipeline design, controller logic, attack/defense scope, and dataset strategy from Sets 1–2 remain valid and should be followed. The numeric results from Set 3 should be treated as illustrative-only formatting examples, never reused as real data.

---

## What's Locked (Don't Re-litigate These Decisions)

- Attacks: BadNets, Blended, Label-Consistent (3 total, Tier-1 only)
- Model: ResNet-18 only
- Datasets: CIFAR-10 (train), STL-10 (cross-dataset eval), Custom ~800 images (real-world/deployment), GTSRB optional
- Skip: CIFAR-100, exotic attacks (WaNet/Dynamic), exact unlearning (SISA), RL/learned controller
- Detection: Activation Clustering (primary) + STRIP (verification) + Grad-CAM (visualization/TAR)
- Defenses: Fine-tune, Prune+FT, BAERASER-lite (+ conditional NAD)
- Controller: rule-based, calibrated thresholds, two-signal (AC + STRIP) quadrant design preferred
- Seed: 2025, target class: 0 (airplane), clean defense budget: 2,500 fixed images

---

## What Was Left Open / Needs Resolving in the Redo

1. **Does the AC-low/STRIP-low-entropy quadrant actually appear in the data?**
   This determines whether the controller uses 3 regimes (simpler) or 4 regimes with NAD (stronger, but only if earned). Label-Consistent was the predicted candidate. Run AC + STRIP across all attack/poison-rate combos and plot the 2×2 scatter before deciding.

2. **BAERASER-lite stability.** The original phase plan flagged this as the highest-difficulty, highest-debug-risk item. Confirm it works reliably before committing to it as the "high severity" answer in the final controller — if it's unstable, fall back to Pruning+FT as the heavy option and document the limitation honestly.

3. **Threshold calibration values (τ1, τ2 or 0.3/0.6 cutoffs)** were never finalized with verified data — these must be recalibrated from scratch on the redone experiment results, not reused from any prior run.

4. **Custom dataset collection status** — was "Active In-Progress" as of the last update. Confirm current collection count (~800 target, 10 classes) before relying on it for OOD/deployment experiments.

---

## Suggested Re-Execution Order (Building on the Locked 12-Day / 7-Week Plans)

### Stage A — Foundations (Re-establish Reproducibility First)
1. Set up `core/` shared scripts: `models.py`, `data_utils.py`, `metrics.py`, `seed_utils.py` — single source of truth, all members import from these (see `06_EXPERIMENT_PROTOCOL_AND_REPRODUCIBILITY.md`)
2. Generate and share `defense_indices.npy` and poison index files — once, centrally, before any training starts
3. Train clean ResNet-18 baseline on CIFAR-10, save checkpoint + activation cache

### Stage B — Attacks + Baseline Poisoning
1. Implement/re-verify BadNets, Blended, Label-Consistent poison generators
2. Train poisoned models at 1%, 5%, 10% poison rates × 3 attacks × at least 2 seeds = 18+ checkpoints minimum
3. Confirm ASR is meaningfully elevated for each — if an attack doesn't produce a real attack effect, stop and fix it before moving on

### Stage C — Detection
1. Run Activation Clustering on every poisoned checkpoint, log silhouette/suspicious_fraction/cluster sizes to CSV
2. Run STRIP on every checkpoint, log entropy distributions
3. Plot the AC-severity vs STRIP-entropy scatter across all conditions — resolve the open question about the 4th controller quadrant here

### Stage D — Controller
1. Finalize 3-regime or 4-regime design based on Stage C's findings
2. Calibrate thresholds empirically on a validation slice
3. Log controller decisions + reasoning for every condition

### Stage E — Defenses
1. Implement/re-verify Fine-tune, Prune+FT, BAERASER-lite (+ NAD if earned)
2. Run the full ablation matrix (attack × poison rate × defense × seed)
3. Collect CA/ASR before-after, compute cost, for every cell

### Stage F — Verification & Figures
1. Re-run STRIP and Grad-CAM on all sanitized checkpoints
2. Compute TAR where defined (primarily BadNets)
3. Generate before/after figures for the paper

### Stage G — Cross-Dataset & Real-World
1. Evaluate sanitized checkpoints on STL-10 (no retraining) — CA + STRIP false-alarm rate
2. Evaluate on Custom dataset — CA, ASR (real-world trigger transfer), STRIP, Grad-CAM
3. Optional: GTSRB single transfer experiment if time remains

### Stage H — Post-Deployment Loop
1. Build the simulated production stream (STL-10 + Custom)
2. Implement STRIP-based suspicious buffer + periodic fine-tune update
3. Produce the 4-row lifecycle table (before deployment / after defense / during deployment attacks / after continual update)

### Stage I — Write-Up
1. Compile all tables/figures
2. Write the explicit failure taxonomy (which attack/poison-rate combos AC or STRIP missed, and why)
3. Draft using the contributions framing in `00_PROJECT_MASTER.md` and `09_REFERENCES_AND_PAPER_FRAMING.md`

---

## Team Role Reference (From Prior Planning — Adapt as Needed)

| Member Focus | Core Ownership |
|---|---|
| Data & Baselines | Dataset ingestion, poison generation scripts, baseline training, checkpoint management |
| Detection & Controller | Activation Clustering, severity scoring, controller logic, threshold calibration, STRIP integration |
| Unlearning Methods | Fine-tuning, pruning, NAD, BAERASER-lite implementation and tuning |
| Verification, Viz & Automation | STRIP, Grad-CAM, TAR, plotting, report generation, CLI (`scan_model.py`), reproducibility bundle |

**Buddy system recommendation (from prior planning):** pair each high-risk task (e.g., BAERASER-lite, fingerprint matching) with a buddy reviewer to share debugging load and avoid single points of failure.

---

## Definition of "Done" for the Redo (Before Writing the Paper)

- [ ] All checkpoints trained with fixed seed(s), shared indices, documented configs
- [ ] AC + STRIP run and logged for every condition (3 attacks × 3 poison rates × ≥2 seeds)
- [ ] AC vs STRIP scatter plot generated → 3 vs 4 regime decision made and documented
- [ ] Controller thresholds calibrated and logged with reasoning
- [ ] Full defense ablation matrix completed with CA/ASR before-after + compute cost
- [ ] STRIP + Grad-CAM verification completed for all sanitized models
- [ ] STL-10 cross-dataset evaluation completed (no retraining)
- [ ] Custom dataset OOD evaluation completed
- [ ] Post-deployment monitoring loop simulation completed with lifecycle table
- [ ] Failure taxonomy documented honestly
- [ ] Repository structured per `01_SYSTEM_ARCHITECTURE.md`, reproducible end-to-end via `run_phase1.sh`
