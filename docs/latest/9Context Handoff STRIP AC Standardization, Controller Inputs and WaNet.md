# Context Handoff: STRIP/AC Standardization, Controller Inputs and WaNet

Status as of Nov 22, 2026. Covers everything discussed since the question 'how did STRIP configs change?'. Items marked (est.) are estimates, not measurements. Items marked (unverified) come from a separate review of notebooks I have not seen myself.

## 0. TL;DR

- The controller's STRIP inputs were measured under TWO different superimposition formulas (BadNets: straight addition; Blended 'verified' numbers: averaging). This explains the old 79.6/92.8/68.0 vs 28.0/40.8/0.0 Blended discrepancy.
- A git commit on Sep 7 changed STRIP from addition to averaging. The docstring in `core/detection.py` still describes addition.
- Silent Killer (SK) AC values in the controller table were probably computed on clean images (bug). Re-measured on the rebuilt poisoned set, SK 5% has silhouette 0.565 and PDR 99%, not 0.37.
- The Blended 10% 'STRIP blind spot' (0% TPR) exists only under averaging. Under addition it is 65%.
- Recommended final STRIP config: addition, 250 triggered + 250 clean, 100 overlays, FRR 1%, one config for all attacks and for pre/post-defense. Report averaging as a labeled sensitivity result.
- The controller's value is compute cost, not security: ANP alone already reaches about 0.5% ASR on the solvable conditions.
- WaNet v5 notebook (patched from v4) is ready to run; defenses do not depend on the STRIP formula.

## 1. How STRIP configs differed across notebooks

### 1.1 What the notebooks show (unverified, from a separate review of 7 notebooks)

| Notebook | Total samples | Overlays | FRR | Overlay weight |
| --- | --- | --- | --- | --- |
| BadNets attack | 500 (250+250) | 100 | 0.01 | 0.5 |
| Blended attack | 500 | 100 | 0.01 | 0.5 |
| SK attack | 500 | 100 | 0.01 | 0.5 |
| BadNets defense | 200 | 50 | 0.01 | default |
| Blended defense 1 | 1000 | 50 | default | default |
| Blended defense 2 (BaEraser) | 500 | 100 | 0.01 | 0.5 |
| SK defense v4 | 1000 | 50 | 0.01 | 0.5 |

The three attack notebooks agree. Three defense notebooks do not match them. Note that `n_samples` in the repo's `run_strip` is the TOTAL (half triggered, half clean).

### 1.2 Repo evidence

- `run_strip` defaults in the repo: 200 total, 50 overlays, labelled first-pass; an older comment says 500/100.
- `strip_entropy_single` docstring says 'straight addition' but the code (since Sep 7) does `np.clip((img + clean) / 2.0, 0, 1)`, which equals a 0.5 weighted blend.
- Git: commit on Mon Sep 7 2026 11:41 changed `np.clip(img_arr + clean_arr, 0, 1)` to the averaged version ('Modify image blending to average before clipping'). So before that time the repo used straight addition.
- Git: commit at 10:54 the same day 'Change badnets detection to PCA-10D'. But the current config is `ATTACK_DR_CONFIG['badnets'] = (False, 10)`, and `use_pca=False` means ICA-10D. The message and the behavior disagree. Blended is `None` (falls back to the ICA-10D default).
- Other AC drift: `run_ac` now applies a StandardScaler, PCA/ICA, and KMeans with `n_init=50` (comment says it was 20 before) on the predicted-class group.
- Because Colab clones whatever is on GitHub at run time and notebooks do not record the commit hash, it cannot be known from the notebooks alone which version each result used.

### 1.3 Other STRIP numbers that were in circulation

- Controller table (stale): Blended TPR 79.6 / 92.8 / 68.0.
- Blended stage C status doc (verified): 28.0 / 40.8 / 0.0 (500 samples, 100 overlays, alpha\_test 0.1).
- Two other Blended sets: 3.4 / 8.8 / 0 (Defense 1 notebook) and 10 / 24.4 / 0 (Defense 2 BaEraser notebook, per the review these may be post-defense evaluations).

## 2. The re-measurement experiment (executed)

Notebook: `remeasure_strip_ac`. Repo HEAD `87fbf96e10bb0216ad426e0f33770ead770c2c16`. Run stamp 20261122\_081505. All parameters explicit: 500 total samples (250+250), 100 overlays, FRR 1%, seed 2027, uint8 truncation, overlay pool = raw CIFAR test set. SK uses its own first-256-images normalizer (recomputed and matched).

### 2.1 Sanity (matched-condition ASR)

Reproduces known baselines: BadNets 82.7 / 95.4 / 96.1%, Blended 99.9 / 100 / 100%, SK 0.5%: 99.7, 1%: 89.5, 5%: 100, 10%: 100. Triggers and checkpoints are valid. 10 checkpoints found.

### 2.2 Fingerprint: Blended pr05 STRIP TPR (%)

| Overlays | add | average |
| --- | --- | --- |
| 50 | 90.8 | 34.4 |
| 100 | 92.0 | 45.2 |

Matches: attack-notebook 40.8 (100 overlays) is the averaging formula (45.2 is within sampling noise); the 92.8 in `blended_attack` Cell 20 is the addition formula. The 8.8 from Defense 1 (50 overlays) matches neither (34.4 or 90.8). That one is UNEXPLAINED (likely post-defense, or a different FRR, pool or trigger strength).

### 2.3 Full STRIP TPR (%) by formula (asr\_idx pool)

| Attack | Rate | add | average | Formula-sensitive |
| --- | --- | --- | --- | --- |
| BadNets | 1% | 11.6 | 0.0 | no |
| BadNets | 5% | 37.2 | 0.4 | yes |
| BadNets | 10% | 34.8 | 1.2 | yes |
| Blended | 1% | 80.0 | 27.6 | yes |
| Blended | 5% | 92.0 | 45.2 | yes |
| Blended | 10% | 65.2 | 0.0 | yes |
| SK | 0.5% | 54.4 | 44.4 | mild |
| SK | 1% | 8.4 | 8.4 | no |
| SK | 5% | 86.8 | 87.6 | no |
| SK | 10% | 95.6 | 96.8 | no |

SK source-class (cat) pool: 0.5%: 62.4/39.2, 1%: 0.4/2.8, 5%: 85.6/90.4, 10%: 94.4/97.6 (add/average). Both pools agree on routing.

Key observations:

- Old BadNets controller inputs (7.6 / 34.4 / 33.2) match ADDITION (11.6 / 37.2 / 34.8). Old Blended 'verified' inputs (28.0 / 40.8 / 0.0) match AVERAGING. The controller table mixed formulas.
- Under averaging, STRIP gives about 0% TPR for BadNets (the repo docstring had already warned about this).
- Overlay count matters under averaging (Blended pr05: 34.4 at 50 vs 45.2 at 100) but barely under addition.
- SK STRIP is robust to the formula.

### 2.3b Interval note (Wilson 95%, 250 triggered inputs, TPR sampling error only)

Under addition: BadNets 1% 11.6 \[8.2, 16.2\]; BadNets 5% 37.2 \[31.4, 43.3\]; BadNets 10% 34.8 \[29.2, 40.9\] (contains 30); Blended 1% 80.0 \[74.6, 84.5\]; Blended 5% 92.0 \[88.0, 94.8\]; Blended 10% 65.2 \[59.1, 70.8\]; SK 1% 8.4 \[5.6, 12.5\]; SK 5% 86.8 \[82.0, 90.4\]. Under averaging Blended 1% (27.6 \[22.4, 33.4\]) also straddles 30. The threshold itself is a normal fit on 250 clean entropies, which adds further uncertainty.

### 2.4 Activation Clustering re-measure

- Blended (PCA-2D): 0.3778 / 0.5408 / 0.6380, PDR 11.16 / 99.76 / 99.96. ICA-10D: 0.1102 / 0.0983 / 0.1744. These reproduce the earlier values EXACTLY at current HEAD, so Blended AC is not affected by the repo drift and needs no re-run.
- SK (rebuilt from saved poisoned pixels): 0.5%: PCA-2D 0.3317, PDR 7.46, recall 83.2; 5%: 0.5652, PDR 99.10, recall 97.2; 10%: 0.4416, PDR 100, recall 29.02 (whole target class is poisoned, so degenerate). SK 1%: cannot be rebuilt because the poisoned pixels were not saved.
- BadNets AC: SKIPPED. The poison-index glob found no files.
- The re-measure notebook states the SK attack notebook's AC used a poison tensor of all zeros, i.e. it clustered clean images. That would explain the controller's SK silhouettes of 0.33 / 0.33 / 0.37. Still to verify: that the saved pixel file is the exact set that trained the model.

## 3. What the re-measurement implies

1. The Blended 10% 'STRIP blind spot' (0% TPR) is formula-dependent: 0% under averaging, 65.2% under addition. It must not be presented as a general STRIP failure. Say: STRIP with a 0.5-weighted overlay dilutes the blended trigger.
2. 'SK evades AC' holds at low poison (0.5%: silhouette 0.33, PDR 7.5%), not at 5% (silhouette 0.565, PDR 99%).
3. The SK AC inputs in the controller table (0.5%, 1%, 5%) are likely invalid and must be replaced or marked unavailable.
4. The 36-cell defense matrix is unaffected: defense results do not depend on detection.
5. Post-defense STRIP numbers must be recomputed under one formula.

## 4. Routing under each formula (thresholds frozen: tau\_ac 0.40, tau\_strip 30%)

AC for BadNets is NOT re-measured (marked \*). SK 1% has no valid AC.

| Condition | AC | STRIP add | STRIP avg | Route (add) | Route (avg) |
| --- | --- | --- | --- | --- | --- |
| BadNets 1% | 0.32\* | 11.6 | 0.0 | ANP | ANP |
| BadNets 5% | 0.37\* | 37.2 | 0.4 | NAD | ANP |
| BadNets 10% | 0.44\* | 34.8 | 1.2 | BaEraser | ANP |
| Blended 1% | 0.378 | 80.0 | 27.6 | NAD | ANP |
| Blended 5% | 0.541 | 92.0 | 45.2 | BaEraser | BaEraser |
| Blended 10% | 0.638 | 65.2 | 0.0 | BaEraser | ANP |
| SK 5% | 0.565 | 86.8 | 87.6 | BaEraser (failed) | BaEraser (failed) |
| SK 1% | n/a | 8.4 | 8.4 | needs AC | needs AC |

Outcomes using the existing 36-cell matrix (my arithmetic, not yet confirmed by `evaluate_controller.py`):

- Addition routing: average ASR about 3.5% on the 6 solvable conditions, 1 failure (Blended 10% via BaEraser, 16.73%, a selection failure). Compute about 51% less than Always-ANP with capped normalized costs, about 58% less in raw seconds.
- Averaging routing: all BadNets go to ANP, so the controller is nearly Always-ANP: about 0.43% average ASR, 0 failures, only about 12% compute saving.
- So the formula choice drives the cost story (about 58% vs about 12% savings). Do not choose the formula by which headline looks better; disclose both.

## 5. Why the controller's value is cost, not security

On the solvable conditions Always-ANP already gets about 0.53% average ASR with 0 failures; Always-BaEraser 3.47% with 1 failure; Always-NAD 17.5% (with the invalid cell penalized); Always-FT 58%. A controller can at best match ANP on security; its gain is using cheaper defenses where they suffice. Matrix timings: FT about 18 s, BaEraser about 216 s, NAD about 256 s, ANP about 750 s (check the earlier ANP 0.0 s timing bug; use real timings).

## 6. Why BaEraser gets chosen when ANP looks best

The quadrant map (AC high + STRIP high -> BaEraser, AC high + STRIP low -> ANP, AC low + STRIP high -> NAD, AC low + STRIP low -> ANP as the policy fix) was written in the design docs before the matrix existed. It was not derived from results. Under addition, more conditions fall in HIGH/HIGH, so BaEraser is chosen more.

### 6.1 What the matrix actually supports (ASR % after defense)

| Condition (quadrant under add) | FT | ANP | NAD | BaEraser |
| --- | --- | --- | --- | --- |
| BadNets 1% (LOW/LOW) | 18.7 | 0.2 | 1.9 | 0.8 |
| BadNets 5% (LOW/HIGH) | 24.2 | 0.4 | 1.1 | 0.9 |
| Blended 1% (LOW/HIGH) | 100 | 0.06 | 0.74 (teacher backdoored) | 0.15 |
| BadNets 10% (HIGH/HIGH) | 6.5 | 0.4 | 1.3 | 0.9 |
| Blended 5% (HIGH/HIGH) | 100 | 1.88 | 0.22 | 1.33 |
| Blended 10% (HIGH/HIGH) | 100 | 0.21 | invalid (collapse) | 16.73 (selection failure) |
| SK (all rates) | 82-100 | 86-100 | 45-71 (teacher backdoored) | failed |

- FT mostly fails; ANP works on all 6; NAD and BaEraser work on 5 of 6 and fail only at Blended 10%.
- No quadrant-specific pattern beyond that. Any new map is a six-point fit to the same data used for evaluation.

## 7. BaEraser checkpoint selection (fresh\_select\_k)

- Blended 10% BaEraser = 16.73% because the selected epoch was 1; epoch 15 (oracle, best real ASR) gives 0.46% but that is a diagnostic, not a defense result.
- `fresh_select_k=1` (used so far): epoch chosen by proxy ASR on the recovered triggers under a CA floor; the proxy can be gamed. `fresh_select_k=3`: the top 3 epochs are re-ranked by the ASR of freshly recovered triggers on that epoch's weights (harder to game).
- If changed, change it for EVERY BaEraser cell (BadNets, Blended, SK, WaNet), locked a priori, with k=1 results kept in an appendix. Otherwise the Always-BaEraser baseline becomes inconsistent. Not required: the selection failure can be disclosed honestly.
- SK 1% and 5% BaEraser failed with no valid checkpoint (no eligible epoch); k probably does not fix that.
- `ca_loader` / `asr_loader` in `baeraser_unlearning` are monitoring only; selection uses the select slice (no leak).

## 8. WaNet notebook versions

- v1 (original): bugs: AC kept only the last batch and never clustered the poisoned class; STRIP reused 20 overlays; controller stub; weight-zeroing ANP; trigger applied before crop/flip.
- v2: infrastructure rewrite (Drive/GitHub, CFG, held-out split, real ANP/NAD). Superseded: its split did not match Blended's and BaEraser never executed.
- v3/v4: built with the repo files. Same Blended split (2000 defense / 500 select / 7500 report), Blended training recipe (SGD lr 0.1, wd 5e-4, 80 epochs, cosine, bs 128), repo `run_ac` (StandardScaler, n\_init 50, predicted-class group), repo `nad_full_pipeline`, repo `recover_trigger_pool` / `baeraser_unlearning` with a WaNet trigger operator. Smoke-test mode (`smoke=True`, `active_rates=[0.10]`).
- v5 (patched from v4): STRIP now has a formula setting (addition primary, averaging logged as sensitivity), 250+250 per class (500 total), uint8 truncation, routing logged under both formulas, post-defense STRIP uses the primary formula. Only the 3 patched cells were syntax-checked; never run on a GPU.
- Defense configs need no change for WaNet; keep `fresh_select_k=1` unless every other attack is re-run. Pin the repo commit; thresholds stay frozen (0.40 / 30%).
- Known deviation to disclose: the trigger is applied after crop/flip augmentation (official WaNet likely warps first). It keeps train and test triggers aligned.
- Risk: WaNet (k=4, s=0.5) may not implant at 80 epochs, especially at 1% and maybe 5%. The `attack_ok` gate (ASR below 50%) logs this as a finding and skips defenses. Do not write expected WaNet results into the paper before measuring.
- Training epochs: BadNets 68; Blended, WaNet and SK 80. Disclose.
- SK settings (from its notebook): ASR measured on source class (cat), trigger L-inf 16/255, poison rates 0.5/1/5/10%.

## 9. Decisions and recommended final STRIP config

```
superimpose   = add          # clip(img + overlay, 0, 1), original STRIP
n_samples     = 500 total   # 250 triggered + 250 clean
n_superimpose = 100
FRR           = 1%          # threshold = normal fit on clean entropies, per checkpoint
overlay pool  = raw CIFAR-10 test images
trigger pool  = non-target images (asr_test_idx); SK also reported on the source-class pool
seed          = 2027; uint8 truncation after blending
log           = git hash + all parameters in every CSV row
```

- Do NOT use addition for BadNets and averaging for others: it looks like per-attack tuning, breaks the attack-agnostic claim (the controller cannot know the attack when running STRIP), and repeats the accidental mixing already in the table.
- Averaging is reported as a labeled sensitivity result. It is itself a finding: STRIP TPR on the same checkpoints swings from 0 to 65% depending on a one-line formula choice.
- `n_samples` is the least important parameter. Overlay count and the formula matter more.

## 10. Meaning of 'report an interval' for STRIP TPR

TPR is a proportion over 250 triggered inputs, so it has sampling error. A Wilson 95% interval gives a range (about +/-6 pp near 30% at n=250). Routing is robust only if the interval excludes the 30% cutoff. Under addition, BadNets 10% (34.8, \[29.2, 40.9\]) is borderline; under averaging Blended 1% is borderline. Say so explicitly rather than quoting one point. Note it ignores the extra uncertainty in the threshold estimate.

## 11. Open items and what can still be checked

| # | Item | Why | Effort (est.) |
| --- | --- | --- | --- |
| 1 | Find BadNets poison-index files and run AC with one explicit DR config | Controller AC values (0.32/0.37/0.44) cannot be tied to a version; BadNets 10% is 0.04 above threshold | 15-30 min if files exist |
| 2 | Settle PCA-2D vs ICA-10D, resolve `(False, 10)` ambiguity | Controller thresholds were calibrated on PCA-2D | 30 min |
| 3 | Replace SK AC values in the controller table (5%: 0.565, 10%: 0.44 degenerate); mark SK 1% AC unavailable | Old values likely from clean images | 1 h |
| 4 | Verify the saved SK poisoned-pixels file is the training set | Confirms the 0.565 | 30 min |
| 5 | Regenerate SK 1% poisoned pixels (optional) | Removes the missing signal | hours (est.) |
| 6 | Explain the Defense 1 TPRs (3.4 / 8.8) | Matches neither formula | 30 min |
| 7 | Fix `run_strip` docstring/comments; add a `superimpose` argument; log commit hash in every CSV | Prevent recurrence | 30 min |
| 8 | Rebuild controller inputs (STRIP add primary, avg sensitivity) and re-run `evaluate_controller.py` | Confirms my arithmetic in section 4 | 1-2 h |
| 9 | Re-run post-defense STRIP on defended checkpoints under the same formula | Pre/post must be comparable | 10 min GPU (est.) plus checkpoints must exist |
| 10 | Real timings per defense; remove cost fallbacks | Cost is the controller's selling point | 30 min |
| 11 | Re-frame the Blended 10% STRIP finding as formula-dependent | Avoid an overclaim | writing only |
| 12 | BaEraser k=3 on all cells (optional) | Removes the smoke-setting objection | 2-4 h GPU (est.) |
| 13 | WaNet: smoke test, then full run | Held-out attack | see section 12 |
| 14 | Evaluation-seed repeats of STRIP on borderline conditions (not training seeds) | Mean +/- SD of TPR | under 1 h GPU (est.) |

## 12. Plans by time budget (time numbers are estimates)

### 12.1 Very little time (about 1-2 days)

1. Lock the STRIP config in section 9; fix the docstring; log the hash.
2. From the existing re-measure CSVs, build the controller inputs table under addition (primary) and averaging (sensitivity). No new GPU work.
3. Replace SK AC values with the re-measured ones; mark SK 1% AC unavailable; state BadNets AC provenance (attack notebook config) if the files cannot be found.
4. Re-run `evaluate_controller.py` on the existing 36-cell matrix; report the controller, Always-X, oracle table with ASR, delta-CA and cost as separate columns; report Wilson intervals for TPR.
5. Disclose: Blended 10% BaEraser selection failure (k=1, oracle shown separately as diagnostic), SK persistent failure, formula sensitivity, single seed, CIFAR-10 only, training-epoch differences.
6. Skip WaNet and BaEraser k=3. Frame the controller as a cost-aware router with documented failure modes.

### 12.2 Moderate time (about 1 week)

Everything in 12.1, plus:

- BadNets AC with an explicit config; post-defense STRIP under the common formula.
- WaNet smoke test, then 5% and 10% only (about 3.5 h total GPU, est.; run one rate per Colab session). If it does not implant, report the attack failure.
- Evaluation-seed repeats of STRIP on borderline conditions.

### 12.3 Plenty of time

Everything in 12.2, plus:

- WaNet at 1% too.
- BaEraser k=3 on all cells (k=1 kept in the appendix).
- A second training seed on a few conditions, or one more held-out attack.
- Regenerate SK 1% poisoned pixels.
- Grad-CAM corroboration for WaNet, where STRIP is expected to be weak.

### 12.4 What matters most for a decent paper

- Consistent detector inputs (one STRIP config, one AC pipeline) is the highest priority; it removes a reviewer-visible inconsistency.
- A held-out attack (WaNet) is the most valuable optional addition; BaEraser k=3 is the least.
- Workshop level: the 12.1 plan is enough. Mid-tier venues will likely push on 3 attacks, 1 seed, 1 dataset, and calibrating and evaluating on the same conditions.

## 13. Claims: safe and unsafe

Safe: 'near-ANP security at lower compute on the solvable conditions, with documented failures'; 'STRIP TPR is sensitive to the superimposition formula (0 to 65% on the same checkpoint)'; 'SK defeats all tested lightweight defenses'; 'AC is blind to SK only at low poison rates'; 'all results are single-seed (2027)'.

Unsafe: '0/6 failures' before the corrected inputs are re-run; 'STRIP is blind to Blended at 10%' without naming the formula; 'the controller generalizes to unseen attacks' before WaNet (or another held-out attack) is measured; 'cost-aware optimization' (cost barely affects the formal score at the current weights; report cost as a separate column).

## 14. What I have not verified

- I have not seen the BadNets, Blended or SK notebooks' STRIP cells; the configuration table in 1.1 comes from a separate review.
- I have not run any of these notebooks on a GPU, including WaNet v5.
- Routing outcome numbers in section 4 and the cost percentages are my arithmetic from the matrix and re-measure outputs, not outputs of `evaluate_controller.py`.
- Whether each past notebook used addition or averaging is inferred from fingerprints, not from commit hashes.
