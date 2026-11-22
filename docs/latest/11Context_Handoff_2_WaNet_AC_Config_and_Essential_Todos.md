# Context Handoff 2: WaNet Results, AC Config Decision, BaEraser Control, Essential To-Dos

Status as of Nov 23, 2026. Companion to `Context_Handoff_STRIP_AC_Standardization__Controller_Inputs_and_WaNet_v2.md` (call it "v2"). v2 covers the STRIP formula issue, the Blended/SK/BadNets re-measures and the routing table. This doc covers everything after v2: the PCA vs ICA decision, threshold provenance, SK AC input fix, the executed WaNet run, the BaEraser clean-model control, and a trimmed to-do list. Items marked (est.) are estimates. Items marked (inferred) are my reading of logs, not confirmed against code. I have not seen the repo code, only logs and CSV output.

## 0. TL;DR

- **AC config decision: use PCA-2D everywhere.** ICA-10D silhouettes (0.08-0.29 across all attacks) never reach tau_ac = 0.40 and do not order conditions consistently. Do not re-tune a threshold for ICA. Report ICA numbers as an appendix sensitivity table only.
- **Threshold provenance is unknown.** The 0.40 (AC) and 30% (STRIP) cutoffs are not documented anywhere I have seen. "Calibrated on PCA-2D" was my inference. If they were picked after looking at BadNets/Blended/SK values, they were tuned in-sample, and WaNet is the first out-of-sample test.
- **WaNet implanted** (ASR 92.4 / 98.8 / 99.5%, CA 94.3-94.9%). STRIP is blind to it (TPR 0.4% under addition, 0.0% under averaging). The controller routed all three conditions to ANP, which matched the oracle (ASR 0.58 / 0.15 / 0.53%).
- **That match is close to trivial.** On WaNet the controller behaves like Always-ANP, so there are no cost savings. The honest claim is "it did not route to a cheap defense that fails", not "it saves cost on unseen attacks".
- **BaEraser recovery is not producing false positives.** On the clean model it accepted 0 of 7 candidates (ASR 0.91-1.14%). On the WaNet models it accepted 2 each at 5% and 10% (ASR 97-99%). The failure on WaNet (post-defense ASR 23% / 62%) is downstream of recovery, in unlearning or checkpoint selection (inferred, not confirmed).
- **Things to check before reporting:** FT is identical to the undefended model in all 3 WaNet cells (possible no-op); pr01 defenses were loaded from cache, not run; NAD collapsed at 5% and 10%; the repo commit hash was not logged.

## 1. Status of v2 open items

| v2 # | Item | Status now |
| --- | --- | --- |
| 1 | BadNets AC re-measure | Done (v2 section 2.5) |
| 2 | PCA-2D vs ICA-10D | Decided: PCA-2D. Still to apply in the repo (to-do 1 below) |
| 3 | Replace SK AC values | Essential, to do (to-do 2) |
| 4 | Verify SK poisoned-pixels file | Essential, to do (part of to-do 2) |
| 5 | Regenerate SK 1% poisoned pixels | Dropped (not needed, see 4.2) |
| 6 | Explain Defense 1 TPRs (3.4 / 8.8) | Unchanged, low priority |
| 7 | Fix `run_strip` docstring, add `superimpose`, log hash | Essential, to do (to-do 3) |
| 8 | Rebuild controller inputs, re-run `evaluate_controller.py` | Essential, to do (to-do 5; now includes WaNet) |
| 9 | Post-defense STRIP under one formula | Unchanged; WaNet's extended CSV already has post-defense STRIP fields |
| 10 | Real timings, remove fallbacks | Essential, to do (to-do 6) |
| 11 | Reframe Blended 10% STRIP as formula-dependent | Writing only (to-do 8) |
| 12 | BaEraser k=3 on all cells | Dropped (optional) |
| 13 | WaNet smoke test + full run | Done (section 3). v2's "may not implant" risk did not materialize |
| 14 | Evaluation-seed repeats of STRIP | Not pursued (scope) |

## 2. Decisions made since v2

### 2.1 AC config: PCA-2D, not ICA-10D

Evidence across all attacks (silhouette, PDR):

| Attack / rate | PCA-2D silhouette | PCA-2D PDR (%) | ICA-10D silhouette | ICA-10D PDR (%) |
| --- | --- | --- | --- | --- |
| BadNets 1% | 0.3206 | 9.72 | 0.0774 | 17.13 |
| BadNets 5% | 0.3746 | 75.97 | 0.1184 | 99.75 |
| BadNets 10% | 0.4394 | 97.15 | 0.1017 | 99.59 |
| Blended 1% | 0.3778 | 11.16 | 0.1102 | n/a here |
| Blended 5% | 0.5408 | 99.76 | 0.0983 | n/a here |
| Blended 10% | 0.6380 | 99.96 | 0.1744 | n/a here |
| WaNet 1% | 0.4416 | 9.19 | 0.2874 | 99.59 |
| WaNet 5% | 0.3513 | 69.60 | 0.0970 | 94.15 |
| WaNet 10% | 0.4590 | 91.98 | 0.1050 | 94.91 |

Reasons for PCA-2D:

1. ICA-10D silhouettes (0.08-0.29) are all below tau_ac = 0.40, so AC would read LOW for every condition and carry no routing information.
2. ICA silhouettes do not order conditions: for example BadNets 1% (PDR 17%) is 0.077 and WaNet 5% (PDR 94%) is 0.097.
3. Re-tuning a threshold for ICA now means choosing it after seeing the data, and it would invalidate every AC input already verified (all PCA-2D).
4. PCA is deterministic; ICA depends on its random initialization.
5. One config for all attacks keeps the attack-agnostic claim. The controller cannot know the attack in advance.

Honest caveat for the write-up: ICA-10D is the better detector (PDR is high almost everywhere). PCA-2D silhouette is only a routing gate, and it tracks detection quality poorly (WaNet 1% is HIGH with PDR 9%; WaNet 5% is LOW with recall 99%). The justification is "this is the feature the controller's thresholds were built on", not "PCA finds poison better".

Repo action: `ATTACK_DR_CONFIG['badnets']` is currently `(False, 10)` (ICA-10D) and Blended is `None` (falls back to the ICA-10D default). Set all attacks, including the fallback default, to PCA-2D (likely `(True, 2)`; confirm from the `run_ac` signature, since `(False, 10)` meaning ICA-10D implies the tuple is `(use_pca, n_components)`). The WaNet notebook already computed both and used PCA-2D for the controller.

### 2.2 Where 0.40 and 30% came from

Not recorded in v2 or in anything I have seen. What I can say:

- Every AC value in the controller table is a PCA-2D silhouette, which is why I assumed PCA-2D calibration.
- 0.40 sits in the gap between the earlier attacks' values (0.375 and 0.378 below, 0.439 and 0.442 above). That pattern is also what in-sample selection would produce.
- v2 section 12.4 already names "calibrating and evaluating on the same conditions" as a likely reviewer objection.
- The quadrant map (AC high/STRIP high to BaEraser, and so on) was written in design docs before the matrix existed (v2 section 6), which is separate from where the numeric cutoffs came from.

Action: find where 0.40 and 30% first appear (git history, design doc, paper draft) and whether that predates the BadNets/Blended/SK measurements. Write "fixed a priori" only if that can be shown. Otherwise write that the thresholds were set on the earlier attacks and WaNet is the held-out check.

### 2.3 SK AC inputs

Replace the controller's old SK AC values (0.33 / 0.33 / 0.37, probably computed on clean images because the SK attack notebook's poison tensor was all zeros) with the rebuilt-set values:

| SK rate | AC silhouette (PCA-2D) | PDR | Notes |
| --- | --- | --- | --- |
| 0.5% | 0.3317 | 7.46 | recall 83.2 |
| 1% | unavailable | n/a | poisoned pixels not saved; do not interpolate |
| 5% | 0.5652 | 99.10 | recall 97.2; flips AC from LOW to HIGH |
| 10% | 0.4416 | 100 | degenerate (whole target class poisoned); keep with a flag |

Why it matters: with the stale 0.37, SK 5% (STRIP add 86.8%) routes AC-LOW/STRIP-HIGH to NAD. With 0.565 it routes AC-HIGH/STRIP-HIGH to BaEraser (which failed). The correction changes a route and the claim "AC is blind to SK" (true at 0.5%, not at 5%).

SK 1%: STRIP is 8.4% (LOW), and both AC-HIGH/STRIP-LOW and AC-LOW/STRIP-LOW route to ANP. So the missing AC value does not change SK 1%'s route. Route it as ANP and footnote that AC is unavailable. This is why regenerating the 1% pixels is not needed.

The 0.565 still depends on v2 open item 4: confirm the saved SK pixel file is the set that trained the checkpoint (check that the SK checkpoint predicts the target class on those images at a high rate, as was done for BadNets at 95-99%).

## 3. WaNet results (executed run)

Run facts from the log: Kaggle (`/kaggle/working`), outputs in `ps-capstone4/wanet_v3/paper`, seed 2027, mode PAPER, 80 epochs, rates 1/5/10%. The log shows the STRIP formula-sensitivity lines, consistent with the v5 patch (output folder name says v3). The repo commit hash was not printed. Split: 2000 defense / 500 selection / 7500 report; the shared Blended split cross-check passed. Poison counts 500 / 2500 / 5000 (cross 1000 / 5000 / 10000 as logged). Non-target triggered report pool: 6750. WaNet operator vs attack warp max|diff| 3.52e-06; repo metrics match the notebook (CA 94.88, ASR 92.39).

Stealth (1000 report images): PSNR mean 29.2 (min 23.6), Linf mean 0.221 (max 0.411).

### 3.1 Attack validity

| Rate | CA | ASR | Noise-ASR | Target-class acc | Clean to airplane | attack_ok |
| --- | --- | --- | --- | --- | --- | --- |
| 1% | 94.88 | 92.39 | 0.19 | 95.07 | 0.58 | True |
| 5% | 94.55 | 98.79 | 0.10 | 95.87 | 0.61 | True |
| 10% | 94.33 | 99.50 | 0.10 | 96.13 | 0.74 | True |

Clean-model floor: CA 92.48, WaNet-trigger ASR 0.70, noise-ASR 2.47. The attack is trigger-specific (noise-ASR about 0.1-0.2%). Note the clean model's CA is 2+ points below the poisoned models', so use the undefended backdoored model as the CA reference for delta-CA, not the clean model.

### 3.2 Activation Clustering (predicted-class group, target class 0)

| Rate | Target group | Poisoned in group | DR | Silhouette | Clusters | Susp. frac | PDR % | Recall % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | 5,496 | 496 | PCA-2D | 0.4416 | 1229 / 4267 | 0.2236 | 9.19 | 22.78 |
| 1% | | | ICA-10D | 0.2874 | 5003 / 493 | 0.0897 | 99.59 | 98.99 |
| 5% | 7,500 | 2,497 | PCA-2D | 0.3513 | 3559 / 3941 | 0.4745 | 69.60 | 99.20 |
| 5% | | | ICA-10D | 0.0970 | 4868 / 2632 | 0.3509 | 94.15 | 99.24 |
| 10% | 10,002 | 4,994 | PCA-2D | 0.4590 | 5421 / 4581 | 0.4580 | 91.98 | 99.84 |
| 10% | | | ICA-10D | 0.1050 | 4759 / 5243 | 0.4758 | 94.91 | 99.64 |

With the frozen tau_ac = 0.40: AC reads HIGH / LOW / HIGH at 1 / 5 / 10%. At 1% PCA-2D flags the smaller cluster (1229) containing only about 113 poisoned images, while ICA-10D isolates 493 images of which 491 are poisoned. The silhouette gate is therefore a poor proxy for detection quality on WaNet (HIGH at 1% with PDR 9%; LOW at 5% with recall 99%).

### 3.3 STRIP (250 triggered + 250 clean, overlay pool 1500)

| Rate | Formula | TPR % | FPR % | AUROC | Clean mu | Trigger mu | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | add | 0.4 | 1.2 | 0.446 | 0.421 | 0.446 | 0.1192 |
| 1% | avg | 0.0 | 1.2 | 0.349 | | | |
| 5% | add | 0.4 | 2.0 | 0.453 | 0.455 | 0.481 | 0.1306 |
| 5% | avg | 0.0 | 2.0 | 0.404 | | | |
| 10% | add | 0.4 | 1.2 | 0.451 | 0.452 | 0.475 | 0.1486 |
| 10% | avg | 0.0 | 1.2 | 0.393 | | | |

STRIP is blind to WaNet under both formulas: triggered inputs have slightly higher entropy than clean ones (AUROC below 0.5), so the formula sensitivity seen on Blended does not appear here. TPR 1/250 has a Wilson 95% interval of about 0.1-2.2% (my arithmetic), far below the 30% cutoff.

### 3.4 Controller decisions (thresholds frozen at 0.40 / 30%)

| Rate | Quadrant | Route (add and avg) |
| --- | --- | --- |
| 1% | AC-HIGH / STRIP-LOW | ANP |
| 5% | AC-LOW / STRIP-LOW | ANP (policy fallback: low evidence does not mean low severity) |
| 10% | AC-HIGH / STRIP-LOW | ANP |

The 5% route is the policy fallback, not evidence from the signals. Do not quote the controller's rule text for 1% ("structurally embedded") as an explanation, since AC's cluster at 1% mostly does not contain the poison.

### 3.5 Defense matrix

| Rate | FT ASR | ANP ASR | NAD ASR | BaEraser-lite ASR | Oracle |
| --- | --- | --- | --- | --- | --- |
| 1% | 92.39 | 0.58 | 1.26 | 2.68 | ANP |
| 5% | 98.79 | 0.15 | 1.02 (invalid_collapse) | 23.16 | ANP |
| 10% | 99.50 | 0.53 | 0.73 (invalid_collapse) | 62.16 | ANP |

CA after defense (undefended in brackets): ANP 90.85 / 90.27 / 91.20 (94.88 / 94.55 / 94.33), a cost of 3.1-4.3 points. NAD 91.29 / 89.67 / 89.41. BaEraser-lite 92.85 / 91.69 / 90.89. FT 94.88 / 94.55 / 94.33.

Always-X mean ASR over valid cells: ANP 0.42 (n=3), BaEraser-lite 29.33 (n=3), FT 96.89 (n=3), NAD 1.26 (n=1, the other two are invalid, so do not compare it with the others).

Measured timings (Kaggle): ANP about 643 s (5%) and 642 s (10%); NAD about 128 s and 127 s; BaEraser-lite about 180 s each; FT about 8 s (10% only). All four pr01 defenses were loaded from cached checkpoints, so they have no timings, and pr05 FT was also loaded. Cost looks rate-independent.

### 3.6 Did WaNet strengthen the paper?

Yes, modestly.

- It adds a held-out attack with frozen thresholds, the biggest missing piece in v2 section 12.4. It implanted without tuning.
- The controller matched the oracle on 3 of 3 and did not route to a defense that fails (FT does nothing; BaEraser-lite leaves 23-62% ASR).
- It does not show cost savings (controller equals Always-ANP here) and it does not show the detector signals generalize (STRIP uninformative, AC silhouette a poor proxy). The success comes from the fallback rule.

Claim to use: "On a held-out attack with frozen thresholds, the controller's fallback kept it on the one robust defense, at no cost savings." Do not claim it "generalizes" more broadly.

## 4. Issues found in the WaNet run

### 4.1 To check before reporting

1. **FT is identical to the undefended model in all three cells** (CA and ASR match the base models exactly, including pr10 where FT ran fresh in 8 s). A fine-tune on 2000 images should shift at least a few predictions. It may be a no-op or an evaluation bug (for example evaluating the original model object). Check by comparing weights between `resnet18_wanet_pr10.pth` and `wanet_pr10_FT_defended.pth` (`torch.equal` over the state dict). If identical, FT is not a real baseline on any attack and "Always-FT" needs a second look.
2. **pr01 defenses were loaded from cached checkpoints** (`loaded_existing_checkpoint` for FT, ANP, NAD, BaEraser), so provenance and code version are unknown and there are no timings. This also explains the NaN recovery fields for pr01 in the extended CSV (no recovery ran in this session). Either state that pr01 is cached, or delete the four pr01 `*_defended.pth` files and re-run (about 16 min GPU by the measured timings, est.).
3. **Repo commit hash not logged** in the WaNet run. Record it in the write-up if recoverable from the clone time, otherwise disclose.
4. **Hardware differs.** WaNet ran on Kaggle; earlier matrix timings (ANP about 750 s, NAD about 256 s, BaEraser about 216 s, FT about 18 s) came from another setup. Do not mix them in one cost table without saying so.

### 4.2 To disclose (not to fix)

- **NAD teacher collapsed at 5% and 10%** (held-out CA dropped 0.942 to 0.888 and 0.938 to 0.878 at lr 0.01; the repo warning says to lower the lr). Both cells are marked invalid. Changing the lr only for WaNet would be per-attack tuning.
- **BaEraser-lite leaves residual ASR** (23.16% at 5%, 62.16% at 10%) despite recovering triggers (section 5). Likely cause is unlearning or the k=1 proxy-ASR checkpoint selection (v2 section 7); not confirmed. `best_epoch` is 15 for both; check whether 15 is the last epoch of the schedule (read the config, no run needed).
- **ANP costs 3-4 CA points** at its 0.15-0.58% ASR; report delta-CA next to it.
- Single seed 2027, CIFAR-10 only, WaNet trigger applied after crop/flip (v2 section 8).

## 5. BaEraser recovery control (clean model vs WaNet models)

### 5.1 Clean-model control (Colab, `resnet18_clean_seed2027.pth`)

BaEraser-lite recovery with the WaNet family was run on the clean model: 7 epsilons (0.20, 0.50, 0.80, 0.35, 0.65, 0.95, 0.05), all rejected, candidate ASR 0.91-1.14%, 0 accepted, 78.8 s, `eval_split: heldout`, `init_mode: random`, `fresh_init_verified: True`, target label 0. Candidate metadata shows `family: wanet_warp, k: 4, s: 0.5, grid_rescale: 1.0`.

Held-out evaluation set size is about 440 images (inferred: 0.909% = 4/440 and 1.136% = 5/440). The run printed "Selection: 500"; the evaluation set is smaller than that.

### 5.2 Recovery on the WaNet models

From the WaNet log and `wanet_defense_matrix_extended.csv`:

| Rate | Accepted triggers | Candidate ASR (accepted) | Recovery time | Unlearning time | Best epoch | Fresh init |
| --- | --- | --- | --- | --- | --- | --- |
| 1% | n/a (cached) | n/a | n/a | n/a | n/a | n/a |
| 5% | 2 | 97.07% (eps 0.20), 99.32% (eps 0.50) | 25.96 s | 80.20 s | 15 | True |
| 10% | 2 | 99.55% (eps 0.20), 99.10% (eps 0.50) | 25.57 s | 80.06 s | 15 | True |

Current module config: epsilons [0.2, 0.5, 0.8], fine epsilons [0.35, 0.65, 0.95, 0.05], 6 epochs, 24 candidates, stop after 2 accepted. Constants: `BAERASER_RECOVERY_ASR_THRESHOLD = 0.8`, `BAERASER_ASR_STOP_THRESHOLD = 0.01`.

### 5.3 Interpretation

- Clean model: 0 of 7 accepted at about 1% ASR. WaNet models: 2 accepted at 97-99%. The earlier hypothesis that BaEraser's recovered "triggers" were generic adversarial perturbations is not supported and is withdrawn.
- Equivalence of historical and clean-run settings is not fully provable (no saved recovery metadata JSON; the CSV keeps only `fresh_init_verified`). The risk is low: the log lines and first two coarse epsilons match, `fresh_init_verified` is True for pr05/pr10, and the gap (1.14% vs 97%) is far wider than any difference between the 0.01 and 0.8 acceptance constants. Do not re-run pr05/pr10 recovery for this.
- Specificity caveat: the control covers one clean model, one random init per epsilon and about 440 held-out images. Write "no false-positive trigger recovery observed", not anything stronger.
- Recovery succeeds on WaNet, so the BaEraser failure is downstream (unlearning or selection). Report it as "recovery succeeds, unlearning leaves residual ASR".
- Disclosure: the recovery candidates are in a `wanet_warp` family with k and s values, which suggests BaEraser is given the trigger family. Read the adapter code to confirm, and check whether the BadNets and Blended adapters are family-specific in the same way. If so, say that BaEraser is given the attack family.

## 6. Claims: additions to v2 section 13

Safe:

- "AC config is PCA-2D for all attacks; ICA-10D is reported as a sensitivity (it never exceeds the 0.40 gate)."
- "On the held-out WaNet attack with frozen thresholds, STRIP TPR is 0.4% under both formulas and the controller routed every condition to ANP, which was the oracle."
- "BaEraser recovery found no trigger on the clean model (0 of 7 epsilons, candidate ASR at most 1.14%) and found 2 on each of the WaNet 5% and 10% models."
- "BaEraser-lite recovers WaNet triggers but leaves 23% and 62% residual ASR after unlearning."

Unsafe:

- "The controller generalizes to unseen attacks" or any WaNet cost-saving claim (it equals Always-ANP there).
- "AC silhouette predicts detection quality" (WaNet 1% HIGH with PDR 9%; 5% LOW with recall 99%).
- "Thresholds were fixed a priori" unless provenance (section 2.2) is shown.
- "FT fails on WaNet" before the weight comparison in section 4.1.
- "BaEraser is blind to the attack" or "BaEraser finds the trigger without attack knowledge", given the family-specific candidates (section 5.3).
- Any comparison of Always-NAD with other baselines on WaNet (n=1 valid cell).

## 7. Essential to-dos (trimmed list)

Estimates are rough (est.).

1. **Lock PCA-2D in the repo (30 min).** Set `ATTACK_DR_CONFIG` for every attack and the fallback to PCA-2D. Test through the controller's own path (no explicit DR argument): BadNets 10% should give 0.4394, Blended 5% 0.5408, SK 5% 0.5652. Commit and record the hash.
2. **Fix the SK AC inputs and verify the pixel file (about 1 h).** Use the section 2.3 values; mark SK 1% unavailable (it routes ANP regardless) and SK 10% degenerate. Confirm the saved SK pixel file is the training set.
3. **Fix the STRIP code (30 min).** Add a `superimpose` argument (`add`: `np.clip(img + clean, 0, 1)`; `avg`: `np.clip((img + clean) / 2, 0, 1)`) to `strip_entropy_single` and `run_strip`, fix the docstring, set defaults to 500 samples / 100 overlays / FRR 0.01, and write the formula and `git rev-parse HEAD` into every CSV row. Regression check on Blended pr05 at 100 overlays: about 92.0% (add) and about 45.2% (avg).
4. **FT no-op check (10 min).** Weight comparison per section 4.1 item 1. Decide what to do with the FT baseline based on the result.
5. **Rebuild controller inputs and re-run (1-2 h).** Two tables (addition primary, averaging sensitivity) from the existing re-measure CSVs, with the WaNet rows added (AC 0.442 / 0.351 / 0.459; STRIP 0.4% add, 0.0% avg) and the 12 WaNet defense cells from the log. Run `evaluate_controller.py` once per formula with thresholds frozen. Mark the two NAD collapses as invalid, not as numeric results. Add Wilson 95% intervals for TPR (`statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`). If the script output differs from v2 section 4's arithmetic, trust the script and fix the doc.
6. **Real timings (30 min).** Make missing timings raise an error instead of using cost fallbacks. Use the measured WaNet times (section 3.5) and state that cost looks rate-independent (ANP 643 s vs 642 s), or re-run the pr01 defenses to get measured times and fix provenance together. Do not mix Kaggle and earlier-setup timings in one table without saying so.
7. **BaEraser read-only checks (15-20 min, no runs).** (a) Is the recovery family-specific, and do the BadNets/Blended adapters work the same way? (b) Is `best_epoch` 15 the last epoch of the schedule? Disclose both.
8. **Write-up.** Reframe Blended 10% STRIP as "65.2% under addition, 0% under averaging". Report the controller, Always-X and oracle table with ASR, delta-CA and cost as separate columns. Include the disclosures from section 4.2 and v2 section 12.1 step 5, the threshold provenance statement from section 2.2, and the WaNet claim wording from section 3.6.

Explicitly not doing (not essential): re-tuning thresholds for ICA, more seeds, changing the NAD lr, BaEraser k=3, re-running pr05/pr10 recovery, regenerating SK 1% pixels, Grad-CAM, any WaNet-specific tuning.

## 8. What I have not verified

- I have not seen the repo code, only logs and CSV output. The `(use_pca, n_components)` tuple meaning, the BaEraser family-specificity and the cost fallbacks are inferred or unchecked.
- The FT no-op is a suspicion from identical numbers, not a confirmed bug.
- The clean-control equivalence argument rests on matching log formats, matching first epsilons, `fresh_init_verified` and the size of the ASR gap, not on saved metadata.
- The about 440 held-out evaluation size, the reading of "cross" counts and the Wilson interval for 1/250 are my arithmetic or inference.
- Whether the WaNet run used exactly the v5 notebook is inferred from the formula-sensitivity log lines; the output folder is named `wanet_v3`.
- Threshold provenance (section 2.2) is unresolved.
