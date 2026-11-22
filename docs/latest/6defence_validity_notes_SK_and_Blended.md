# Defence results: validity notes for Silent Killer (SK) and Blended (v4_1 / v3 re-run)

Source runs: `silentkiller_defense_combined_v4_1` (executed, no errors) and `blended_defense2_baeraser_v3` (re-run with the
clean-model control). Blended defence1 (FT/ANP/NAD) was not re-run; its numbers are unchanged from the earlier upload.
This file replaces the earlier version: the SK NAD numbers changed in the re-run.

## 1. What is safe to quote, and how

| Result | Quote? | Note |
|---|---|---|
| SK baselines, FT, ANP, ANP+FT, matched FT | Yes | Reproduction gate passed (B3). |
| SK clean-model control (NAD, matched FT) | Yes, and required | Defines the reference for every SK claim. |
| SK NAD | Only as "teacher still backdoored, not interpretable" | All three NAD rows are flagged (teacher ASR 60.0 / 56.8 / 90.4%). Quote the residual against the clean model, not the raw ASR. |
| SK BaEraser | Yes, as a failure | pr01/pr05: no valid checkpoint; pr10: ASR 83.6%, CA 77.9%. |
| SK pr01 Activation Clustering | Unavailable | The pilot never saved its poisoned pixels. State as a limitation. |
| Blended FT, ANP, ANP+FT, matched FT | Yes | From defence1 (not re-run). |
| Blended NAD | pr05 yes; pr01 not interpretable; pr10 invalid | pr10 failed the collapse gate. |
| Blended BaEraser true-trigger ASR | Yes | Describe the mechanism as generic perturbation unlearning, not key recovery (clean control, below). |
| BaEraser selection (both attacks) | Say it was a smoke setting | `FRESH_SELECT_K = 1`. Re-run with 3 only if BaEraser becomes a headline result. |

## 2. Silent Killer

Baselines (report half): CA 91.88 / 90.91 / 81.60%, ASR_src 89.33 / 100 / 100% for pr01 / pr05 / pr10.
Clean model: CA 92.47%, ASR_src 67.73% (transfer floor).

**The transfer floor is not a stable reference.** The same defences applied to the never-poisoned model cut its
ASR_src from 67.7% to **24.8% (NAD)** and **10.9% (matched FT)**. So the right reference for a defended poisoned model is
the clean model under the same defence, and the right metric is the residual above it.

| Defence | pr01 / pr05 / pr10 ASR_src after | Clean model, same defence | Residual above clean (pp) |
|---|---|---|---|
| Plain FT (lr 1e-4) | 81.6 / 100 / 100 | not run | n/a, ineffective |
| ANP | 85.7 / 100 / 100 | not run | n/a, ineffective (pr10: nothing pruned) |
| ANP+FT | 74.3 / 100 / 99.5 | not run | n/a, ineffective |
| Matched FT | 45.5 / 99.2 / 94.4 | 10.9 | +34.5 / +88.3 / +83.5 |
| NAD (teacher still backdoored) | 45.6 / 71.1 / 60.9 | 24.8 | +20.8 / +46.3 / +36.1 |
| BaEraser | failed / failed / 83.6 | not applicable | failure |

**Reading.** No tested defence removes the SK backdoor. NAD leaves the smallest residual, but its teacher is still
backdoored in every case, so it cannot be presented as a working defence. Everything else leaves 35 to 88pp above
the clean reference or does nothing.

**Limitations to state.**
1. ASR drops toward the floor are not backdoor removal (clean control above).
2. BaEraser recovers generic target-class perturbations: Pearson against the true trigger about 0 (range -0.03 to 0.10),
   sign agreement about 0.5. The clean model is as recoverable as the poisoned ones (candidate ASR 99.1%, fresh audit 99.55%,
   UAP 98.9%). At pr10 the fresh audit falls to 0.0% after sanitising while UAP stays at 99.97%, which points to gradient
   masking, not robustness.
3. STRIP cannot verify removal on SK (it stays blind at pr01, and TPR can collapse while ASR stays high). Use ASR-style audits.
4. pr10 is degenerate (the whole target class is perturbed; airplane accuracy is very low); CA "gains" after FT are airplane recovery.
5. Overall CA hides class damage (for example cat accuracy under ANP and NAD).
6. pr01 AC is unavailable. For pr05 and pr10, AC is nearly identical under predicted-class and label grouping, so the
   grouping choice does not change conclusions. Pick one and say so.
7. The notebook's own verdict labels are threshold-based ("INEFFECTIVE" for NAD pr01 at 45.6% vs a 44.7% cut-off). Quote
   residuals, not the labels.

## 3. Blended (alpha = 0.1, target class 0)

Baseline: CA 94.95 / 94.81 / 94.48%, ASR 99.99 / 100 / 100%. Clean model: CA 92.48% (report half), blend-trigger ASR 0.46%.

| Defence | pr01 / pr05 / pr10 ASR after | CA after | Reading |
|---|---|---|---|
| Plain FT | 99.99 / 100 / 100 | 94.2 / 93.7 / 93.0 | No effect |
| ANP | 0.06 / 1.9 / 0.2 | 90.0 / 92.9 / 91.6 | Works; pr01 CA drop 5.00pp (at budget) |
| ANP+FT | 1.3 / 3.8 / 0.15 | 93.5 / 94.0 / 93.2 | Works, CA mostly recovered |
| NAD | 0.74 / 0.22 / 0.21 | 90.3 / 90.8 / 89.3 | pr01 teacher ASR 69% (not interpretable); pr10 invalid (collapse gate) |
| Matched FT | 48.6 / 97.6 / 0.28 | 89.7 / 91.3 / 89.7 | NAD adds value at pr01 and pr05, none at pr10 |
| BaEraser | 0.15 / 1.3 / 16.7 | 92.0 / 91.2 / 92.8 | Works at pr01/pr05; pr10 selection failure (selected epoch 1; oracle epoch 15 gives 0.46%) |

**Clean-model control for BaEraser (now run).** The clean model accepts recovered triggers too (2/2 accepted, candidate
ASR 97.5%, fresh audit 96.6%, UAP 76%), the same as the backdoored models. So the recovery is finding generic universal
perturbations, not the blend key. Report BaEraser as "perturbation-based unlearning that lowers true-trigger ASR", not as
key recovery. After sanitising, fresh-trigger and UAP ASR stay at about 100%, so generic robustness did not improve.

**Other limitations.** STRIP baseline TPR differs across notebooks (28/41/0, 3.4/8.8/0 and 10/24/0% for
pr01/pr05/pr10): choose one configuration and report only it. Grad-CAM correlation has low valid-sample counts at baseline.

## 4. Suggested paper wording

- "We report Silent Killer ASR against the clean model under the same defence. Defences also lower a never-poisoned
  model's transfer ASR (67.7% to 24.8% for NAD and 10.9% for matched fine-tuning), so raw ASR drops are not interpreted as
  removal."
- "No evaluated lightweight defence removes the Silent Killer backdoor. The best, NAD, leaves 21 to 46pp above the clean
  reference and its teacher remains backdoored."
- "Runs that failed the collapse gate or whose teacher remained backdoored are marked invalid or non-interpretable."
- "BaEraser-lite recovers generic target-class perturbations on both attacks (the clean model accepts them equally); we
  report true-trigger ASR only."
- "Activation Clustering is unavailable for SK at 1% because the pilot run did not store its poisoned pixels."

## 5. What the controller should take from this

- **Blended:** route to ANP or ANP+FT first (cheap and effective at all rates), NAD as the alternative.
- **SK:** no tested lightweight defence succeeds. Do not route to "NAD works". Use NAD as the least-bad option only with an
  explicit "residual above clean reference" check, and escalate (retrain or reject the model) when the residual stays high.
- **Verification:** ASR-style audits against the clean model under the same defence, not STRIP.
- **Inputs:** the clean-model floor, and the clean model under each candidate defence, as controller inputs.
- **Missing or unreliable signals** (pr01 AC, STRIP on SK, STRIP on Blended) must be handled explicitly.
