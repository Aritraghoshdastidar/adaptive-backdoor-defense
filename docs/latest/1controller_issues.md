Let me look at the formal score computation and calibration code to finish my analysis:

Viewed controller.py:195-310
Viewed controller.py:310-370

Good, I have everything I need. Here's my deep analysis:

---

## Issue 4: STRIP TPR Needs Triggered Inputs — Best Solution

**The problem in full:** `run_strip()` in [detection.py L327–428](file:///c:/Users/Asus/Desktop/PESU/capstone/sem7/repo/adaptive-backdoor-defense/core/detection.py#L327-L428) takes `trigger_fn` and `asr_test_idx` — both require knowing the trigger. It then computes TPR = (flagged triggered) / (total triggered). A real defender doesn't have labeled triggered inputs, so feeding TPR into `controller_decide()` is not deployable.

**The options I considered:**

| Option | What changes | Deployable? | Keeps 2×2 quadrant? | Effort |
|--------|-------------|:-----------:|:-------------------:|:------:|
| A. Use AC-flagged samples as STRIP probes | Run STRIP on AC's minority cluster | ✅ | Partially — breaks when AC is blind | High (re-run everything) |
| B. Use entropy gap (μ_clean − μ_triggered) | Replace binary TPR with continuous signal | ❌ Same problem | ✅ | Medium |
| C. Drop STRIP from controller input entirely | Controller = AC-only (1D, not 2×2) | ✅ | ❌ Kills the paper's structure | Low |
| D. Keep TPR, state the assumption | Add one sentence to threat model | ❌ But honest | ✅ | Zero |
| E. Use STRIP's clean-side statistics | μ_clean, σ_clean as controller signal | ✅ | Debatable — weak signal | Medium |

**My recommendation: Option D, with Option A as a "future work" note.**

Here's why:

1. **The pipeline isn't a live deployment system anymore — it's a research framework.** You explicitly pivoted from product to paper. Reviewers expect a clean experimental evaluation, not a production-deployable tool.

2. **The assumption is standard in the literature.** STRIP's original paper (Gao et al., 2019) evaluates TPR using known triggers. NAD, ANP, Neural Cleanse — all defense papers use known triggers for ASR/TPR evaluation. No one calls this unfair because the evaluation setup is separate from the deployment assumption.

3. **The real question a reviewer would ask** isn't "is TPR deployable?" — it's "does the controller's quadrant assignment correlate with a useful defense choice?" That's answered by the matrix results, not by how the signal was obtained.

4. **What to write in the paper:**

> "**Threat model.** We evaluate the controller in an *offline model auditing* setting where the defender has access to a held-out set of triggered validation inputs (e.g., from preliminary trigger reverse-engineering or red-team probing). This is consistent with prior work [cite STRIP, Neural Cleanse, ANP] which evaluates detection under known-trigger conditions. Extending the controller to fully label-free operation — for instance, by using AC-flagged suspicious samples as STRIP probes — is an immediate direction for future work."

This is honest, standard, and doesn't require re-running any experiments.

---

## Issue 5: Cost Barely Affects the Score — Best Solution

**The problem in numbers.** With α=1.0, β=0.5, γ=0.1, the scoring function at [controller.py L239–243](file:///c:/Users/Asus/Desktop/PESU/capstone/sem7/repo/adaptive-backdoor-defense/controller/controller.py#L239-L243) gives:

```
Score = 1.0 × ASR + 0.5 × ΔCA + 0.1 × C × 100
```

In practice:
- ASR term: 0 to 100 (dominates everything)
- ΔCA term: 0.5 × 3pp ≈ 1.5 (negligible)
- Cost term: 0.1 × 100 = 10 at maximum (secondary)

So two defenses that achieve ASR of 2% vs 5% are separated by **3 points on ASR alone** — larger than the maximum possible cost difference. The controller will always pick the lowest-ASR defense regardless of cost. The "cost-aware" claim is empty.

**The options I considered:**

| Option | Pros | Cons |
|--------|------|------|
| A. Increase γ | Cost matters more | Arbitrary — why γ=0.5 rather than 0.1? |
| B. γ sensitivity sweep (figure) | Shows cost matters at some γ | Doesn't fix the headline |
| C. Decouple: security score + cost as separate axes | Clean, interpretable | Changes the paper's framing |
| D. Pareto frontier plot | Visually compelling | Same as C but prettier |

**My recommendation: Option C+D combined. Drop the single "formal score" for the headline table. Report security and cost separately.**

Here's the concrete change:

**Instead of this headline table:**
```
Strategy              | Score
Always-FT             | 42.3
Adaptive Controller   | 8.7
Oracle                | 5.2
```

**Do this:**
```
Strategy              | Avg ASR↓ | Avg ΔCA↓ | Avg Cost↓ | Conditions where defense fails (ASR>10%)
Always-FT             | 66.7%    | 0.5pp    | 1.0       | 5/9 ❌
Always-ANP            | 8.2%     | 3.1pp    | 4.0       | 1/9
Always-BaEraser-lite  | 3.1%     | 3.0pp    | 9.0       | 0/9
Adaptive Controller   | 4.5%     | 2.8pp    | 3.2       | 0/9 ← same security, 64% less compute
Oracle                | 2.1%     | 2.5pp    | 5.8       | 0/9
```

The story becomes: **"The adaptive controller matches Always-BaEraser security while using 64% less compute, because it applies heavy defenses only when detector signals indicate they're needed."**

This is much more convincing than a weighted score where cost contributes 10/110 points. The compute savings are visible *in their own column*.

You can still keep the formal objective `d* = argmin[α·ASR + β·ΔCA + γ·C]` in the Methods section as the *framework*. But the results section should present the axes separately and let the reader see the trade-off directly.

**Add a Pareto figure:** X-axis = average compute cost, Y-axis = average ASR after defense. Plot each strategy as a point. The adaptive controller should sit near the Pareto frontier — lower ASR than cheap strategies, lower cost than expensive strategies. This is the visual proof of the "saves compute" claim.

**For the γ sweep:** include it as a supplementary figure, not the headline. Show: "At γ=0, the controller is equivalent to Always-Best-ASR. At γ>0.3, the controller starts preferring FT for easy conditions (BadNets). The default γ=0.1 represents a security-first trade-off."

---

## Issue 8: In-Sample Calibration — Best Solution

**The problem:** `calibrate_thresholds()` at [controller.py L249–327](file:///c:/Users/Asus/Desktop/PESU/capstone/sem7/repo/adaptive-backdoor-defense/controller/controller.py#L249-L327) grid-searches τ_ac ∈ [0.25, 0.60] and τ_strip ∈ [10%, 70%] to minimize average score over all 9 conditions. Then `evaluate_strategies()` evaluates on the same 9. That's 2 parameters tuned on 9 points and evaluated on the same 9.

**Why standard cross-validation doesn't work well here:**

Leave-one-attack-out would give 3 folds, each calibrating on 6 points and testing on 3. But:
- 6 calibration points for 2 thresholds is borderline
- 3 test points per fold gives very noisy estimates (one lucky/unlucky condition swings the mean by 33%)
- Worse: the attacks occupy very different parts of AC×STRIP space. Calibrating without BadNets (the only one with clear AC-HIGH at 10%) would shift τ_ac dramatically. Calibrating without SK (the one with the blind spot) would miss the policy-fix motivation entirely.

**The options I considered:**

| Option | Honesty | Statistical soundness | Story impact |
|--------|:-------:|:--------------------:|:------------:|
| A. Leave-one-attack-out | High | Poor (3 test points per fold) | Noisy, hard to draw conclusions |
| B. State "in-sample" as limitation | High | N/A (no claim of generalization) | Neutral — honest |
| C. Threshold sensitivity analysis | High | Good (shows robustness, not generalization) | Positive — shows stability |
| D. Calibrate on BadNets+Blended, test on SK | High | Better (natural train/test split) | Strong — tests "unseen attack" claim |

**My recommendation: Option D as the primary result, supplemented by Option C.**

Here's why D is the right frame:

1. **It mirrors the paper's actual research question.** The controller is supposed to be *attack-agnostic*. The strongest test of that claim is: calibrate on well-studied attacks (BadNets, Blended), then deploy against a genuinely different attack family (Silent Killer). SK is the obvious held-out attack because:
   - It's the newest/least-studied of the three
   - It's the one that causes the blind spot
   - It's the stress test the paper is built around

2. **The split is natural, not arbitrary.** "Calibrate on known, test on unknown" is a deployment-realistic scenario. It's much more compelling to a reviewer than random cross-validation.

3. **It gives you a clean 2-sentence framing:**

> "Thresholds τ_ac and τ_strip are calibrated on BadNets and Blended conditions (6 data points). Silent Killer conditions (3 data points) serve as the held-out evaluation, testing whether the controller generalizes to an unseen, adversarially sophisticated attack family."

4. **Supplement with threshold sensitivity (Option C):** After showing D, add a figure or small table:

```
τ_ac shift  | τ_strip shift | Decisions that change | Score change
−0.05       | −10%          | 1/9                   | +0.3
 0.00       |  0%           | 0/9 (baseline)        | 0.0
+0.05       | +10%          | 1/9                   | +0.8
+0.10       | +10%          | 2/9                   | +1.5
```

If the controller's decisions are stable across a ±0.05/±10% window around the calibrated thresholds, overfitting isn't the concern — the quadrant boundaries are naturally separated enough that exact threshold placement doesn't matter much. This directly addresses the reviewer objection without requiring cross-validation.

**Concrete implementation:**

```python
def calibrate_thresholds(detection_data, defense_matrix, holdout_attack=None):
    # If holdout_attack specified, exclude it from calibration
    calib_data = [d for d in detection_data if d["attack"] != holdout_attack]
    # ... grid search on calib_data ...
    
    # Report held-out performance separately
    test_data = [d for d in detection_data if d["attack"] == holdout_attack]
    # ... evaluate on test_data ...
```

---

## Summary — One Sentence Each

| Issue | Best solution | What to write in the paper |
|-------|--------------|---------------------------|
| **4 (STRIP TPR)** | Keep TPR, state "offline audit" threat model with known-trigger assumption. Standard in the literature. | One sentence in §3 (Threat Model) |
| **5 (Cost negligible)** | Drop the single weighted score from the headline. Report ASR, ΔCA, and Cost as separate columns. Add a Pareto plot. Keep the formal objective in Methods only. | Restructure results table + add one figure |
| **8 (In-sample)** | Calibrate on BadNets+Blended, hold out Silent Killer. Add threshold sensitivity table. | Two sentences in §4 + small supplementary table |