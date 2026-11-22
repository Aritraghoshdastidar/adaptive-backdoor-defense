# Complete Action Plan: Fixes + Experiments + Paper

This document combines:
1. The 6 audit issues that must be fixed before writing the paper
2. The zero-GPU experiment (LOOCV) that addresses in-sample criticism
3. The GPU experiment (WaNet) that proves generalization
4. How everything fits into the paper

---

## PART 1: CRITICAL FIXES (Must Do Before Paper Draft)

> [!CAUTION]
> These are data integrity issues. Skipping them risks reviewer rejection.

---

### Fix 1: Lock the STRIP Configuration

**The problem:** Three different Blended STRIP TPR sets exist:

| Source | Config | TPR at 1%/5%/10% |
|---|---|---|
| Stage C doc (`blended_stage_C_status.md`) | n=500, α_test=0.1 (matched to α_train) | 28.0 / 40.8 / **0.0** |
| Old controller code (stale) | Unknown/legacy | 79.6 / 92.8 / 68.0 |
| Latest defense notebook | n=125, different α | 71.2 / 91.2 / 64.8 |

**Currently using:** Stage C values (28.0/40.8/0.0) — already applied in Day 1.

**The right choice:** Stage C values ARE the defensible ones because:
- α_test = 0.1 matches α_train = 0.1 (the trigger strength the model was trained with)
- n=500 is a larger sample than 125
- The 0% TPR at pr10 is a confirmed real finding per the Stage C doc (not a bug — the entire triggered-sample entropy distribution sits above the detection threshold)

**What you need to do:**

1. Open the Blended defense notebook and the Blended Stage C notebook side by side
2. For each, find and record:
   - `alpha_test` (trigger α used when building the STRIP test pool)
   - STRIP overlay α (usually 0.5 — the perturbation blend weight)
   - `n_samples` (how many triggered/clean samples were tested)
3. Confirm they're different configs → the Stage C one (α_test=0.1, n=500) is the matched one
4. Write a 2-line footnote for the paper: *"STRIP evaluated with α_test = 0.1 matching α_train; n = 500 triggered and 500 clean test samples; detection threshold set at FPR ≤ 1%."*

**If you discover both notebooks actually used the SAME config and the difference is a bug:** that's a bigger problem — you'd need to re-run STRIP. But the Stage C doc is very thorough (it checked for bugs, matched conditions, etc.), so this is unlikely.

> [!IMPORTANT]
> **The key point:** don't switch to the defense notebook's higher values (71.2/91.2/64.8) just because they give nicer headlines. A reviewer will check if your STRIP config matches your trigger config, and α_test = 0.1 matching α_train = 0.1 is the only defensible choice.

**Impact on headlines:** With the Stage C values (currently loaded), Blended 10% routes to ANP (0.21% ASR) → good. With the defense notebook values (64.8%), Blended 10% would be STRIP-HIGH, routing to BaEraser (16.73% ASR, a fail) → bad. The Stage C values are both more principled AND give better results. Report a sensitivity analysis showing what happens if STRIP TPR were higher — the threshold stability sweep already covers this.

---

### Fix 2: Always-NAD's Imputed 100%

**The problem:** Blended 10% NAD was excluded from the defense matrix (collapse gate failure). But `evaluate_strategies()` imputes missing cells at ASR = 100% (line 399-410 of `controller.py`). So Always-NAD's solvable average includes a 100% penalty for Blended 10% NAD, inflating it to 17.54%.

**The actual NAD data (solvable, 5 valid cells):**
- BadNets: 1.9, 1.1, 1.3
- Blended: 0.74, 0.22
- Mean of valid cells: **1.05%** with 0 fails

**With the imputed 100% (current):**
- Mean = (1.9 + 1.1 + 1.3 + 0.74 + 0.22 + 100.0) / 6 = **17.54%** with 1 fail

**What to do — pick ONE approach and be consistent:**

**Option A (Recommended): Keep the 100% penalty but footnote it clearly.**
- Rationale: If you're going to use NAD on every model, the fact that it collapses on one condition IS a real cost. The penalty is fair — you just need to say *"Always-NAD includes a failure penalty (ASR=100%) for Blended 10% where NAD collapsed during training."*
- Keeps the comparison apples-to-apples (all strategies penalized uniformly for missing cells)
- Makes the controller look better against NAD → but that's honestly correct (the controller avoided NAD on that condition)

**Option B: Exclude the cell and report N-1.**
- Report Always-NAD as 1.05% over 5/6 conditions, with a note that one cell was invalid
- Problem: now you're comparing a 5-cell average against 6-cell averages for other strategies — not apples-to-apples
- A reviewer will notice the different denominators

**Recommendation:** Keep Option A (penalty), but add a column or footnote to the table: `†: includes failure penalty for collapsed/invalid cells`.

---

### Fix 3: Always-BaEraser Row

**The problem:** The audit says the reported 1.68% with 0 fails doesn't match manual calculation.

**Manual calculation from defense_matrix.csv (solvable, 6 cells):**

| Condition | BaEraser ASR |
|---|---|
| BadNets 1% | 0.80 |
| BadNets 5% | 0.90 |
| BadNets 10% | 0.90 |
| Blended 1% | 0.15 |
| Blended 5% | 1.33 |
| Blended 10% | **16.73** (selection_failure) |
| **Mean** | **3.47%** |
| **Fails** | **1** (Blended 10% > 10%) |

**What to do:**
1. Re-run `python scripts/evaluate_controller.py` and check the Always-BaEraser-lite row
2. If it says 1.68% / 0 fails → there's a bug in the code (maybe the Blended 10% BaEraser cell wasn't in the CSV when you first ran it, or there's a filtering issue)
3. If it says 3.47% / 1 fail → the earlier reported number was from a stale run; the current numbers are correct
4. Either way, the correct answer is **3.47% with 1 fail** based on the defense_matrix.csv data

**Impact:** This actually hurts the controller's relative story slightly. Always-BaEraser at 3.47% is still worse than the controller's 0.60%, but it's not as bad as reported. And BaEraser costs 0.36 vs controller's 0.69 — so BaEraser is cheaper AND decent. The controller's advantage over BaEraser is **security** (0.60% vs 3.47%) not cost.

---

### Fix 4: Restate the Cost Claim

**The problem:** The "31% compute saving" is only vs Always-ANP. But Always-BaEraser (cost 0.36) is cheaper than the controller (cost 0.69).

**Current cost comparison (solvable):**

| Strategy | Avg ASR | Cost | Fails |
|---|---|---|---|
| Always-FT | 58.23% | 0.03 | 5 |
| Always-ANP | 0.53% | 1.00 | 0 |
| Always-NAD | 17.54%† | 0.43 | 1 |
| Always-BaEraser | 3.47% | 0.36 | 1 |
| **Adaptive Controller** | **0.60%** | **0.69** | **0** |
| Oracle | 0.53% | 0.48 | 0 |

**The honest claim:**
- vs Always-ANP: "Matches ANP's near-zero ASR (0.60% vs 0.53%) while saving 31% compute"  ✅
- vs Always-BaEraser: "Achieves much lower ASR (0.60% vs 3.47%) and 0 fails (vs 1 fail), at higher but comparable cost"  ✅
- DON'T say: "60% cheaper than BaEraser" ← this is backwards, BaEraser is cheaper  ❌

**Better framing for the paper:**

*"The adaptive controller occupies the Pareto-efficient region of the ASR–cost trade-off: it achieves the lowest ASR among all automated strategies (0.60%) with zero failures, at 31% lower cost than the nearest equally-reliable alternative (Always-ANP). Cheaper strategies (Always-FT, Always-BaEraser) sacrifice either security or reliability."*

---

### Fix 5: Check if Blended Defense Rows Need Updating

**The problem:** The audit mentions that the corrected Blended re-run gives slightly different ANP numbers:
- Old xlsx: ANP at 0.06 / 1.88 / 0.21, baselines 94.95 / 94.81 / 94.48
- New notebook: ANP at 0.14 / 0.60 / 0.01, baselines 94.83 / 94.73 / 94.39

**What to do:**
1. Check which numbers are in your defense_matrix.csv right now → they show 0.06 / 1.88 / 0.21 (the old xlsx values)
2. Check the Blended defense notebook output for the most recent run
3. **The "CA Before" fingerprint test:** Every defense notebook's "CA before" must match the attack notebook's CA. If the defense notebook says CA_before=94.83 but defense_matrix.csv says 94.95, the normalization differs.
4. If the new notebook's numbers are from a corrected run, update the defense matrix

**Impact:** Likely small — the differences are tiny (0.06→0.14, 1.88→0.60, 0.21→0.01). The controller's routing decisions won't change. But you need ONE canonical set of numbers.

---

### Fix 6: Frame the Controller Honestly

**The problem:** The controller is basically "ANP everywhere except when AC+STRIP say something specific." Its actual routing on solvable conditions:

| Condition | Controller picks | ASR |
|---|---|---|
| BadNets 1% | ANP (policy fix) | 0.20% |
| BadNets 5% | NAD | 1.10% |
| BadNets 10% | BaEraser | 0.90% |
| Blended 1% | ANP (policy fix) | 0.06% |
| Blended 5% | BaEraser | 1.33% |
| Blended 10% | ANP (STRIP blind spot) | 0.21% |

It picks ANP 3/6 times, BaEraser 2/6, NAD 1/6. ANP alone would give 0.53%.

**Honest framing:**
- The controller's value is NOT "dramatically better security" — ANP alone is basically as good
- The controller's value IS:
  1. **Cost saving** (0.69 vs 1.00) — it uses cheaper defenses when detector signals indicate they'll work
  2. **Routing transparency** — every decision has an auditable reason tied to detector signals
  3. **Failure escalation** — when signals are weak AND defenses fail (SK), the audit trail flags it
  4. **Policy adaptability** — the policy fix shows the routing logic can be updated without retraining

**Don't claim:** "Our controller outperforms all baselines" (it basically matches ANP)
**Do claim:** "Our controller achieves equivalent security to the most expensive defense at 31% lower cost, with transparent routing decisions and built-in failure escalation"

---

## PART 2: ZERO-GPU EXPERIMENT — LOOCV

> [!IMPORTANT]
> This is the single highest-ROI thing you can do. Zero GPU cost, 1-2 hours, and it kills the biggest reviewer concern.

### What It Is

Leave-One-Out Cross-Validation on the 6 solvable conditions. For each condition:
1. Hold it out as the "test" condition
2. Calibrate thresholds on the remaining 5 conditions
3. Run the controller on the held-out condition
4. Record: did it pick the right defense? What was the ASR?

### Why It Matters

Right now your thresholds (τ_ac=0.40, τ_strip=30%) were calibrated on the same 6 conditions they're evaluated on. A reviewer will say *"of course it works — it was tuned on the test set."*

LOOCV shows: "Even when a condition is completely unseen during threshold calibration, the controller still routes correctly."

### How to Implement

I'll create this script for you. The logic:

```
For each i in 0..5:
    held_out = solvable_conditions[i]
    training = solvable_conditions[:i] + solvable_conditions[i+1:]
    
    # Re-calibrate thresholds on training set only
    thresholds_i = calibrate_thresholds(training, defense_matrix)
    
    # Test on held-out condition
    decision_i = controller_decide(held_out, thresholds_i)
    
    # Look up defense result
    result_i = defense_matrix[(held_out.attack, held_out.condition, decision_i.defense)]
    
    # Record
    record(held_out, thresholds_i, decision_i, result_i)
```

### What to Report in the Paper

A table like:

| Held-Out Condition | Calibrated τ_ac | Calibrated τ_strip | Defense Chosen | ASR After | Same as Full-Calibration? |
|---|---|---|---|---|---|
| BadNets 1% | ? | ? | ? | ? | ✓/✗ |
| BadNets 5% | ? | ? | ? | ? | ✓/✗ |
| ... | | | | | |

If all 6 rows show correct routing (ASR < 10%), you can claim: *"Leave-one-out cross-validation confirms the controller generalizes to unseen attack conditions: all 6 held-out tests achieved ASR < 10% with correct defense routing."*

### Expected Outcome

Very likely to pass because:
- The AC silhouettes split cleanly into LOW (0.32-0.38) and HIGH (0.44-0.64) groups → τ_ac has a wide margin
- The STRIP TPRs split into LOW (0-28%) and HIGH (33-86%) → τ_strip has a margin between 28 and 33
- The threshold sensitivity sweep already showed the controller is stable across ±10%/±15% perturbations

---

## PART 3: GPU EXPERIMENT — WaNet (Held-Out Attack)

> [!TIP]
> This is the experiment that upgrades the paper from "workshop" to "conference-competitive."

### What It Is

Train backdoored models using the **WaNet** attack (geometric warping) at 2 poison rates (5% and 10%), then run your full detection → controller → defense pipeline on them **without recalibrating thresholds**. The controller has never seen WaNet during calibration.

### Why WaNet

| Factor | Why it's ideal |
|---|---|
| Mechanism | Geometric image warping — completely different from patch (BadNets) or noise (Blended) |
| Defenses work? | ANP reliably kills it (ASR → 0.3-2% per literature). NAD works but with CA tradeoff |
| AC detectable? | Yes — warped samples create activation clusters. Expect AC silhouette > 0.4 |
| STRIP detectable? | Moderate — warping changes entropy moderately. Expect TPR 20-50% |
| Literature status | Well-known, published at ICLR 2021. Reviewers won't question the choice |

### Implementation Steps

#### Step 1: Set up WaNet attack code (~1-2 hours)

**Option A: Use BackdoorBench** (recommended if you have it)
```bash
pip install backdoorbench  # or clone github.com/SCLBD/BackdoorBench
```
BackdoorBench has WaNet built in with CIFAR-10 + ResNet-18.

**Option B: Implement from scratch** (1-2 hours)
WaNet generates a warping field using a small noise grid that's upsampled with bicubic interpolation:

```python
import torch
import torch.nn.functional as F

def generate_wanet_trigger(image, grid_size=4, warping_strength=0.5, noise_strength=2.0):
    """
    WaNet: warp the image using a smooth deformation field.
    - grid_size: control points for the warping grid (4 = subtle)
    - warping_strength: magnitude of warping (s in the paper)
    - noise_strength: scale of random noise grid
    """
    h, w = image.shape[-2:]
    
    # Create identity grid
    grid = F.affine_grid(
        torch.eye(2, 3).unsqueeze(0), 
        [1, 1, h, w], 
        align_corners=True
    )
    
    # Create small noise grid and upsample (smooth warping)
    noise = torch.randn(1, 2, grid_size, grid_size) * noise_strength
    noise_upsampled = F.interpolate(
        noise, size=(h, w), mode='bicubic', align_corners=True
    ).permute(0, 2, 3, 1)  # [1, H, W, 2]
    
    # Warp
    warped_grid = grid + warping_strength * noise_upsampled / h
    warped_image = F.grid_sample(
        image.unsqueeze(0), warped_grid, align_corners=True
    ).squeeze(0)
    
    return warped_image
```

**Key:** The noise grid must be **fixed** (same `torch.manual_seed` for all triggered samples). The warping is the trigger — same deformation applied to every poison sample.

#### Step 2: Train poisoned models (~2-3 hours GPU each)

```python
# Pseudo-code — adapt to your existing training pipeline
POISON_RATES = [0.05, 0.10]  # 5% and 10%
TARGET_CLASS = 0  # airplane, same as your other attacks
SEED = 2027

for pr in POISON_RATES:
    # Poison the training data
    n_poison = int(len(train_set) * pr)
    poison_indices = select_non_target_indices(train_set, TARGET_CLASS, n_poison, seed=SEED)
    
    for idx in poison_indices:
        img, label = train_set[idx]
        train_set[idx] = (generate_wanet_trigger(img), TARGET_CLASS)
    
    # Train ResNet-18 (same hyperparams as your BadNets/Blended training)
    model = train_resnet18(train_set, seed=SEED, epochs=200, lr=0.1, ...)
    
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoints/wanet/resnet18_wanet_pr{pr}_seed2027.pth')
    
    # Evaluate baseline CA and ASR
    ca = evaluate_clean_accuracy(model, test_set)
    asr = evaluate_attack_success(model, test_set, generate_wanet_trigger, TARGET_CLASS)
    print(f"WaNet pr={pr}: CA={ca:.2f}%, ASR={asr:.2f}%")
```

**Expected baseline results** (from literature):
- CA: ~93-94% (similar to clean)
- ASR: >99% (WaNet is very effective)

#### Step 3: Run detection pipeline (~30 min per model)

```python
# Run your existing AC + STRIP pipeline on WaNet models
# Use the EXACT same detection code as BadNets/Blended

from core.detection import run_ac, run_strip

for pr in ['pr05', 'pr10']:
    model = load_model(f'checkpoints/wanet/resnet18_wanet_{pr}_seed2027.pth')
    
    # AC: use same defense_indices.npy, same clustering
    ac_results = run_ac(model, defense_data, n_clusters=2)
    # Record: silhouette, PDR, suspicious_fraction
    
    # STRIP: use alpha_test=0.1, n_samples=500 (SAME config as all other attacks!)
    strip_results = run_strip(model, test_data, 
                              trigger_fn=generate_wanet_trigger,
                              alpha_test=0.1, n_samples=500)
    # Record: TPR, FPR, mu, sigma
```

**Expected detection results:**
- AC silhouette: probably 0.35-0.55 (warped samples cluster, but subtly)
- STRIP TPR: probably 15-45% (moderate — warping changes entropy but not dramatically)

#### Step 4: Run controller → defense → evaluate (~4-6 hours GPU)

```python
# DO NOT recalibrate thresholds — use τ_ac=0.40, τ_strip=30% from BadNets+Blended calibration
# This is the whole point: the controller is being tested on an UNSEEN attack

for pr in ['5%', '10%']:
    det = {
        "attack": "WaNet", "poison_condition": pr,
        "ac_silhouette": measured_silhouette,
        "strip_tpr": measured_tpr,
        # ... other AC/STRIP values
    }
    
    decision = controller_decide(**det, thresholds=default_thresholds)
    print(f"WaNet {pr}: Controller picks → {decision.defense_chosen}")
    print(f"  Reason: {decision.reason}")
    
    # Run the chosen defense
    defended_model = run_defense(decision.defense_chosen, model, clean_data)
    
    # Evaluate post-defense
    ca_after = evaluate_clean_accuracy(defended_model, test_set)
    asr_after = evaluate_attack_success(defended_model, test_set, generate_wanet_trigger, TARGET_CLASS)
    print(f"  Post-defense: CA={ca_after:.2f}%, ASR={asr_after:.2f}%")
```

**Also run ALL 4 defenses** (not just the controller's pick) for the ablation matrix:
- FT, ANP, NAD, BaEraser on each WaNet model
- This fills out the defense matrix rows for WaNet, enabling fair Always-X comparisons

#### Step 5: What to expect and how to handle results

**Best case (likely):** Controller picks ANP or BaEraser, ASR drops to <5%. WaNet joins the "solvable" family.

**Acceptable case:** Controller picks a defense that partially works (ASR 5-15%). Shows the controller's routing is reasonable but not perfect on unseen attacks. Still publishable with honest discussion.

**Bad case (unlikely):** Controller picks a defense that fails (ASR > 50%). This would mean the AC+STRIP signals for WaNet don't map to the same quadrant semantics as BadNets/Blended. Write it up as a limitation and don't include WaNet in the solvable averages.

### Effort Summary

| Step | Time | GPU? |
|---|---|---|
| Implement WaNet trigger | 1-2 hours | No |
| Train 2 poisoned models | 4-6 hours | Yes (Colab) |
| Run AC + STRIP | 1 hour | Yes |
| Run 4 defenses × 2 models | 4-6 hours | Yes |
| Evaluate + integrate results | 1-2 hours | No |
| **Total** | **~12-16 hours** | **~10-12 hours GPU** |

You can run the GPU steps overnight on Colab. The non-GPU steps fit into a morning.

---

## PART 4: MIXED ATTACK (Optional Bonus — Only If Time Permits)

### What It Is

Train ONE model with BOTH BadNets and Blended triggers simultaneously. Test whether the controller routes correctly and whether one defense removes both backdoors.

### Setup

```python
# Total poison rate: 10% (5% BadNets + 5% Blended)
n_poison_badnets = 2500  # 5% of 50K
n_poison_blended = 2500  # 5% of 50K

# Poison with BadNets trigger (patch in corner)
for idx in badnets_indices:
    img, _ = train_set[idx]
    img = add_badnets_trigger(img)  # your existing function
    train_set[idx] = (img, TARGET_CLASS)

# Poison with Blended trigger (alpha blend)
for idx in blended_indices:
    img, _ = train_set[idx]
    img = add_blended_trigger(img, alpha=0.1)  # your existing function
    train_set[idx] = (img, TARGET_CLASS)

# Train
model = train_resnet18(train_set, seed=2027, ...)
```

### Evaluation

After defense, test against BOTH triggers separately:
```python
asr_badnets = evaluate_attack_success(model, test_set, add_badnets_trigger, TARGET_CLASS)
asr_blended = evaluate_attack_success(model, test_set, add_blended_trigger, TARGET_CLASS)
```

### How to Frame Results

| Outcome | Paper Framing |
|---|---|
| Defense removes BOTH | "Controller is robust to compound threats" |
| Defense removes ONE only | "Current routing assumes single-trigger models; compound threats require multi-pass defense (future work)" |
| Defense removes NEITHER | "Compound poisoning at high total rates overwhelms lightweight defenses" (limitation) |

### Recommendation

**Only do this if WaNet is done and you still have GPU time.** The mixed attack is interesting but the result is unpredictable. WaNet is the safe, high-ROI experiment.

---

## PART 5: EXECUTION ORDER & TIMELINE

### Day 1 (Today): Fixes + LOOCV

| Time | Task | Location |
|---|---|---|
| 1 hour | Verify STRIP config: compare Stage C vs defense notebook configs | Notebooks (local) |
| 30 min | Re-run `evaluate_controller.py`, verify BaEraser/NAD rows | Local terminal |
| 30 min | Fix any discrepancies in defense_matrix.csv | Local |
| 1-2 hours | Implement + run LOOCV script | I can do this for you |
| 30 min | Update cost claim framing in notes | Local |

### Day 1 Evening: Start WaNet on Colab

| Time | Task | Location |
|---|---|---|
| 1-2 hours | Implement WaNet trigger + training notebook | Colab |
| Overnight | Train 2 WaNet models (5% and 10%) | Colab GPU |

### Day 2: WaNet Detection + Defense + Integration

| Time | Task | Location |
|---|---|---|
| Morning | Run AC + STRIP on WaNet models | Colab GPU |
| Afternoon | Run 4 defenses × 2 WaNet models | Colab GPU |
| Evening | Integrate WaNet results into controller evaluation | Local |
| Evening | Re-run figures/tables with updated data | Local |

### Days 3-4: Paper Draft

With all data locked:
- Write §1-§7 of the paper
- Plug in figures and tables
- Write the LOOCV + WaNet generalization subsection
- Write the honest limitations paragraph

---

## PART 6: WHAT THE PAPER SECTION STRUCTURE LOOKS LIKE

```
§1 Introduction (1 page)
  - Problem: picking the right backdoor defense is attack-dependent
  - Our approach: detection-driven adaptive routing
  - Contributions: (1) 2×2 controller, (2) cost savings, (3) failure taxonomy

§2 Background & Related Work (1.5 pages)
  - Backdoor attacks: BadNets, Blended, WaNet, Silent Killer
  - Detection: AC, STRIP
  - Defenses: FT, ANP, NAD, BaEraser
  - Gap: no existing work adaptively routes to defenses based on detector signals

§3 Threat Model & System Design (2 pages)
  - Threat model: defender receives a potentially backdoored model + small clean dataset
  - Controller design: AC silhouette × STRIP TPR → 2×2 quadrant → defense selection
  - Policy: AC-HIGH/STRIP-HIGH → BaEraser, AC-HIGH/STRIP-LOW → ANP, 
    AC-LOW/STRIP-HIGH → NAD, AC-LOW/STRIP-LOW → ANP (safety floor)
  - Formal cost-weighted score function

§4 Experimental Setup (1 page)
  - 4 attacks × 2-3 rates = 10 conditions (BadNets, Blended, WaNet, SK)
  - 4 defenses, seed 2027, CIFAR-10, ResNet-18
  - Detection pipeline: AC (n_clusters=2), STRIP (α_test=0.1, n=500)
  - Calibration: LOOCV on BadNets + Blended, WaNet as held-out test

§5 Results (3 pages)
  - Table 1: Strategy comparison (solvable, solvable+WaNet, full suite)
  - Table 2: Per-condition controller decisions + LOOCV results
  - Figure 1: 2×2 quadrant scatter (now with WaNet points!)
  - Figure 2: ASR vs Cost Pareto
  - Figure 3: Defense heatmap
  - Figure 4: Policy ablation (old vs new)
  - Figure 5: Threshold stability

§6 Failure Analysis & Limitations (1 page)
  - SK failure taxonomy (all defenses fail)
  - STRIP blind spot on Blended 10% (absorbed by AC routing)
  - Single seed, single dataset limitation
  - Controller assumes single-trigger models

§7 Conclusion (0.5 page)
  - Summary of contributions
  - Future work: multi-trigger, more datasets, learned controller
```

---

## CHECKLIST SUMMARY

### Must-Do (before paper draft):
- [ ] Verify STRIP config (compare Stage C vs defense notebook)
- [ ] Re-run evaluate_controller.py, check BaEraser/NAD rows
- [ ] Fix any CSV discrepancies
- [ ] Implement and run LOOCV script
- [ ] Rewrite cost claim (31% vs ANP, not vs BaEraser)
- [ ] Frame controller contribution as "cost-saving router" not "security improvement"

### Should-Do (makes paper conference-competitive):
- [ ] Implement WaNet attack
- [ ] Train WaNet at 5%, 10% on Colab
- [ ] Run detection + defense pipeline on WaNet
- [ ] Integrate WaNet into controller evaluation
- [ ] Update all figures and tables

### Nice-to-Have (if time remains):
- [ ] Mixed BadNets+Blended attack
- [ ] STL-10 cross-dataset inference
- [ ] Update ANP timing from 750s to 591s (if verified)
