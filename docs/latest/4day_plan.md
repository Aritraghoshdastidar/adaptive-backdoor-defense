# 4-Day Execution Plan — Getting This Paper Done

> [!IMPORTANT]
> **Fixing SK is NOT the goal. It's the finding.** Your paper's contribution isn't "we solved all backdoor attacks." It's: *"We built an adaptive controller that efficiently routes to the right defense, and we document exactly where lightweight defenses hit their limits."* That's a publishable story at venues like ACSAC, RAID, AISec (workshop at CCS), ESORICS, or NDSS-workshops — provided the experiments are complete, honest, and the writing is clean.

---

## What You Already Have (more than you think)

### ✅ Complete
| Item | Status | Where |
|------|--------|-------|
| Detection data (9 conditions) | All 9 AC + STRIP rows filled | results.xlsx rows 5–14 |
| BadNets defense matrix | 12/12 cells (3 rates × 4 defenses) | rows 18–29 |
| Blended defense matrix | 12/12 cells (3 rates × 4 defenses) | rows 48–63 |
| SK defense matrix (core 4) | 10/12 cells (BaEraser missing at pr01, pr05) | rows 30–45 |
| Controller code | Working, with policy fix | controller.py |
| Evaluation script | Working, needs defense_matrix.csv | evaluate_controller.py |
| NAD implementation | v3, correct (deepcopy of poisoned) | nad.py |
| SK failure analysis | Thorough, with clean-model controls | sk_defense_v4.ipynb |
| Blended STRIP blind spot | Verified, 5-step chain | blended_stage_C_status.md |

### ❌ Missing (blockers for the headline)
| Item | Effort | Blocker? |
|------|--------|----------|
| `defense_matrix.csv` (export from xlsx) | 30 min | **YES** — controller can't run |
| Blended STRIP correction (79.6→28.0) | 10 min code change | **YES** — wrong routing |
| SK 0.5% defense data | None available | Drop the condition → 8 conditions |
| Threshold stability sweep | 1–2 hrs coding | Needed for calibration story |
| Headline table + Pareto figure | 2–3 hrs | Needed for the paper |
| Cost column in xlsx | 30 min (use notebook timings) | Needed for cost story |

---

## Your Paper's Actual Contribution (be honest about this)

### What you CAN claim:
1. **Adaptive defense routing via detector signals** — a 2-signal (AC + STRIP) controller that maps to 4 defense tiers. No prior work does this exact thing.
2. **Empirical validation on 3 attack families** — BadNets, Blended, SK with 3 poison rates each = 9 conditions, 4 defenses = 36-cell matrix.
3. **Compute-efficiency argument** — the controller achieves near-oracle security on solvable attacks at ~60% less compute than always running the heaviest defense.
4. **Documented failure taxonomy** — STRIP blind spot on Blended pr10 (novel finding), SK defeating all lightweight defenses (confirms SK paper's claims independently), controller correctly escalating.
5. **Policy fix for blind spots** — AC-LOW/STRIP-LOW → ANP (not FT), motivated by the SK discovery.

### What you CANNOT claim:
- ❌ "We defend against all attacks" (SK defeats everything)
- ❌ "The controller generalises to unseen attacks" (only 3 attacks, single seed)
- ❌ "Cost-aware optimisation" (cost barely affects decisions at current γ)

### Target venues (realistic):

| Tier | Venues | What they want |
|------|--------|----------------|
| **Strong workshop** | AISec (CCS workshop), DLSP (IEEE S&P workshop), SaTML workshop | Novel empirical findings, honest evaluation, 6–8 pages |
| **Mid-tier conf** | ACSAC, RAID, ESORICS, CANS, SecureComm | Complete experiments, clear contribution, 12–16 pages |
| **Indian conf** | ISEA, INDOCRYPT, COMSNETS security track | Thorough evaluation, well-written |

For any of these, you need: **complete experiments, honest framing, clean writing, and reproducible results.** You have the first two mostly done.

---

## The 4-Day Plan

### Day 1 (Nov 20): Fix Data → Run Controller → Get Headline Numbers

**Morning (3 hrs):**

1. **Fix Blended STRIP in `evaluate_controller.py`** (10 min)
   - Change lines 60–66: `strip_tpr` from 79.6/92.8/68.0 → 28.0/40.8/0.0
   - Decision: use verified numbers from blended_stage_C_status.md

2. **Drop SK 0.5%** from `DETECTION_DATA` (5 min)
   - Remove the entry at line 67–69 — no defense data exists for it
   - Now 8 conditions, not 9

3. **Export `defense_matrix.csv`** from xlsx (30 min)
   - Script to read xlsx rows 18–65 and write CSV
   - Only include the 4 core defenses (FT, ANP, NAD, BaEraser-lite)
   - Mark SK BaEraser pr01/pr05 as missing
   - Add cost column using notebook timings from Cell I1:
     ```
     FT: ~18s, ANP: ~750s, NAD: ~256s, BaEraser: ~216s
     ```

4. **Handle invalid cells** (15 min)
   - Blended pr10 NAD: exclude (collapse gate failed, CA drop 5.19pp)
   - SK NAD: include raw numbers with `interpretable=False` flag
   - SK BaEraser pr01/pr05: exclude (failed, no checkpoint)

5. **Run the controller** (10 min)
   ```
   python scripts/evaluate_controller.py --matrix results/defense_matrix.csv
   ```

**Afternoon (3 hrs):**

6. **Run with `--calibrate`** and inspect thresholds
7. **Generate the headline table** — the 7-strategy comparison:
   ```
   Strategy              | Avg ASR↓ | Avg ΔCA↓ | Avg Cost↓ | Fails (ASR>10%)
   Always-FT             | ...
   Always-ANP            | ...
   Always-NAD            | ...
   Always-BaEraser       | ...
   Adaptive (old policy) | ...
   Adaptive (new policy) | ...
   Oracle                | ...
   ```
8. **Check:** Does the controller actually beat Always-X on the solvable attacks (BadNets + Blended)? If yes, that's your headline. If not, debug thresholds.

**Expected outcome:** You have the headline numbers. The story either works or it doesn't, and you know today.

---

### Day 2 (Nov 21): Threshold Stability + Figures

**Morning (3 hrs):**

1. **Implement `threshold_sensitivity_sweep()`** in evaluate_controller.py:
   ```python
   for d_ac in [-0.10, -0.05, 0, +0.05, +0.10]:
       for d_strip in [-15, -10, -5, 0, +5, +10, +15]:
           # run controller with perturbed thresholds
           # record: n_decisions_changed, avg_ASR, worst_condition
   ```
2. **Generate the stability table** — proves thresholds aren't overfitted

**Afternoon (3 hrs):**

3. **Pareto plot** (X = avg cost, Y = avg ASR, one point per strategy)
4. **AC × STRIP scatter plot** with quadrant lines (the 2×2 diagram)
5. **Per-condition heatmap** (attack × rate on rows, defense on columns, cell = ASR after)
6. **Policy comparison diagram** — old vs new policy, highlighting the 2 conditions that change

**These 5 figures + the headline table are the paper's results section.**

---

### Day 3 (Nov 22): SK Aggressive NAD + Fill Any Gaps

**Morning: Launch the aggressive teacher sweep on Colab (if GPU available)**

```python
# Try 2 teacher recipes on SK pr01 only (cheapest test)
# Recipe 1: lr=0.05, epochs=20, max_ca_drop=0.08
# Recipe 2: lr=0.02, epochs=30, max_ca_drop=0.08
```

This is a ~2 hour GPU run. If it works (teacher ASR < 40%), run the full NAD distillation and update the matrix. If it doesn't, you have a confirmed negative result — which is also valuable.

**Afternoon: Fill remaining gaps**

- If Blended STRIP numbers changed the routing, verify the new routes make sense
- Double-check that the controller's routing decisions are correct per the corrected data
- Run the LORO cross-validation if time (optional — threshold stability alone is sufficient)
- Write up the SK failure paragraph

**Evening: Compile all results into one clean notebook / script output**

---

### Day 4 (Nov 23): Results Polish + Paper Draft Outline

**Morning:**

1. **Final controller run** with any updates from Day 3
2. **Export all tables/figures** as publication-quality PNGs/PDFs
3. **Write the results section bullet points** (not prose yet, just the claims with numbers)

**Afternoon:**

4. **Paper outline with section-by-section content plan:**
   ```
   §1 Introduction (1 page)
   §2 Background & Related Work (1.5 pages)
   §3 Threat Model & System Design (2 pages)
      - 2×2 quadrant controller
      - Policy fix motivation
   §4 Experimental Setup (1 page)
      - 3 attacks, 3 rates, 4 defenses, seed 2027
      - Detection: AC (PCA-2D) + STRIP (500 samples)
      - Calibration: in-sample + stability analysis
   §5 Results (3 pages)
      - Table 1: Headline strategy comparison
      - Table 2: Per-condition controller decisions
      - Figure 1: AC×STRIP scatter with quadrants
      - Figure 2: Pareto plot
      - Table 3: Threshold sensitivity
   §6 Failure Analysis & Limitations (1 page)
      - SK: no lightweight defense works
      - STRIP blind spot on Blended pr10
      - Single seed, CIFAR-10 only
   §7 Conclusion (0.5 page)
   ```

---

## What Makes or Breaks This Paper

### 🟢 You're strong on:
- **Thoroughness of evaluation** — 36-cell matrix is genuine work, with clean-model controls and validity flags
- **Honest failure documentation** — the SK failure + STRIP blind spot are genuinely interesting findings
- **Code quality** — nad.py with the leak fix, the notebook's verification chain, the controller's audit trail

### 🔴 You're weak on:
- **Single seed** — every result is seed=2027 only. State this as a limitation. For a workshop paper this is fine. For a full conference paper at RAID/ACSAC, reviewers may push back.
- **Single dataset** — CIFAR-10 only. Same: fine for a workshop, limitation for a conference.
- **SK makes the headline noisy** — if you include SK in the averages, the controller's avg ASR looks bad. 

> [!TIP]
> **Split the headline.** Report two rows:
> ```
> Adaptive Controller (all 8)     | 25.3% | ...  ← includes SK failures
> Adaptive Controller (solvable)  |  1.2% | ...  ← BadNets + Blended only
> ```
> Then say: *"On attacks where at least one defense is effective (6/8 conditions), the controller achieves 1.2% avg ASR at 60% less compute than Always-BaEraser. On Silent Killer (2/8 conditions), no lightweight defense succeeds, and the controller correctly identifies the need for escalation."*

### 🟡 Addressable in writing:
- Frame the **policy fix** (FT → ANP for AC-LOW/STRIP-LOW) as a contribution, not a patch
- Frame the **failure taxonomy** as a contribution, not a weakness
- Frame the **STRIP blind spot** as a novel empirical finding, not a bug

---

## Day 1 Checklist (start here)

- [ ] Fix Blended STRIP numbers in evaluate_controller.py
- [ ] Drop SK 0.5% from DETECTION_DATA
- [ ] Write xlsx → defense_matrix.csv export script
- [ ] Add timing data to CSV
- [ ] Handle invalid/failed cells
- [ ] Run `evaluate_controller.py`
- [ ] Run `evaluate_controller.py --calibrate`
- [ ] Check: does the controller beat Always-X on BadNets + Blended?
- [ ] Record the headline numbers

**If the headline works, the paper is real. Everything after Day 1 is making it stronger.**
