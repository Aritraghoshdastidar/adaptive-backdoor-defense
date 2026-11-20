Yes. One correction first: **the post-defense matrix is 36 rows**, but the controller-input sheet should remain **9 rows for BadNets + Blended + 3 canonical Silent Killer conditions**, because the controller sees only the pre-defense signals. The full 36-row matrix is the ground truth used to evaluate whether the controller's choice was good. Your project context explicitly locks this full matrix and the four defenses as FT, ANP, NAD, and BAERASER-lite. 

I can safely fill the **BadNets values I have actually verified**. I **do not have verified post-defense Blended FT/ANP/NAD/BaEraser numbers in the supplied material**, so I won't invent them. Your plan explicitly says those cells are experiments to run, not known results. 

## 1. Post-defense master table — copy/paste

```markdown
| Attack | Poison Rate | Defense | Clean Acc After (%) | ASR After (%) | CA Drop (pp) | ASR Reduction (pp) | Defense Success |
|---|---:|---|---:|---:|---:|---:|---|
| BadNets | 1% | FT | — | — | — | — | — |
| BadNets | 1% | ANP | — | — | — | — | — |
| BadNets | 1% | NAD | — | — | — | — | — |
| BadNets | 1% | BaEraser-lite | 87.26 | 0.8 | — | — | — |
| BadNets | 5% | FT | — | — | — | — | — |
| BadNets | 5% | ANP | — | — | — | — | — |
| BadNets | 5% | NAD | — | — | — | — | — |
| BadNets | 5% | BaEraser-lite | 86.70 | 0.9 | — | — | — |
| BadNets | 10% | FT | — | — | — | — | — |
| BadNets | 10% | ANP | — | — | — | — | — |
| BadNets | 10% | NAD | — | — | — | — | — |
| BadNets | 10% | BaEraser-lite | 86.72 | 0.9 | — | — | — |
| Blended | 1% | FT | — | — | — | — | — |
| Blended | 1% | ANP | — | — | — | — | — |
| Blended | 1% | NAD | — | — | — | — | — |
| Blended | 1% | BaEraser-lite | — | — | — | — | — |
| Blended | 5% | FT | — | — | — | — | — |
| Blended | 5% | ANP | — | — | — | — | — |
| Blended | 5% | NAD | — | — | — | — | — |
| Blended | 5% | BaEraser-lite | — | — | — | — | — |
| Blended | 10% | FT | — | — | — | — | — |
| Blended | 10% | ANP | — | — | — | — | — |
| Blended | 10% | NAD | — | — | — | — | — |
| Blended | 10% | BaEraser-lite | — | — | — | — | — |
| Silent Killer | n=1 | FT | — | — | — | — | — |
| Silent Killer | n=1 | ANP | — | — | — | — | — |
| Silent Killer | n=1 | NAD | — | — | — | — | — |
| Silent Killer | n=1 | BaEraser-lite | — | — | — | — | — |
| Silent Killer | n=10 | FT | — | — | — | — | — |
| Silent Killer | n=10 | ANP | — | — | — | — | — |
| Silent Killer | n=10 | NAD | — | — | — | — | — |
| Silent Killer | n=10 | BaEraser-lite | — | — | — | — | — |
| Silent Killer | n=250 | FT | — | — | — | — | — |
| Silent Killer | n=250 | ANP | — | — | — | — | — |
| Silent Killer | n=250 | NAD | — | — | — | — | — |
| Silent Killer | n=250 | BaEraser-lite | — | — | — | — | — |
```

**Why n=1, 10, 250 for SK?** Your current project context says those are the suggested canonical low/transition/saturated points, while the denser n=1,2,5,10,25,250 sweep remains a detection-side ablation. 

---

# 2. Don't put "Defense Chosen" into this matrix

This is important.

The **36-row matrix is the evaluation ground truth**:

```text
Attack × poison condition × defense
             ↓
          CA / ASR
```

Then separately, the controller makes **one choice per attack/condition**.

So you'll eventually have:

```text
BadNets 1% → Controller chooses ANP
BadNets 5% → Controller chooses NAD
BadNets 10% → Controller chooses BaEraser
...
```

That choice should be recorded in the **controller table**, not replace the 36 matrix rows.

Your project explicitly wants the controller compared against **Always-FT, Always-ANP, Always-NAD, Always-BaEraser, and Oracle**. 

---

# 3. Controller input table — THIS is what I would store

Your controller is supposed to be **attack-agnostic and use AC + STRIP signals**, not attack identity. 

So I recommend this exact schema:

```markdown
| Attack | Poison Condition | AC Silhouette | AC PDR (%) | AC Poison Fraction (%) | STRIP TPR (%) | STRIP FPR (%) | STRIP Mean Entropy | STRIP Entropy Gap | Clean Confidence | Triggered Confidence | Confidence Gap | AC State | STRIP State | Controller Quadrant | Controller Score | Defense Chosen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|
| BadNets | 1% | | | | | | | | | | | | | | | |
| BadNets | 5% | | | | | | | | | | | | | | | |
| BadNets | 10% | | | | | | | | | | | | | | | |
| Blended | 1% | | | | | | | | | | | | | | | |
| Blended | 5% | | | | | | | | | | | | | | | |
| Blended | 10% | | | | | | | | | | | | | | | |
| Silent Killer | n=1 | | | | | | | | | | | | | | | |
| Silent Killer | n=10 | | | | | | | | | | | | | | | |
| Silent Killer | n=250 | | | | | | | | | | | | | | | |
```

### But there's an important distinction:

**The first 12-ish columns are inputs/signals.**

```text
AC Silhouette
AC PDR
AC Poison Fraction
STRIP TPR
STRIP FPR
STRIP entropy metrics
confidence metrics
```

**The last columns are controller outputs.**

```text
AC State
STRIP State
Controller Quadrant
Controller Score
Defense Chosen
```

Don't feed the latter back into the controller.

---

# 4. What the controller actually uses

Your locked controller is fundamentally:

```text
             AC
              │
       ┌──────┴──────┐
       │             │
      high          low
       │             │
   ┌───┴───┐     ┌───┴───┐
   │       │     │       │
 STRIP   STRIP  STRIP   STRIP
 high    low    high    low
```

with the important revised policy:

> **AC-low + STRIP-weak = low evidence, not "clean"; floor defense = ANP.** 

So the controller's core input is **not all those columns simultaneously as an ML feature vector**. It's a rule-based system using the AC/STRIP state variables.

---

# 5. I would actually keep a simpler "canonical controller sheet"

For the sheet your team will actually work with, I'd use:

```markdown
| Attack | Poison Condition | AC Silhouette | AC PDR (%) | STRIP TPR (%) | STRIP FPR (%) | STRIP Entropy Gap | AC State | STRIP State | Controller Quadrant | Defense Chosen | Reason |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| BadNets | 1% | | | | | | | | | | |
| BadNets | 5% | | | | | | | | | | | |
| BadNets | 10% | | | | | | | | | | | |
| Blended | 1% | | | | | | | | | | |
| Blended | 5% | | | | | | | | | | | |
| Blended | 10% | | | | | | | | | | |
| Silent Killer | n=1 | | | | | | | | | | |
| Silent Killer | n=10 | | | | | | | | | | |
| Silent Killer | n=250 | | | | | | | | | | |
```

Then your **main results section** has:

```text
ATTACK / CONTROLLER INPUT TABLE
        ↓
      9 rows
        ↓
CONTROLLER CHOICE
        ↓
POST-DEFENSE MATRIX
        ↓
      36 rows
        ↓
CA / ASR / compute
```

---

## One thing I would NOT add

I wouldn't make **poison rate** a controller input.

The controller shouldn't know:

```text
"this is BadNets 10%"
```

It gets the observed detector signals and decides. That's the entire point of your attack-agnostic controller. 

You keep `Attack` and `Poison Condition` in the spreadsheet **for ground-truth analysis**, not as controller features.

And similarly, **don't feed CA_after or ASR_after into the controller** — those only exist after the defense and are used to evaluate whether its choice was good.

### So the clean architecture of your spreadsheet is:

```text
TABLE 1 — ATTACK / PRE-DEFENSE
9 rows
Attack + PR + detector signals
             │
             ▼
TABLE 2 — CONTROLLER
9 rows
Quadrant + score + chosen defense
             │
             ▼
TABLE 3 — POST-DEFENSE MATRIX
36 rows
Attack × PR × Defense → CA/ASR
             │
             ▼
TABLE 4 — SUMMARY
Always-FT / Always-ANP / Always-NAD /
Always-BaEraser / Adaptive Controller / Oracle
```

That is the structure I'd lock now. It directly matches the revised project plan: full matrix, attack-agnostic controller, fixed-defense baselines, and oracle. 
