# Controller Objective — Literature Grounding

**Purpose of this doc:** capture where our controller's decision formula came from
originally, what we found when we went looking for literature backing, and how
that reshapes (not replaces) our existing design. This supplements
`03_CONTROLLER_LOGIC.md`, which remains the locked spec — this doc explains
*why* that spec is defensible against a reviewer asking "where does this
formula come from?"

---

## 1. Where We Were — the original formulation

From `03_CONTROLLER_LOGIC.md`, locked early in the project:

**State vector:**
```
s = (d, a, u, c)
```
- `d` — AC severity score
- `a` — estimated ASR
- `u` — utility loss proxy (clean accuracy drop)
- `c` — compute cost

**Objective:**
```
min   a(s) + λ·u(s) + μ·c(s)
 π
```

**Framing we committed to:** *not* a learned policy (no RL, no neural controller).
Thresholds are empirically calibrated offline; the decision rule stays fixed
and auditable. Reviewer-safe sentence already in use:

> "Thresholds are empirically calibrated via offline validation to minimize
> ASR + utility loss under compute constraints."

This was an **original formulation** — a cost-constrained decision problem we
set up ourselves, not derived from a specific paper. It was defensible on
first-principles grounds (security, utility, and cost are the three things
that obviously matter), but had no direct literature citation behind its
exact shape.

---

## 2. What We Found — two game-theoretic references

Searching for prior work on cost-aware backdoor defense decision-making
surfaced two recent papers that independently frame backdoor
attack/defense as a **cost-weighted optimization problem** — structurally the
same idea as our objective, arrived at from a game-theory angle rather than a
systems angle.

### [4] Kallas, Tannous & Faraoun (2025) — *A Zero-Sum Game-Theoretic Analysis
for Cost-Aware Backdoor Attacks and Defenses in Deep Learning*
*Journal of Intelligent Communication, 4(2), 74–92*

Models the backdoor problem as a **zero-sum game between attacker and
defender**, each with their own strategy space and cost-weighted utility.

**Strategy spaces:**
```
S_A = (α_tr, Δ_tr, Δ_ts) ∈ [0,1]³      (attacker's strategy)
S_D = Δ_def ∈ [0,1]                    (defender's strategy)
```

**Attacker's utility:**
```
μ_A = ASR × 1[CDA > CDA_inf] − λ_A·C_A(α_tr, Δ_tr, Δ_ts) − λ_D·C_D(Δ_def)
```

The attacker only "wins" if ASR is high **and** clean-data accuracy (CDA)
stays above an acceptable floor (`CDA_inf`) — i.e. a stealthy, undetected
attack — and pays a cost penalty for how aggressive its own strategy is,
plus a term depending on the defender's chosen defense strength.

**What this validates for us:** the core idea that **cost penalizes both
sides**, not just the defender, and that "success" is conditioned on staying
under a detectability/accuracy constraint — the same spirit as our
`u(s)` utility-loss term, just modeled from the attacker's side too.

### [3] Ding & Liu — *A Game-Theoretic Framework for Strategic Machine
Unlearning in Backdoor Mitigation*
*City University of Macau*

Frames the **defender's side** as a utility-maximization problem directly:

```
U_D(π_D, π_A) = −BSR(π_D, π_A) + λ·CA(π_D, π_A)
```

Defender utility is **negative Backdoor Success Rate, plus a weighted clean-
accuracy term** — i.e. almost exactly our `−a(s) + λ·u(s)` shape, just with
opposite sign convention (utility to maximize vs. cost to minimize) and BSR
in place of ASR.

It also defines a **per-sample influence score** for selecting which training
points to target during unlearning:

```
s_i = −∇_θ L_bd(θ*)ᵀ H_θ*⁻¹ ∇_θ ℓ(f_θ*(xᵢ), yᵢ)
```

This is an **influence-function**-style score (Koh & Liang lineage): it
estimates how much removing/down-weighting a given training sample `xᵢ`
would reduce the backdoor loss `L_bd`, using the inverse Hessian of the loss
landscape at the converged model `θ*`. This is a different, finer-grained
tool than anything currently in our pipeline — it operates per-sample, not
per-checkpoint-severity — and is **not** something we are adopting; it's
listed here as related work, not a component we're building.

**What this validates for us:** the defender-side objective — "minimize
backdoor success while protecting clean accuracy" — is the same two-term
trade-off we already had, from an independently-derived, published source.

---

## 3. How This Reshapes Our Formula (Refinement, Not Replacement)

The two references don't change our underlying design decision (rule-based,
calibrated, two-signal controller stays locked). What they do is let us
**tighten the notation** to make the state vector and decision rule more
explicit and more clearly literature-grounded:

**Updated state vector** — makes the two detection signals explicit instead
of folding them into a single `d`:
```
s = (D_ac, D_strip, ASR, ΔCA, C)
```
- `D_ac` — AC severity signal (silhouette / suspicious_fraction)
- `D_strip` — STRIP behavioral signal (TPR / entropy)
- `ASR` — residual attack success rate
- `ΔCA` — clean accuracy drop (utility loss)
- `C` — compute cost

**Updated decision rule** — written as an explicit argmin over our actual
discrete defense set, rather than an abstract `min_π`:
```
d* = argmin   [ α·ASR_d + β·ΔCA_d + γ·C_d ]
     d ∈ {FT, PF, UL}
```
where `FT` = fine-tune, `PF` = prune+fine-tune, `UL` = unlearning
(BAERASER-lite tier), and `α, β, γ` play the same role as our original
`1, λ, μ` weights — just renamed for consistency with how the external papers
write their cost-weighted objectives.

This is **the same formula as `03_CONTROLLER_LOGIC.md`**, restated so that
(a) the two detection signals that actually feed our quadrant matrix are
visible in the state vector instead of hidden inside a single `d`, and (b)
the defense choice is written as a literal minimization over our actual
candidate set, matching the style both papers use. Nothing about the
controller's logic, thresholds, or quadrant structure changes.

**One-line summary of the whole pipeline**, now stated with literature
behind the cost-aware framing rather than asserted from first principles:

> Detect → estimate severity → evaluate defense trade-off → choose the
> cheapest defense that sufficiently reduces ASR without unacceptable CA
> loss.

---

## 4. What to Say to the Guide / Reviewer

> "Our controller's objective — minimize attack success subject to utility
> and compute cost — was originally derived independently from first
> principles. Searching the literature afterward, we found two recent papers
> that frame backdoor attack/defense as a cost-weighted game-theoretic
> problem from both the attacker's and defender's side, and their formulations
> converge on the same two-term trade-off (security vs. utility, cost-
> weighted) that we had already built. We've since updated our notation to
> make the state vector and decision rule explicit in the same style, which
> strengthens the formal grounding of the objective without changing the
> controller's actual logic or thresholds."

This is a strong thing to say in a review: the formula wasn't retrofitted to
match a paper — it was designed independently and then found to agree with
published work, which is evidence the design is principled rather than
arbitrary.
