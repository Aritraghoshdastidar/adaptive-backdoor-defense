"""
controller/controller.py — Adaptive Hybrid Controller

Formal objective (from 03_CONTROLLER_LOGIC.md + controller_literature_grounding.md):

    d* = argmin_{d ∈ {FT, ANP, NAD, BaEraser}} [α·ASR_d + β·ΔCA_d + γ·C_d]

    where:
        ASR_d   = residual attack success rate after applying defense d
        ΔCA_d   = clean accuracy drop after applying defense d
        C_d     = normalized compute cost of defense d

This is approximated via the 2×2 AC+STRIP quadrant rule with empirically
calibrated thresholds. The controller is:
  - Rule-based (not learned)
  - Auditable (every decision explainable in one sentence)
  - Attack-agnostic (uses detector signals, never attack identity)

Literature grounding:
  - Kallas et al. (2025): attacker-side cost-weighted game → same structure
  - Ding & Liu: defender utility U_D = -BSR + λ·CA → algebraically equivalent

POLICY FIX (from SK blind-spot discovery):
  AC-low + STRIP-weak → "low evidence" (NOT "low severity") → floor defense = ANP
  This prevents catastrophic under-defense when both detectors are blind
  (e.g., Silent Killer low-n: ASR=99%+ but AC/STRIP both weak).
"""

import json
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ── Defense Set ───────────────────────────────────────────────────────────────

DEFENSE_SET = ["FT", "ANP", "NAD", "BaEraser-lite"]


# ── Controller Thresholds ─────────────────────────────────────────────────────
# These MUST be recalibrated from the completed pre-defense detection table.
# The values below are STARTER estimates from the existing data in
# Capstone Expt Logs - main.csv; replace them once the full 9-condition
# scatter is plotted.

@dataclass
class ControllerThresholds:
    """Empirically calibrated thresholds for the 2×2 quadrant decision.

    Calibration procedure (document in the paper's methods section):
    1. Plot AC silhouette vs STRIP TPR for all 9 attack conditions
    2. Find natural cluster boundaries in this 2D space
    3. Set τ_ac and τ_strip_tpr at those boundaries
    4. Validate: check that the quadrant assignments match the expected
       severity regimes from the defense × attack matrix
    """
    # AC severity threshold: silhouette score above this = "AC HIGH"
    tau_ac: float = 0.40

    # STRIP behavioral dominance threshold: TPR above this = "STRIP HIGH"
    tau_strip_tpr: float = 30.0  # percentage

    # Cost-aware objective weights (for formal score computation)
    alpha: float = 1.0    # weight on ASR (security — primary)
    beta: float = 0.5     # weight on ΔCA (utility loss)
    gamma: float = 0.1    # weight on compute cost (normalized 0-1)


# ── Controller Decision ──────────────────────────────────────────────────────

@dataclass
class ControllerDecision:
    """Full audit record for one controller invocation."""
    # Input condition (ground-truth metadata, NOT controller inputs)
    attack: str
    poison_condition: str

    # Detector signals (these ARE controller inputs)
    ac_silhouette: float
    ac_pdr: float
    ac_suspicious_fraction: float
    strip_tpr: float
    strip_fpr: float
    strip_mu: float
    strip_sigma: float

    # Controller outputs
    ac_state: str        # "HIGH" or "LOW"
    strip_state: str     # "HIGH" or "LOW"
    quadrant: str        # e.g. "AC-HIGH/STRIP-HIGH"
    defense_chosen: str  # one of DEFENSE_SET
    reason: str          # 1-sentence numeric reasoning

    # Optional: formal objective score if post-defense data available
    formal_score: Optional[float] = None


def classify_ac_state(ac_silhouette: float, thresholds: ControllerThresholds) -> str:
    """Classify AC severity as HIGH or LOW based on silhouette score."""
    return "HIGH" if ac_silhouette >= thresholds.tau_ac else "LOW"


def classify_strip_state(strip_tpr: float, thresholds: ControllerThresholds) -> str:
    """Classify STRIP behavioral dominance as HIGH or LOW based on TPR."""
    return "HIGH" if strip_tpr >= thresholds.tau_strip_tpr else "LOW"


def controller_decide(
    ac_silhouette: float,
    ac_pdr: float,
    ac_suspicious_fraction: float,
    strip_tpr: float,
    strip_fpr: float,
    strip_mu: float,
    strip_sigma: float,
    attack: str,
    poison_condition: str,
    thresholds: ControllerThresholds = None,
    use_old_policy: bool = False,
) -> ControllerDecision:
    """
    Run the adaptive controller on one condition.

    The controller uses ONLY detector signals (AC + STRIP) to choose a defense.
    It does NOT use attack identity — attack/poison_condition are logged for
    ground-truth analysis only.

    2×2 Quadrant Matrix:
    ┌─────────────┬────────────────────────┬────────────────────────┐
    │             │ STRIP HIGH (dominant)  │ STRIP LOW (weak)       │
    ├─────────────┼────────────────────────┼────────────────────────┤
    │ AC HIGH     │ BaEraser-lite          │ ANP                    │
    │ (embedded)  │ Both signals fire →    │ Structural fix         │
    │             │ heavy unlearning       │ sufficient             │
    ├─────────────┼────────────────────────┼────────────────────────┤
    │ AC LOW      │ NAD                    │ ANP (POLICY FIX)       │
    │ (stealthy)  │ Behavioral fix needed  │ "Low evidence" ≠       │
    │             │ AC blind → NAD         │ "low severity"         │
    └─────────────┴────────────────────────┴────────────────────────┘

    Args:
        use_old_policy: If True, uses the pre-fix policy where AC-low+STRIP-low
                        routes to FT. Used for the policy ablation experiment
                        (RQ3) to show the before/after difference.
    """
    if thresholds is None:
        thresholds = ControllerThresholds()

    ac_state = classify_ac_state(ac_silhouette, thresholds)
    strip_state = classify_strip_state(strip_tpr, thresholds)
    quadrant = f"AC-{ac_state}/STRIP-{strip_state}"

    # ── Quadrant → Defense mapping ──
    if ac_state == "HIGH" and strip_state == "HIGH":
        defense = "BaEraser-lite"
        reason = (
            f"AC silhouette={ac_silhouette:.2f}≥{thresholds.tau_ac} (HIGH), "
            f"STRIP TPR={strip_tpr:.1f}%≥{thresholds.tau_strip_tpr}% (HIGH) "
            f"→ both signals fire → heavy unlearning"
        )
    elif ac_state == "HIGH" and strip_state == "LOW":
        defense = "ANP"
        reason = (
            f"AC silhouette={ac_silhouette:.2f}≥{thresholds.tau_ac} (HIGH), "
            f"STRIP TPR={strip_tpr:.1f}%<{thresholds.tau_strip_tpr}% (LOW) "
            f"→ structurally embedded, behavioral dominance weak → pruning"
        )
    elif ac_state == "LOW" and strip_state == "HIGH":
        defense = "NAD"
        reason = (
            f"AC silhouette={ac_silhouette:.2f}<{thresholds.tau_ac} (LOW), "
            f"STRIP TPR={strip_tpr:.1f}%≥{thresholds.tau_strip_tpr}% (HIGH) "
            f"→ AC blind but behavioral signal present → attention distillation"
        )
    else:  # AC LOW + STRIP LOW
        if use_old_policy:
            defense = "FT"
            reason = (
                f"AC silhouette={ac_silhouette:.2f}<{thresholds.tau_ac} (LOW), "
                f"STRIP TPR={strip_tpr:.1f}%<{thresholds.tau_strip_tpr}% (LOW) "
                f"→ [OLD POLICY] assume low severity → fine-tune"
            )
        else:
            defense = "ANP"
            reason = (
                f"AC silhouette={ac_silhouette:.2f}<{thresholds.tau_ac} (LOW), "
                f"STRIP TPR={strip_tpr:.1f}%<{thresholds.tau_strip_tpr}% (LOW) "
                f"→ [POLICY FIX] low evidence ≠ low severity → floor defense ANP"
            )

    return ControllerDecision(
        attack=attack,
        poison_condition=poison_condition,
        ac_silhouette=ac_silhouette,
        ac_pdr=ac_pdr,
        ac_suspicious_fraction=ac_suspicious_fraction,
        strip_tpr=strip_tpr,
        strip_fpr=strip_fpr,
        strip_mu=strip_mu,
        strip_sigma=strip_sigma,
        ac_state=ac_state,
        strip_state=strip_state,
        quadrant=quadrant,
        defense_chosen=defense,
        reason=reason,
    )


# ── Formal Objective Score ────────────────────────────────────────────────────

def compute_formal_score(
    asr_after: float,
    ca_before: float,
    ca_after: float,
    compute_cost_normalized: float,
    thresholds: ControllerThresholds = None,
) -> float:
    """
    Compute the formal controller objective:
        Score = α·ASR_after + β·ΔCA + γ·C

    where ΔCA = ca_before - ca_after (positive = accuracy lost).

    Lower is better. This score is used to:
    1. Evaluate controller decisions against baselines
    2. Determine the Oracle (best defense per condition)

    Args:
        asr_after: ASR (%) after defense (0-100)
        ca_before: Clean accuracy (%) before defense
        ca_after: Clean accuracy (%) after defense
        compute_cost_normalized: Cost in [0, 1] (0=cheapest, 1=most expensive)
    """
    if thresholds is None:
        thresholds = ControllerThresholds()

    delta_ca = ca_before - ca_after  # positive = accuracy lost
    score = (
        thresholds.alpha * asr_after
        + thresholds.beta * delta_ca
        + thresholds.gamma * compute_cost_normalized * 100  # scale to same range
    )
    return round(score, 2)


# ── Threshold Calibration ─────────────────────────────────────────────────────

def calibrate_thresholds(
    detection_data: List[Dict],
    defense_matrix: List[Dict],
) -> ControllerThresholds:
    """
    Calibrate τ_ac and τ_strip_tpr from the pre-defense detection table
    and the defense × attack matrix.

    Procedure:
    1. Collect AC silhouette and STRIP TPR for all conditions
    2. For each candidate threshold pair, simulate controller decisions
    3. Look up the resulting defense in the matrix
    4. Pick the threshold pair that minimizes the average formal score

    Args:
        detection_data: List of dicts with keys:
            {attack, poison_condition, ac_silhouette, strip_tpr, ...}
        defense_matrix: List of dicts with keys:
            {attack, poison_condition, defense, ca_before, asr_before,
             ca_after, asr_after, compute_cost_normalized}

    Returns:
        Calibrated ControllerThresholds
    """
    # Build lookup: (attack, condition, defense) → post-defense metrics
    matrix_lookup = {}
    for row in defense_matrix:
        key = (row["attack"], row["poison_condition"], row["defense"])
        matrix_lookup[key] = row

    best_score = float("inf")
    best_thresholds = ControllerThresholds()

    # Grid search over candidate thresholds
    for tau_ac in [x / 100.0 for x in range(25, 65, 5)]:
        for tau_strip in range(10, 80, 10):
            thresholds = ControllerThresholds(tau_ac=tau_ac, tau_strip_tpr=float(tau_strip))

            total_score = 0.0
            n_valid = 0

            for det in detection_data:
                decision = controller_decide(
                    ac_silhouette=det["ac_silhouette"],
                    ac_pdr=det.get("ac_pdr", 0),
                    ac_suspicious_fraction=det.get("ac_suspicious_fraction", 0),
                    strip_tpr=det["strip_tpr"],
                    strip_fpr=det.get("strip_fpr", 0),
                    strip_mu=det.get("strip_mu", 0),
                    strip_sigma=det.get("strip_sigma", 0),
                    attack=det["attack"],
                    poison_condition=det["poison_condition"],
                    thresholds=thresholds,
                )

                key = (det["attack"], det["poison_condition"], decision.defense_chosen)
                if key in matrix_lookup:
                    row = matrix_lookup[key]
                    score = compute_formal_score(
                        asr_after=row["asr_after"],
                        ca_before=row["ca_before"],
                        ca_after=row["ca_after"],
                        compute_cost_normalized=row["compute_cost_normalized"],
                        thresholds=thresholds,
                    )
                    total_score += score
                    n_valid += 1

            if n_valid > 0:
                avg_score = total_score / n_valid
                if avg_score < best_score:
                    best_score = avg_score
                    best_thresholds = thresholds

    print(f"Calibrated thresholds: τ_ac={best_thresholds.tau_ac:.2f}, "
          f"τ_strip={best_thresholds.tau_strip_tpr:.0f}%")
    print(f"Best average formal score: {best_score:.2f}")

    return best_thresholds


# ── Controller vs. Baselines vs. Oracle ───────────────────────────────────────

@dataclass
class StrategyResult:
    """Aggregated result for one defense strategy across all conditions."""
    strategy_name: str
    avg_asr_after: float
    avg_ca_after: float
    avg_delta_ca: float
    avg_compute_cost: float
    avg_formal_score: float
    per_condition: List[Dict] = field(default_factory=list)


def evaluate_strategies(
    detection_data: List[Dict],
    defense_matrix: List[Dict],
    thresholds: ControllerThresholds = None,
) -> Dict[str, StrategyResult]:
    """
    Compare the adaptive controller against fixed-defense baselines and oracle.

    This produces the HEADLINE RESULT of the paper (Table/Figure):

    Strategy              | Avg ASR After | Avg CA After | Avg Cost | Score
    ──────────────────────|───────────────|──────────────|──────────|──────
    Always-FT             |     X%        |     X%       |   X      |  X
    Always-ANP            |     X%        |     X%       |   X      |  X
    Always-NAD            |     X%        |     X%       |   X      |  X
    Always-BaEraser-lite  |     X%        |     X%       |   X      |  X
    Adaptive Controller   |     X%        |     X%       |   X      |  X
    Oracle (best per-cell)|     X%        |     X%       |   X      |  X

    Returns dict mapping strategy name to StrategyResult.
    """
    if thresholds is None:
        thresholds = ControllerThresholds()

    # Build lookup: (attack, condition, defense) → row
    matrix_lookup = {}
    for row in defense_matrix:
        key = (row["attack"], row["poison_condition"], row["defense"])
        matrix_lookup[key] = row

    # Get all unique conditions
    conditions = [(d["attack"], d["poison_condition"]) for d in detection_data]

    results = {}

    # ── Fixed-defense baselines ──
    for defense in DEFENSE_SET:
        strategy_name = f"Always-{defense}"
        per_condition = []
        for attack, condition in conditions:
            key = (attack, condition, defense)
            if key in matrix_lookup:
                row = matrix_lookup[key]
                score = compute_formal_score(
                    asr_after=row["asr_after"],
                    ca_before=row["ca_before"],
                    ca_after=row["ca_after"],
                    compute_cost_normalized=row["compute_cost_normalized"],
                    thresholds=thresholds,
                )
                per_condition.append({
                    "attack": attack,
                    "condition": condition,
                    "defense": defense,
                    "asr_after": row["asr_after"],
                    "ca_after": row["ca_after"],
                    "delta_ca": row["ca_before"] - row["ca_after"],
                    "compute_cost": row["compute_cost_normalized"],
                    "formal_score": score,
                })

        if per_condition:
            results[strategy_name] = StrategyResult(
                strategy_name=strategy_name,
                avg_asr_after=_mean([r["asr_after"] for r in per_condition]),
                avg_ca_after=_mean([r["ca_after"] for r in per_condition]),
                avg_delta_ca=_mean([r["delta_ca"] for r in per_condition]),
                avg_compute_cost=_mean([r["compute_cost"] for r in per_condition]),
                avg_formal_score=_mean([r["formal_score"] for r in per_condition]),
                per_condition=per_condition,
            )

    # ── Adaptive Controller ──
    per_condition_adaptive = []
    for det in detection_data:
        decision = controller_decide(
            ac_silhouette=det["ac_silhouette"],
            ac_pdr=det.get("ac_pdr", 0),
            ac_suspicious_fraction=det.get("ac_suspicious_fraction", 0),
            strip_tpr=det["strip_tpr"],
            strip_fpr=det.get("strip_fpr", 0),
            strip_mu=det.get("strip_mu", 0),
            strip_sigma=det.get("strip_sigma", 0),
            attack=det["attack"],
            poison_condition=det["poison_condition"],
            thresholds=thresholds,
        )
        key = (det["attack"], det["poison_condition"], decision.defense_chosen)
        if key in matrix_lookup:
            row = matrix_lookup[key]
            score = compute_formal_score(
                asr_after=row["asr_after"],
                ca_before=row["ca_before"],
                ca_after=row["ca_after"],
                compute_cost_normalized=row["compute_cost_normalized"],
                thresholds=thresholds,
            )
            per_condition_adaptive.append({
                "attack": det["attack"],
                "condition": det["poison_condition"],
                "defense": decision.defense_chosen,
                "quadrant": decision.quadrant,
                "reason": decision.reason,
                "asr_after": row["asr_after"],
                "ca_after": row["ca_after"],
                "delta_ca": row["ca_before"] - row["ca_after"],
                "compute_cost": row["compute_cost_normalized"],
                "formal_score": score,
            })

    if per_condition_adaptive:
        results["Adaptive Controller"] = StrategyResult(
            strategy_name="Adaptive Controller",
            avg_asr_after=_mean([r["asr_after"] for r in per_condition_adaptive]),
            avg_ca_after=_mean([r["ca_after"] for r in per_condition_adaptive]),
            avg_delta_ca=_mean([r["delta_ca"] for r in per_condition_adaptive]),
            avg_compute_cost=_mean([r["compute_cost"] for r in per_condition_adaptive]),
            avg_formal_score=_mean([r["formal_score"] for r in per_condition_adaptive]),
            per_condition=per_condition_adaptive,
        )

    # ── Adaptive Controller (OLD POLICY — for ablation) ──
    per_condition_old = []
    for det in detection_data:
        decision = controller_decide(
            ac_silhouette=det["ac_silhouette"],
            ac_pdr=det.get("ac_pdr", 0),
            ac_suspicious_fraction=det.get("ac_suspicious_fraction", 0),
            strip_tpr=det["strip_tpr"],
            strip_fpr=det.get("strip_fpr", 0),
            strip_mu=det.get("strip_mu", 0),
            strip_sigma=det.get("strip_sigma", 0),
            attack=det["attack"],
            poison_condition=det["poison_condition"],
            thresholds=thresholds,
            use_old_policy=True,
        )
        key = (det["attack"], det["poison_condition"], decision.defense_chosen)
        if key in matrix_lookup:
            row = matrix_lookup[key]
            score = compute_formal_score(
                asr_after=row["asr_after"],
                ca_before=row["ca_before"],
                ca_after=row["ca_after"],
                compute_cost_normalized=row["compute_cost_normalized"],
                thresholds=thresholds,
            )
            per_condition_old.append({
                "attack": det["attack"],
                "condition": det["poison_condition"],
                "defense": decision.defense_chosen,
                "quadrant": decision.quadrant,
                "reason": decision.reason,
                "asr_after": row["asr_after"],
                "ca_after": row["ca_after"],
                "delta_ca": row["ca_before"] - row["ca_after"],
                "compute_cost": row["compute_cost_normalized"],
                "formal_score": score,
            })

    if per_condition_old:
        results["Adaptive (Old Policy)"] = StrategyResult(
            strategy_name="Adaptive (Old Policy)",
            avg_asr_after=_mean([r["asr_after"] for r in per_condition_old]),
            avg_ca_after=_mean([r["ca_after"] for r in per_condition_old]),
            avg_delta_ca=_mean([r["delta_ca"] for r in per_condition_old]),
            avg_compute_cost=_mean([r["compute_cost"] for r in per_condition_old]),
            avg_formal_score=_mean([r["formal_score"] for r in per_condition_old]),
            per_condition=per_condition_old,
        )

    # ── Oracle (best defense per condition) ──
    per_condition_oracle = []
    for attack, condition in conditions:
        best_score = float("inf")
        best_row = None
        for defense in DEFENSE_SET:
            key = (attack, condition, defense)
            if key in matrix_lookup:
                row = matrix_lookup[key]
                score = compute_formal_score(
                    asr_after=row["asr_after"],
                    ca_before=row["ca_before"],
                    ca_after=row["ca_after"],
                    compute_cost_normalized=row["compute_cost_normalized"],
                    thresholds=thresholds,
                )
                if score < best_score:
                    best_score = score
                    best_row = row
                    best_defense = defense

        if best_row is not None:
            per_condition_oracle.append({
                "attack": attack,
                "condition": condition,
                "defense": best_defense,
                "asr_after": best_row["asr_after"],
                "ca_after": best_row["ca_after"],
                "delta_ca": best_row["ca_before"] - best_row["ca_after"],
                "compute_cost": best_row["compute_cost_normalized"],
                "formal_score": best_score,
            })

    if per_condition_oracle:
        results["Oracle"] = StrategyResult(
            strategy_name="Oracle",
            avg_asr_after=_mean([r["asr_after"] for r in per_condition_oracle]),
            avg_ca_after=_mean([r["ca_after"] for r in per_condition_oracle]),
            avg_delta_ca=_mean([r["delta_ca"] for r in per_condition_oracle]),
            avg_compute_cost=_mean([r["compute_cost"] for r in per_condition_oracle]),
            avg_formal_score=_mean([r["formal_score"] for r in per_condition_oracle]),
            per_condition=per_condition_oracle,
        )

    return results


# ── Output / Logging ──────────────────────────────────────────────────────────

def print_strategy_comparison(results: Dict[str, StrategyResult]):
    """Print the headline comparison table."""
    print("\n" + "=" * 90)
    print("CONTROLLER vs. BASELINES vs. ORACLE — Headline Comparison")
    print("=" * 90)
    print(f"{'Strategy':<25} {'Avg ASR↓':>10} {'Avg CA↑':>10} {'Avg ΔCA':>10} "
          f"{'Avg Cost':>10} {'Score↓':>10}")
    print("-" * 90)

    # Print in a fixed order
    order = [
        "Always-FT", "Always-ANP", "Always-NAD", "Always-BaEraser-lite",
        "Adaptive (Old Policy)", "Adaptive Controller", "Oracle"
    ]
    for name in order:
        if name in results:
            r = results[name]
            print(f"{r.strategy_name:<25} {r.avg_asr_after:>9.1f}% {r.avg_ca_after:>9.1f}% "
                  f"{r.avg_delta_ca:>9.2f}pp {r.avg_compute_cost:>9.2f} "
                  f"{r.avg_formal_score:>9.2f}")

    print("=" * 90)
    print("Score = α·ASR + β·ΔCA + γ·C×100  (lower is better)")
    print()


def save_controller_decisions(decisions: List[ControllerDecision], filepath: str):
    """Save all controller decisions to CSV for audit trail."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = list(asdict(decisions[0]).keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in decisions:
            writer.writerow(asdict(d))
    print(f"Saved {len(decisions)} controller decisions to {filepath}")


def save_strategy_results(results: Dict[str, StrategyResult], filepath: str):
    """Save strategy comparison to JSON for figures."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    output = {}
    for name, r in results.items():
        output[name] = {
            "avg_asr_after": r.avg_asr_after,
            "avg_ca_after": r.avg_ca_after,
            "avg_delta_ca": r.avg_delta_ca,
            "avg_compute_cost": r.avg_compute_cost,
            "avg_formal_score": r.avg_formal_score,
            "per_condition": r.per_condition,
        }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved strategy results to {filepath}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(values):
    return round(sum(values) / len(values), 2) if values else 0.0


# ── Demo / Quick Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick demo using the data we already have from the CSV.
    This shows what the controller WOULD decide for each condition
    based on pre-defense detector signals.
    """
    # Pre-defense detection data from Capstone Expt Logs - main.csv
    detection_data = [
        {"attack": "BadNets", "poison_condition": "1%",
         "ac_silhouette": 0.32, "ac_pdr": 9.72, "ac_suspicious_fraction": 0.46,
         "strip_tpr": 7.6, "strip_fpr": 1.6, "strip_mu": 0.64, "strip_sigma": 0.16},
        {"attack": "BadNets", "poison_condition": "5%",
         "ac_silhouette": 0.37, "ac_pdr": 75.97, "ac_suspicious_fraction": 0.35,
         "strip_tpr": 34.4, "strip_fpr": 1.6, "strip_mu": 0.69, "strip_sigma": 0.17},
        {"attack": "BadNets", "poison_condition": "10%",
         "ac_silhouette": 0.44, "ac_pdr": 97.15, "ac_suspicious_fraction": 0.48,
         "strip_tpr": 33.2, "strip_fpr": 4.0, "strip_mu": 0.69, "strip_sigma": 0.19},
        {"attack": "Blended", "poison_condition": "1%",
         "ac_silhouette": 0.38, "ac_pdr": 11.16, "ac_suspicious_fraction": 0.27,
         "strip_tpr": 79.6, "strip_fpr": 1.2, "strip_mu": 0.61, "strip_sigma": 0.18},
        {"attack": "Blended", "poison_condition": "5%",
         "ac_silhouette": 0.54, "ac_pdr": 99.76, "ac_suspicious_fraction": 0.33,
         "strip_tpr": 92.8, "strip_fpr": 2.4, "strip_mu": 0.67, "strip_sigma": 0.18},
        {"attack": "Blended", "poison_condition": "10%",
         "ac_silhouette": 0.64, "ac_pdr": 99.96, "ac_suspicious_fraction": 0.50,
         "strip_tpr": 68.0, "strip_fpr": 1.6, "strip_mu": 0.68, "strip_sigma": 0.19},
        {"attack": "Silent Killer", "poison_condition": "0.5%",
         "ac_silhouette": 0.33, "ac_pdr": 4.32, "ac_suspicious_fraction": 0.50,
         "strip_tpr": 52.4, "strip_fpr": 2.4, "strip_mu": 0.78, "strip_sigma": 0.20},
        {"attack": "Silent Killer", "poison_condition": "1%",
         "ac_silhouette": 0.33, "ac_pdr": 15.0, "ac_suspicious_fraction": 0.50,
         "strip_tpr": 8.4, "strip_fpr": 2.8, "strip_mu": 0.72, "strip_sigma": 0.19},
        {"attack": "Silent Killer", "poison_condition": "5%",
         "ac_silhouette": 0.37, "ac_pdr": 69.13, "ac_suspicious_fraction": 0.41,
         "strip_tpr": 85.6, "strip_fpr": 1.6, "strip_mu": 0.72, "strip_sigma": 0.17},
    ]

    print("=" * 70)
    print("ADAPTIVE CONTROLLER — Demo Run")
    print("=" * 70)

    thresholds = ControllerThresholds()
    decisions = []

    for det in detection_data:
        decision = controller_decide(**det, thresholds=thresholds)
        decisions.append(decision)
        print(f"\n{decision.attack} {decision.poison_condition}:")
        print(f"  Quadrant: {decision.quadrant}")
        print(f"  Defense:  {decision.defense_chosen}")
        print(f"  Reason:   {decision.reason}")

    # Also show old policy for comparison
    print("\n" + "=" * 70)
    print("OLD POLICY COMPARISON (AC-low+STRIP-low → FT)")
    print("=" * 70)
    for det in detection_data:
        old = controller_decide(**det, thresholds=thresholds, use_old_policy=True)
        new = controller_decide(**det, thresholds=thresholds, use_old_policy=False)
        if old.defense_chosen != new.defense_chosen:
            print(f"\n  {det['attack']} {det['poison_condition']}:")
            print(f"    OLD: {old.defense_chosen}")
            print(f"    NEW: {new.defense_chosen}  ← policy fix")
