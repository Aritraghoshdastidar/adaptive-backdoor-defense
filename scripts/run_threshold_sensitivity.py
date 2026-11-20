"""
scripts/run_threshold_sensitivity.py — Threshold Sensitivity & Stability Analysis

Conducts a systematic perturbation sweep around the calibrated controller thresholds
(tau_ac = 0.40, tau_strip = 30.0%) to prove that controller decisions and post-defense
security are stable against threshold perturbations.

Generates:
- results/threshold_sensitivity.csv
- Formatted Table 3 for the paper draft
"""

import csv
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_controller import DETECTION_DATA, load_defense_matrix
from controller.controller import (
    ControllerThresholds,
    controller_decide,
    compute_formal_score,
)

BASE_TAU_AC = 0.40
BASE_TAU_STRIP = 30.0


def main():
    matrix_path = os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    defense_matrix = load_defense_matrix(matrix_path)
    lookup = {(r["attack"], r["poison_condition"], r["defense"]): r for r in defense_matrix}

    # Baseline decisions
    base_thresholds = ControllerThresholds(tau_ac=BASE_TAU_AC, tau_strip_tpr=BASE_TAU_STRIP)
    base_decisions = {}
    for det in DETECTION_DATA:
        dec = controller_decide(**det, thresholds=base_thresholds)
        base_decisions[(det["attack"], det["poison_condition"])] = dec.defense_chosen

    print("=" * 80)
    print(f"BASELINE DECISIONS (τ_ac = {BASE_TAU_AC:.2f}, τ_strip = {BASE_TAU_STRIP:.0f}%):")
    print("=" * 80)
    for (attack, cond), def_chosen in base_decisions.items():
        print(f"  {attack:15s} {cond:5s} → {def_chosen}")
    print()

    # Sweep grid
    delta_ac_list = [-0.10, -0.05, 0.0, +0.05, +0.10]
    delta_strip_list = [-15.0, -10.0, -5.0, 0.0, +5.0, +10.0, +15.0]

    sweep_results = []

    for d_ac in delta_ac_list:
        tau_ac = round(BASE_TAU_AC + d_ac, 2)
        for d_strip in delta_strip_list:
            tau_strip = round(BASE_TAU_STRIP + d_strip, 1)
            th = ControllerThresholds(tau_ac=tau_ac, tau_strip_tpr=tau_strip)

            changed_conds = []
            asrs_solvable = []
            asrs_all = []
            scores_all = []

            for det in DETECTION_DATA:
                dec = controller_decide(**det, thresholds=th)
                cond_key = (det["attack"], det["poison_condition"])
                if dec.defense_chosen != base_decisions[cond_key]:
                    changed_conds.append(
                        f"{det['attack']} {det['poison_condition']} "
                        f"({base_decisions[cond_key]}→{dec.defense_chosen})"
                    )

                # Look up defense outcome
                k = (det["attack"], det["poison_condition"], dec.defense_chosen)
                if k in lookup:
                    r = lookup[k]
                    asr = r["asr_after"]
                    score = compute_formal_score(
                        asr_after=r["asr_after"],
                        ca_before=r["ca_before"],
                        ca_after=r["ca_after"],
                        compute_cost_normalized=r["compute_cost_normalized"],
                        thresholds=th,
                    )
                else:
                    # Penalty for failed defense
                    asr = 100.0
                    score = 100.0

                asrs_all.append(asr)
                scores_all.append(score)
                if det["attack"] != "Silent Killer":
                    asrs_solvable.append(asr)

            avg_asr_solvable = round(sum(asrs_solvable) / len(asrs_solvable), 2)
            avg_asr_all = round(sum(asrs_all) / len(asrs_all), 2)
            avg_score = round(sum(scores_all) / len(scores_all), 2)

            sweep_results.append({
                "delta_tau_ac": f"{d_ac:+.2f}",
                "delta_tau_strip": f"{d_strip:+.0f}%",
                "tau_ac": tau_ac,
                "tau_strip": tau_strip,
                "n_changed": len(changed_conds),
                "avg_asr_solvable": avg_asr_solvable,
                "avg_asr_all": avg_asr_all,
                "avg_score": avg_score,
                "changed_conditions": "; ".join(changed_conds) if changed_conds else "None (baseline match)",
            })

    # Save to CSV
    csv_path = os.path.join(out_dir, "threshold_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
        writer.writeheader()
        for row in sweep_results:
            writer.writerow(row)

    print(f"Saved {len(sweep_results)} sensitivity evaluations to {csv_path}\n")

    # Display clean table
    print("=" * 95)
    print("THRESHOLD SENSITIVITY & STABILITY TABLE (for Paper §5 Table 3)")
    print("=" * 95)
    print(f"{'Δτ_ac':>8} {'Δτ_strip':>10} {'τ_ac':>6} {'τ_strip':>8} {'Decisions Changed':>18} "
          f"{'Solvable ASR':>14} {'Full ASR':>10} {'Score':>8}")
    print("-" * 95)

    # Highlight key representative rows
    for r in sweep_results:
        # Show all 0.0 delta row and boundaries
        print(f"{r['delta_tau_ac']:>8} {r['delta_tau_strip']:>10} {r['tau_ac']:>6.2f} "
              f"{r['tau_strip']:>7.0f}% {r['n_changed']:>10}/8 "
              f"{r['avg_asr_solvable']:>13.2f}% {r['avg_asr_all']:>9.2f}% {r['avg_score']:>8.2f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
