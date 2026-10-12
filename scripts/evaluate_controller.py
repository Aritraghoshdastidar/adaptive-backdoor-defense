"""
scripts/evaluate_controller.py — Run the full controller evaluation

This script:
1. Loads the pre-defense detection data
2. Loads the completed defense × attack matrix
3. Optionally calibrates thresholds (if --calibrate flag)
4. Runs the controller on all 9 conditions
5. Computes controller vs. baselines vs. oracle
6. Runs the old-vs-new policy ablation (RQ3)
7. Saves all results + generates summary tables

Usage:
    python scripts/evaluate_controller.py
    python scripts/evaluate_controller.py --calibrate
    python scripts/evaluate_controller.py --matrix results/defense_matrix.csv

This is the script that produces the HEADLINE RESULT of the paper.
"""

import argparse
import csv
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from controller.controller import (
    ControllerThresholds,
    controller_decide,
    compute_formal_score,
    calibrate_thresholds,
    evaluate_strategies,
    print_strategy_comparison,
    save_controller_decisions,
    save_strategy_results,
    DEFENSE_SET,
)


# ── Data Loading ─────────────────────────────────────────────────────────────

# Pre-defense detection data from Capstone Expt Logs - main.csv
# This is the CONTROLLER INPUT table (9 rows)
DETECTION_DATA = [
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


def load_defense_matrix(filepath: str) -> list:
    """Load the defense × attack matrix CSV into a list of dicts."""
    matrix = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip 'No Defense' rows and rows with missing data
            if row["Defense"] == "No Defense":
                continue
            if not row["CA After (%)"] or not row["ASR After (%)"]:
                continue

            matrix.append({
                "attack": row["Attack"],
                "poison_condition": row["Poison Condition"],
                "defense": row["Defense"],
                "ca_before": float(row["CA Before (%)"]),
                "asr_before": float(row["ASR Before (%)"]),
                "ca_after": float(row["CA After (%)"]),
                "asr_after": float(row["ASR After (%)"]),
                "compute_cost_normalized": _normalize_cost(
                    row["Defense"],
                    float(row["Compute Cost (sec)"]) if row["Compute Cost (sec)"] else 0
                ),
            })

    print(f"Loaded {len(matrix)} cells from defense matrix")
    return matrix


def _normalize_cost(defense: str, raw_seconds: float) -> float:
    """
    Normalize compute cost to [0, 1] range.

    If raw timing data is available, normalize by the maximum observed cost.
    Otherwise, use the expected relative cost ranking:
        FT < ANP < NAD < BaEraser-lite
    """
    if raw_seconds > 0:
        # Will be properly normalized once all data is in
        return min(raw_seconds / 600.0, 1.0)  # assume max ~10 min

    # Fallback: relative cost based on literature/experience
    cost_map = {
        "FT": 0.1,
        "ANP": 0.4,
        "NAD": 0.5,
        "BaEraser-lite": 0.9,
    }
    return cost_map.get(defense, 0.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the adaptive controller")
    parser.add_argument("--matrix", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv"),
                        help="Path to the defense × attack matrix CSV")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate thresholds from data (otherwise use defaults)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"),
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    print("\n" + "=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    defense_matrix = load_defense_matrix(args.matrix)

    if len(defense_matrix) == 0:
        print("\n⚠️  No complete cells found in the defense matrix!")
        print("   Fill in the empty cells in results/defense_matrix.csv first.")
        print("   The controller evaluation needs post-defense CA/ASR data.\n")
        print("   Currently available cells:")
        _show_matrix_coverage(args.matrix)
        return

    # ── Calibrate or use defaults ──
    print("\n" + "=" * 70)
    print("STEP 2: Controller thresholds")
    print("=" * 70)

    if args.calibrate:
        thresholds = calibrate_thresholds(DETECTION_DATA, defense_matrix)
    else:
        thresholds = ControllerThresholds()
        print(f"Using default thresholds: τ_ac={thresholds.tau_ac}, "
              f"τ_strip={thresholds.tau_strip_tpr}%")

    # ── Run controller on all conditions ──
    print("\n" + "=" * 70)
    print("STEP 3: Controller decisions (new policy)")
    print("=" * 70)

    decisions = []
    for det in DETECTION_DATA:
        decision = controller_decide(**det, thresholds=thresholds)
        decisions.append(decision)
        print(f"  {decision.attack:15s} {decision.poison_condition:5s} → "
              f"{decision.quadrant:25s} → {decision.defense_chosen}")

    save_controller_decisions(
        decisions,
        os.path.join(args.output_dir, "controller_decisions.csv")
    )

    # ── Old policy comparison ──
    print("\n" + "=" * 70)
    print("STEP 4: Policy ablation (old vs. new)")
    print("=" * 70)

    print("\nConditions where old and new policy DIFFER:")
    for det in DETECTION_DATA:
        old = controller_decide(**det, thresholds=thresholds, use_old_policy=True)
        new = controller_decide(**det, thresholds=thresholds, use_old_policy=False)
        if old.defense_chosen != new.defense_chosen:
            print(f"  {det['attack']:15s} {det['poison_condition']:5s}: "
                  f"{old.defense_chosen} → {new.defense_chosen}")

    # ── Strategy comparison ──
    print("\n" + "=" * 70)
    print("STEP 5: Controller vs. Baselines vs. Oracle")
    print("=" * 70)

    results = evaluate_strategies(DETECTION_DATA, defense_matrix, thresholds)
    print_strategy_comparison(results)

    save_strategy_results(
        results,
        os.path.join(args.output_dir, "strategy_comparison.json")
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("DONE — Results saved to:")
    print("=" * 70)
    print(f"  Controller decisions: {args.output_dir}/controller_decisions.csv")
    print(f"  Strategy comparison:  {args.output_dir}/strategy_comparison.json")
    print()

    # Show coverage
    n_filled = len(defense_matrix)
    n_total = len(DETECTION_DATA) * len(DEFENSE_SET)
    n_missing = n_total - n_filled
    if n_missing > 0:
        print(f"⚠️  Matrix coverage: {n_filled}/{n_total} cells filled "
              f"({n_missing} missing)")
        print("   Results above are PARTIAL — fill remaining cells for final numbers.")
    else:
        print(f"✅ Matrix coverage: {n_filled}/{n_total} cells — COMPLETE")


def _show_matrix_coverage(filepath: str):
    """Show which cells in the matrix have data."""
    filled = set()
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Defense"] != "No Defense" and row["CA After (%)"]:
                    filled.add((row["Attack"], row["Poison Condition"], row["Defense"]))
    except FileNotFoundError:
        print(f"   File not found: {filepath}")
        return

    attacks = ["BadNets", "Blended", "Silent Killer"]
    conditions = {
        "BadNets": ["1%", "5%", "10%"],
        "Blended": ["1%", "5%", "10%"],
        "Silent Killer": ["0.5%", "1%", "5%"],
    }

    print(f"\n   {'':20s} {'FT':>10s} {'ANP':>10s} {'NAD':>10s} {'BaEraser':>10s}")
    print("   " + "-" * 60)
    for attack in attacks:
        for cond in conditions[attack]:
            row = f"   {attack:12s} {cond:5s}"
            for defense in DEFENSE_SET:
                if (attack, cond, defense) in filled:
                    row += f"{'✅':>10s}"
                else:
                    row += f"{'—':>10s}"
            print(row)
    print()


if __name__ == "__main__":
    main()
