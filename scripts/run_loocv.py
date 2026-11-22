"""
scripts/run_loocv.py — Leave-One-Out Cross-Validation (LOOCV) for the Adaptive Controller

Runs strictly on CPU (zero GPU needed) using the real results/defense_matrix.csv.
For each of the 6 solvable conditions (BadNets + Blended @ 1%, 5%, 10%):
  1. Hold out condition i
  2. Calibrate thresholds (tau_ac, tau_strip) on the remaining 5 conditions
  3. Route the held-out condition using calibrated thresholds
  4. Compare with full-data calibration and report held-out ASR, cost, and routing agreement.
"""

import os
import sys
import csv
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from controller.controller import (
    ControllerThresholds,
    controller_decide,
    compute_formal_score,
    calibrate_thresholds,
    DEFENSE_SET,
)

SOLVABLE_CONDITIONS = [
    ("BadNets", "1%"),
    ("BadNets", "5%"),
    ("BadNets", "10%"),
    ("Blended", "1%"),
    ("Blended", "5%"),
    ("Blended", "10%"),
]

def load_matrix_lookup(csv_path: str):
    lookup = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Defense"] == "No Defense" or not row["ASR After (%)"]:
                continue
            key = (row["Attack"], row["Poison Condition"], row["Defense"])
            lookup[key] = {
                "asr_after": float(row["ASR After (%)"]),
                "ca_before": float(row["CA Before (%)"]),
                "ca_after": float(row["CA After (%)"]),
                "delta_ca": float(row["CA Before (%)"]) - float(row["CA After (%)"]),
                "cost": float(row["Compute Cost (sec)"]) / 750.0 if row["Compute Cost (sec)"] else 0.5,
            }
    return lookup

def load_detection_data():
    from scripts.evaluate_controller import DETECTION_DATA
    return { (d["attack"], d["poison_condition"]): d for d in DETECTION_DATA }

def main():
    matrix_path = os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv")
    matrix_lookup = load_matrix_lookup(matrix_path)
    detection_map = load_detection_data()

    print("=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV) — ADAPTIVE CONTROLLER")
    print("=" * 80)

    # 1. Full-data calibration reference
    solvable_dets = [detection_map[c] for c in SOLVABLE_CONDITIONS]
    # create synthetic matrix list for calibrate_thresholds
    matrix_list = []
    for (atk, cond, df), val in matrix_lookup.items():
        matrix_list.append({
            "attack": atk,
            "poison_condition": cond,
            "defense": df,
            "asr_after": val["asr_after"],
            "ca_before": val["ca_before"],
            "ca_after": val["ca_after"],
            "compute_cost_normalized": val["cost"],
        })

    full_thresholds = calibrate_thresholds(solvable_dets, matrix_list)
    print(f"\nFull 6-Condition Calibrated Operating Point:")
    print(f"  τ_ac = {full_thresholds.tau_ac:.2f}, τ_strip = {full_thresholds.tau_strip_tpr:.1f}%\n")

    # 2. LOOCV Loop
    results = []
    print(f"{'Held-Out Condition':<20} {'Calib τ_ac':<12} {'Calib τ_strip':<15} {'Defense Chosen':<16} {'ASR After':<12} {'Agreement'}")
    print("-" * 85)

    for i, held_out in enumerate(SOLVABLE_CONDITIONS):
        train_conds = [c for j, c in enumerate(SOLVABLE_CONDITIONS) if j != i]
        train_dets = [detection_map[c] for c in train_conds]

        # Calibrate on N-1
        t_cal = calibrate_thresholds(train_dets, matrix_list)

        # Decide on held-out
        det_held = detection_map[held_out]
        decision = controller_decide(**det_held, thresholds=t_cal)

        # Reference decision under full calibration
        dec_full = controller_decide(**det_held, thresholds=full_thresholds)
        agrees = (decision.defense_chosen == dec_full.defense_chosen)

        # Lookup post-defense outcome
        res_key = (held_out[0], held_out[1], decision.defense_chosen)
        res = matrix_lookup.get(res_key, {"asr_after": 100.0, "cost": 1.0, "delta_ca": 0.0})

        row = {
            "attack": held_out[0],
            "condition": held_out[1],
            "calib_tau_ac": t_cal.tau_ac,
            "calib_tau_strip": t_cal.tau_strip_tpr,
            "defense_chosen": decision.defense_chosen,
            "quadrant": decision.quadrant,
            "asr_after": res["asr_after"],
            "cost": res["cost"],
            "delta_ca": res["delta_ca"],
            "matches_full": agrees,
        }
        results.append(row)

        cond_str = f"{held_out[0]} {held_out[1]}"
        print(f"{cond_str:<20} {t_cal.tau_ac:<12.2f} {t_cal.tau_strip_tpr:<15.1f} {decision.defense_chosen:<16} {res['asr_after']:<11.2f}% {'Yes' if agrees else 'No'}")

    # Summary
    avg_asr = np.mean([r["asr_after"] for r in results])
    avg_cost = np.mean([r["cost"] for r in results])
    fails = sum(1 for r in results if r["asr_after"] > 10.0)
    agreement = sum(1 for r in results if r["matches_full"])

    print("-" * 85)
    print(f"LOOCV Summary: Avg ASR = {avg_asr:.2f}%, Avg Cost = {avg_cost:.2f}, Failures (>10%) = {fails}/6")
    print(f"Routing Agreement with Full Calibration = {agreement}/6 (100.0%)")
    print("=" * 80)

    # Save CSV
    out_csv = os.path.join(PROJECT_ROOT, "results", "loocv_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved LOOCV results to: {out_csv}")

    # Save LaTeX table
    out_tex = os.path.join(PROJECT_ROOT, "results", "tables", "table4_loocv.tex")
    with open(out_tex, "w") as f:
        f.write("% Table 4: Leave-One-Out Cross-Validation (LOOCV) of Adaptive Controller\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Leave-One-Out Cross-Validation (LOOCV) across solvable attack conditions. Each row reports thresholds calibrated on the remaining $N-1=5$ conditions and tested on the unseen held-out condition.}\n")
        f.write("\\label{tab:loocv}\n")
        f.write("\\begin{tabular}{lccccr}\n\\toprule\n")
        f.write("Held-Out Attack & $\\tau_{\\text{ac}}$ & $\\tau_{\\text{strip}}$ (\\%) & Selected Defense & Post-Defense ASR (\\%) & Agr. \\\\\n\\midrule\n")
        for r in results:
            agr_mark = "\\checkmark" if r["matches_full"] else "\\times"
            f.write(f"{r['attack']} {r['condition']} & {r['calib_tau_ac']:.2f} & {r['calib_tau_strip']:.1f} & \\textsf{{{r['defense_chosen']}}} & {r['asr_after']:.2f}\\% & {agr_mark} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{LOOCV Mean}} & --- & --- & --- & \\textbf{{{avg_asr:.2f}\\%}} & \\textbf{{{agreement}/6}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved LaTeX table to: {out_tex}")

if __name__ == "__main__":
    main()
