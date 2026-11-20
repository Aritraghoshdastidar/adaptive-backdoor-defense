"""
scripts/export_latex_tables.py — Export Results as Camera-Ready LaTeX Tables

Reads results from:
- results/strategy_comparison.json & results/strategy_comparison_solvable.json
- results/controller_decisions.csv
- results/threshold_sensitivity.csv

Generates:
- results/tables/table1_strategy_comparison.tex
- results/tables/table2_controller_decisions.tex
- results/tables/table3_threshold_sensitivity.tex
"""

import csv
import json
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)


def generate_table1():
    """Generate Table 1: Strategy Comparison (Decoupled metrics, Solvable vs Full Suite)."""
    with open(os.path.join(PROJECT_ROOT, "results", "strategy_comparison_solvable.json")) as f:
        solv = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "results", "strategy_comparison.json")) as f:
        full = json.load(f)

    order = [
        ("Always-FT", "Always Fine-Tuning (FT)"),
        ("Always-ANP", "Always Adv. Neuron Pruning (ANP)"),
        ("Always-NAD", "Always Attention Distillation (NAD)"),
        ("Always-BaEraser-lite", "Always BaEraser-lite"),
        ("Adaptive (Old Policy)", "Adaptive Controller (Pre-Fix Policy)"),
        ("Adaptive Controller", "\\textbf{Adaptive Controller (Proposed)}"),
        ("Oracle", "Oracle Defense Selection"),
    ]

    tex = [
        "% Table 1: Overall Defense Strategy Comparison (Camera-Ready)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Performance comparison of defense strategies across solvable conditions (BadNets + Blended) and the full evaluation suite (including Silent Killer stress-test). Lower ASR, lower $\\Delta$CA, and lower Cost indicate superior defense utility. Failed/unavailable defenses are penalized at ASR=100\\%.}",
        "\\label{tab:strategy_comparison}",
        "\\begin{tabular}{l | cccc | cccc}",
        "\\hline",
        "\\multirow{2}{*}{\\textbf{Defense Strategy}} & \\multicolumn{4}{c|}{\\textbf{Solvable Attacks (6 conditions)}} & \\multicolumn{4}{c}{\\textbf{Full Evaluation Suite (8 conditions)}} \\\\",
        " & \\textbf{Avg ASR} $\\downarrow$ & \\textbf{Avg CA} $\\uparrow$ & \\textbf{Avg Cost} $\\downarrow$ & \\textbf{Fails} ($>10\\%$) & \\textbf{Avg ASR} $\\downarrow$ & \\textbf{Avg CA} $\\uparrow$ & \\textbf{Avg Cost} $\\downarrow$ & \\textbf{Fails} ($>10\\%$) \\\\",
        "\\hline",
    ]

    for key, label in order:
        if key in solv and key in full:
            s = solv[key]
            f_row = full[key]

            s_asr = f"{s['avg_asr_after']:.1f}\\%"
            s_ca = f"{s['avg_ca_after']:.1f}\\%"
            s_cost = f"{s['avg_compute_cost']:.2f}"
            s_fail = f"{s['fails']}/{s['n_evaluated']}"

            f_asr = f"{f_row['avg_asr_after']:.1f}\\%"
            f_ca = f"{f_row['avg_ca_after']:.1f}\\%"
            f_cost = f"{f_row['avg_compute_cost']:.2f}"
            f_fail = f"{f_row['fails']}/{f_row['n_evaluated']}"

            if "Proposed" in label:
                line = f"{label} & \\textbf{{{s_asr}}} & {s_ca} & {s_cost} & \\textbf{{{s_fail}}} & \\textbf{{{f_asr}}} & {f_ca} & {f_cost} & \\textbf{{{f_fail}}} \\\\"
            else:
                line = f"{label} & {s_asr} & {s_ca} & {s_cost} & {s_fail} & {f_asr} & {f_ca} & {f_cost} & {f_fail} \\\\"
            tex.append(line)

    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])

    out_path = os.path.join(TABLES_DIR, "table1_strategy_comparison.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"Generated: {out_path}")


def generate_table2():
    """Generate Table 2: Per-Condition Controller Decisions & Defense Verification."""
    dec_path = os.path.join(PROJECT_ROOT, "results", "controller_decisions.csv")
    matrix_path = os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv")

    df_dec = pd.read_csv(dec_path)
    df_mat = pd.read_csv(matrix_path)

    tex = [
        "% Table 2: Per-Condition Controller Invocations and Post-Defense Outcomes",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Per-condition detector signals, assigned operational quadrants, defense selected by the Adaptive Controller, and resulting post-defense validation metrics.}",
        "\\label{tab:controller_decisions}",
        "\\begin{tabular}{ll | cc | l | l | cccc}",
        "\\hline",
        "\\textbf{Attack} & \\textbf{Rate} & \\textbf{AC Sil.} & \\textbf{STRIP TPR} & \\textbf{Assigned Quadrant} & \\textbf{Selected Defense} & \\textbf{Baseline ASR} & \\textbf{Defended ASR} & \\textbf{Defended CA} & \\textbf{Status} \\\\",
        "\\hline",
    ]

    for _, row in df_dec.iterrows():
        atk = row["attack"]
        cond = row["poison_condition"]
        sil = f"{row['ac_silhouette']:.2f}"
        tpr = f"{row['strip_tpr']:.1f}\\%"
        quad = row["quadrant"]
        chosen = row["defense_chosen"]

        # Lookup metrics
        sub = df_mat[(df_mat["Attack"] == atk) & (df_mat["Poison Condition"] == cond) & (df_mat["Defense"] == chosen)]
        if not sub.empty:
            m = sub.iloc[0]
            asr_bef = f"{m['ASR Before (%)']:.1f}\\%"
            asr_aft = f"{m['ASR After (%)']:.2f}\\%"
            ca_aft = f"{m['CA After (%)']:.1f}\\%"
            status = "\\checkmark Remediated" if m['ASR After (%)'] <= 10.0 else "\\textbf{SK Persistent}"
        else:
            asr_bef, asr_aft, ca_aft, status = "-", "-", "-", "Failed"

        line = f"{atk} & {cond} & {sil} & {tpr} & {quad} & \\textbf{{{chosen}}} & {asr_bef} & {asr_aft} & {ca_aft} & {status} \\\\"
        tex.append(line)

    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])

    out_path = os.path.join(TABLES_DIR, "table2_controller_decisions.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"Generated: {out_path}")


def generate_table3():
    """Generate Table 3: Threshold Sensitivity Sweep."""
    sens_path = os.path.join(PROJECT_ROOT, "results", "threshold_sensitivity.csv")
    df = pd.read_csv(sens_path)

    # Pick representative rows: all tau_ac with -10%, 0%, +10% tau_strip
    selected_deltas = [
        (-0.10, -10.0), (-0.10, 0.0), (-0.10, 10.0),
        (-0.05, -10.0), (-0.05, 0.0), (-0.05, 10.0),
        (0.00, -10.0), (0.00, 0.0), (0.00, 10.0),
        (0.05, -10.0), (0.05, 0.0), (0.05, 10.0),
        (0.10, -10.0), (0.10, 0.0), (0.10, 10.0),
    ]

    tex = [
        "% Table 3: Threshold Sensitivity and Controller Stability Analysis",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Threshold perturbation analysis around the baseline calibration ($\\tau_{ac} = 0.40, \\tau_{strip} = 30\\%$). The controller's post-defense security on solvable conditions remains stable across the entire sweep window.}",
        "\\label{tab:threshold_sensitivity}",
        "\\begin{tabular}{cccc | ccc}",
        "\\hline",
        "$\\Delta \\tau_{ac}$ & $\\Delta \\tau_{strip}$ & $\\tau_{ac}$ & $\\tau_{strip}$ & \\textbf{Decisions Changed} & \\textbf{Solvable ASR} $\\downarrow$ & \\textbf{Full Suite ASR} $\\downarrow$ \\\\",
        "\\hline",
    ]

    for d_ac, d_strip in selected_deltas:
        sub = df[(df["delta_tau_ac"] == f"{d_ac:+.2f}") & (df["delta_tau_strip"] == f"{d_strip:+.0f}%")]
        if not sub.empty:
            r = sub.iloc[0]
            is_base = (d_ac == 0.0 and d_strip == 0.0)
            n_chg = f"{r['n_changed']}/8"
            solv_asr = f"{r['avg_asr_solvable']:.2f}\\%"
            full_asr = f"{r['avg_asr_all']:.2f}\\%"

            if is_base:
                line = f"\\textbf{{{d_ac:+.2f}}} & \\textbf{{{d_strip:+.0f}\\%}} & \\textbf{{{r['tau_ac']:.2f}}} & \\textbf{{{r['tau_strip']:.0f}\\%}} & \\textbf{{{n_chg}}} & \\textbf{{{solv_asr}}} & \\textbf{{{full_asr}}} \\\\ % Baseline"
            else:
                line = f"{d_ac:+.2f} & {d_strip:+.0f}\\% & {r['tau_ac']:.2f} & {r['tau_strip']:.0f}\\% & {n_chg} & {solv_asr} & {full_asr} \\\\"
            tex.append(line)

    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    out_path = os.path.join(TABLES_DIR, "table3_threshold_sensitivity.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"Generated: {out_path}")


def main():
    print("=" * 70)
    print("EXPORTING CAMERA-READY LATEX TABLES")
    print("=" * 70)
    generate_table1()
    generate_table2()
    generate_table3()
    print("=" * 70)
    print(f"All LaTeX tables written to: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
