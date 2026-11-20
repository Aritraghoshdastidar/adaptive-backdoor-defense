"""
scripts/export_defense_matrix.py — Export results.xlsx → defense_matrix.csv

Reads the defense results from results.xlsx (rows 18–65) and writes them
into the CSV format that evaluate_controller.py expects.

Handles:
- Only the 4 core defenses (FT, ANP, NAD, BaEraser-lite)
- Marks cells with missing CA/ASR as excluded
- Adds compute cost from notebook timing data
- Excludes invalid cells (Blended 10% NAD: collapse gate failed)

Run:
    python scripts/export_defense_matrix.py
"""

import csv
import os
import sys

try:
    import openpyxl
except ImportError:
    print("Need openpyxl: pip install openpyxl")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH = os.path.join(PROJECT_ROOT, "results", "results.xlsx")
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv")

# Map xlsx defense names to controller defense names
DEFENSE_NAME_MAP = {
    "FT": "FT",
    "ANP": "ANP",
    "NAD": "NAD",
    "BaEraser-lite": "BaEraser-lite",
}

# Compute cost in seconds from notebook Cell I1 outputs
# BadNets costs from defence_badnets.ipynb, Blended from blended notebooks, SK from sk_defense_v4
COST_DEFAULTS = {
    # BadNets (from defence_badnets notebook)
    ("BadNets", "FT"): 18.0,
    ("BadNets", "ANP"): 750.0,
    ("BadNets", "NAD"): 256.0,
    ("BadNets", "BaEraser-lite"): 216.0,
    # Blended (from blended_defense notebooks)
    ("Blended", "FT"): 18.0,
    ("Blended", "ANP"): 750.0,
    ("Blended", "NAD"): 256.0,
    ("Blended", "BaEraser-lite"): 216.0,
    # SK (from sk_defense_v4 Cell I1)
    ("Silent Killer", "FT"): 18.0,
    ("Silent Killer", "ANP"): 750.0,
    ("Silent Killer", "NAD"): 256.0,
    ("Silent Killer", "BaEraser-lite"): 216.0,
}

# Only include target poison conditions (SK 0.5% and 10% not in core 8-condition suite)
TARGET_CONDITIONS = {
    ("BadNets", "1%"), ("BadNets", "5%"), ("BadNets", "10%"),
    ("Blended", "1%"), ("Blended", "5%"), ("Blended", "10%"),
    ("Silent Killer", "1%"), ("Silent Killer", "5%"),
}
EXCLUDE_CELLS = {
    # Blended 10% NAD: collapse gate failed (CA drop 5.19pp > 5pp budget)
    ("Blended", "10%", "NAD"): "INVALID: collapse gate failed",
    # SK BaEraser pr01, pr05: no valid checkpoint produced
    ("Silent Killer", "1%", "BaEraser-lite"): "FAILED: no valid checkpoint",
    ("Silent Killer", "5%", "BaEraser-lite"): "FAILED: no valid checkpoint",
}

# Cells to FLAG (usable with caveats)
FLAG_CELLS = {
    ("Silent Killer", "1%", "NAD"): "teacher_backdoored",
    ("Silent Killer", "5%", "NAD"): "teacher_backdoored",
    ("Silent Killer", "10%", "NAD"): "teacher_backdoored",
    ("Blended", "1%", "NAD"): "teacher_backdoored_pr01",
    ("Blended", "10%", "BaEraser-lite"): "selection_failure",
}


def main():
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    ws = wb.active

    rows_out = []
    skipped = []
    excluded = []

    # Defense rows start at row 18, header at row 17
    for r in range(18, ws.max_row + 1):
        attack = ws.cell(r, 1).value
        poison_rate = ws.cell(r, 2).value
        ca_before = ws.cell(r, 3).value
        asr_before = ws.cell(r, 4).value
        defense_raw = ws.cell(r, 5).value
        ca_after = ws.cell(r, 6).value
        asr_after = ws.cell(r, 7).value

        # Stop at empty rows
        if attack is None and defense_raw is None:
            break

        # Skip non-core defenses (FT matched, ANP+FT are supplementary)
        if defense_raw not in DEFENSE_NAME_MAP:
            skipped.append(f"Row {r}: {attack} {poison_rate} {defense_raw} (not core)")
            continue

        defense = DEFENSE_NAME_MAP[defense_raw]

        # Format poison condition
        if poison_rate is not None:
            pr_float = float(poison_rate)
            if pr_float < 0.01:
                poison_cond = f"{pr_float*100:.1f}%"
            else:
                poison_cond = f"{int(pr_float*100)}%"
        else:
            continue

        # Check target conditions
        if (attack, poison_cond) not in TARGET_CONDITIONS:
            skipped.append(f"{attack} {poison_cond} (not in 8 target conditions)")
            continue

        # Check exclusions
        key = (attack, poison_cond, defense)
        if key in EXCLUDE_CELLS:
            excluded.append(f"{attack} {poison_cond} {defense}: {EXCLUDE_CELLS[key]}")
            continue

        # Skip if missing data
        if ca_after is None or asr_after is None:
            excluded.append(f"{attack} {poison_cond} {defense}: missing CA/ASR after")
            continue

        # Get cost
        cost_key = (attack, defense)
        cost_sec = COST_DEFAULTS.get(cost_key, 0)

        # Flag
        flag = FLAG_CELLS.get(key, "")

        rows_out.append({
            "Attack": attack,
            "Poison Condition": poison_cond,
            "Defense": defense,
            "CA Before (%)": float(ca_before) if ca_before else "",
            "ASR Before (%)": float(asr_before) if asr_before else "",
            "CA After (%)": float(ca_after),
            "ASR After (%)": float(asr_after),
            "Compute Cost (sec)": cost_sec,
            "Flag": flag,
        })

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fieldnames = [
        "Attack", "Poison Condition", "Defense",
        "CA Before (%)", "ASR Before (%)",
        "CA After (%)", "ASR After (%)",
        "Compute Cost (sec)", "Flag",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    # Summary
    print("=" * 70)
    print("DEFENSE MATRIX EXPORT")
    print("=" * 70)
    print(f"Source:  {XLSX_PATH}")
    print(f"Output:  {CSV_PATH}")
    print(f"Rows written: {len(rows_out)}")
    print()

    # Coverage matrix
    attacks_conds = [
        ("BadNets", "1%"), ("BadNets", "5%"), ("BadNets", "10%"),
        ("Blended", "1%"), ("Blended", "5%"), ("Blended", "10%"),
        ("Silent Killer", "1%"), ("Silent Killer", "5%"),
    ]
    defenses = ["FT", "ANP", "NAD", "BaEraser-lite"]
    filled = {(r["Attack"], r["Poison Condition"], r["Defense"]) for r in rows_out}

    print(f"{'':22s} {'FT':>8s} {'ANP':>8s} {'NAD':>8s} {'BaEra':>8s}")
    print("-" * 60)
    for attack, cond in attacks_conds:
        line = f"{attack:15s} {cond:5s}"
        for d in defenses:
            if (attack, cond, d) in filled:
                # Find ASR
                asr = [r["ASR After (%)"] for r in rows_out
                       if r["Attack"] == attack and r["Poison Condition"] == cond and r["Defense"] == d][0]
                line += f" {asr:>7.1f}%"
            else:
                line += f" {'---':>8s}"
        print(line)

    n_total = len(attacks_conds) * len(defenses)
    n_filled = sum(1 for a, c in attacks_conds for d in defenses if (a, c, d) in filled)
    print(f"\nCoverage: {n_filled}/{n_total} cells")

    if excluded:
        print(f"\nExcluded ({len(excluded)}):")
        for e in excluded:
            print(f"  ✗ {e}")

    if skipped:
        print(f"\nSkipped (non-core defenses, {len(skipped)}):")
        for s in skipped[:5]:
            print(f"  → {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")


if __name__ == "__main__":
    main()
