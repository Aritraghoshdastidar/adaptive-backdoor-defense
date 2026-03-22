# core/controller.py
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_poison_score(silhouette, suspicious_fraction,
                          confidence_gap=0.0):
    """
    Combine AC signals into a single severity score [0, 1].
    confidence_gap is optional — defaults to 0 if not computed.
    """
    sil_norm  = sigmoid((silhouette - 0.1) / 0.2)
    frac_norm = sigmoid((suspicious_fraction - 0.02) / 0.05)
    conf_norm = sigmoid((confidence_gap - 0.1) / 0.15)

    score = 0.5 * sil_norm + 0.3 * frac_norm + 0.2 * conf_norm
    return round(float(score), 4)


def decide_defense(poison_score):
    """
    Rule-based controller.
    Returns method name + reason string.
    """
    if poison_score >= 0.6:
        method = "prune_finetune"
        reason = f"High severity (score={poison_score})"
    elif poison_score >= 0.3:
        method = "finetune"
        reason = f"Medium severity (score={poison_score})"
    else:
        method = "finetune_light"
        reason = f"Low severity (score={poison_score})"

    return {
        "method":       method,
        "poison_score": poison_score,
        "reason":       reason
    }


def log_decision(attack, poison_rate, silhouette,
                  suspicious_fraction, PDR,
                  poison_score, decision,
                  csv_path="results/csv/ac_results.csv"):
    """
    Append one row to the shared AC results CSV.
    """
    import csv, os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {
        "attack":               attack,
        "poison_rate":          poison_rate,
        "seed":                 2025,
        "silhouette":           silhouette,
        "suspicious_fraction":  suspicious_fraction,
        "PDR":                  PDR,
        "poison_score":         poison_score,
        "defense_chosen":       decision["method"],
        "reason":               decision["reason"],
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Logged to CSV:", row)
    return row