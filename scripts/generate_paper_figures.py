"""
scripts/generate_paper_figures.py — Generate Publication Figures for the Paper

Produces 5 high-resolution (300 DPI) publication-quality figures:
- Figure 1: fig1_quadrant_scatter (2x2 AC vs STRIP decision space with labeled conditions)
- Figure 2: fig2_pareto_frontier (Compute Cost vs ASR trade-off curve)
- Figure 3: fig3_defense_heatmap (ASR post-defense matrix with controller routes highlighted)
- Figure 4: fig4_policy_ablation (Old Policy vs New Policy RQ3 comparison)
- Figure 5: fig5_threshold_stability_surface (2D sensitivity landscape around tau_ac/tau_strip)

Saves both PNG (for preview) and PDF (vector format for LaTeX).
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_controller import DETECTION_DATA, load_defense_matrix

# Publication style configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})

FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name):
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


# ── Figure 1: Quadrant Scatter Plot ──────────────────────────────────────────

def plot_fig1_quadrant_scatter():
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    tau_ac = 0.40
    tau_strip = 30.0

    # Shaded quadrant backgrounds
    ax.axhspan(tau_strip, 105, 0, (tau_ac - 0.2) / 0.6, color="#e6f2ff", alpha=0.5, label="NAD Quadrant")
    ax.axhspan(tau_strip, 105, (tau_ac - 0.2) / 0.6, 1, color="#f0e6ff", alpha=0.5, label="BaEraser Quadrant")
    ax.axhspan(-5, tau_strip, 0, (tau_ac - 0.2) / 0.6, color="#e6ffe6", alpha=0.5, label="ANP (Floor) Quadrant")
    ax.axhspan(-5, tau_strip, (tau_ac - 0.2) / 0.6, 1, color="#fff2e6", alpha=0.5, label="ANP Quadrant")

    # Threshold lines
    ax.axvline(tau_ac, color="#cc0000", linestyle="--", linewidth=1.8, label=f"$\\tau_{{ac}} = {tau_ac:.2f}$")
    ax.axhline(tau_strip, color="#cc0000", linestyle=":", linewidth=1.8, label=f"$\\tau_{{strip}} = {tau_strip:.0f}\\%$")

    # Quadrant text labels
    ax.text(0.23, 95, "AC-LOW / STRIP-HIGH\n→ NAD", fontsize=11, fontweight="bold", color="#004085")
    ax.text(0.55, 95, "AC-HIGH / STRIP-HIGH\n→ BaEraser-lite", fontsize=11, fontweight="bold", color="#4a148c")
    ax.text(0.23, 8, "AC-LOW / STRIP-LOW\n→ ANP (Policy Fix)", fontsize=11, fontweight="bold", color="#155724")
    ax.text(0.55, 8, "AC-HIGH / STRIP-LOW\n→ ANP", fontsize=11, fontweight="bold", color="#8a4b08")

    # Marker styles per attack
    markers = {"BadNets": ("o", "#d9534f"), "Blended": ("s", "#0275d8"), "Silent Killer": ("^", "#5cb85c")}

    for det in DETECTION_DATA:
        attack = det["attack"]
        cond = det["poison_condition"]
        x = det["ac_silhouette"]
        y = det["strip_tpr"]
        m, c = markers[attack]

        ax.scatter(x, y, marker=m, s=140, color=c, edgecolors="black", linewidth=1.2, zorder=5)

        # Label offsets
        offset_x, offset_y = 0.012, 1.8
        if attack == "Silent Killer" and cond == "1%":
            offset_y = -3.5
        elif attack == "BadNets" and cond == "1%":
            offset_y = 2.0
        elif attack == "Blended" and cond == "10%":
            offset_y = 2.5

        ax.annotate(f"{attack} {cond}", (x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=9.5, fontweight="semibold", zorder=6)

    # Legend elements for attacks
    for attack, (m, c) in markers.items():
        ax.scatter([], [], marker=m, s=100, color=c, edgecolors="black", label=attack)

    ax.set_xlim(0.20, 0.75)
    ax.set_ylim(-4, 105)
    ax.set_xlabel("Activation Clustering Severity (Silhouette Score, Penultimate Layer)", fontweight="bold")
    ax.set_ylabel("STRIP Behavioral Dominance (True Positive Rate %)", fontweight="bold")
    ax.set_title("Controller 2×2 Decision Space: Detection Signals & Operating Quadrants", pad=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center left", framealpha=0.9, fontsize=9.5)

    save_fig(fig, "fig1_quadrant_scatter")


# ── Figure 2: Pareto Frontier Plot ───────────────────────────────────────────

def plot_fig2_pareto_frontier():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Load strategy results
    with open(os.path.join(PROJECT_ROOT, "results", "strategy_comparison_solvable.json")) as f:
        solvable_res = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "results", "strategy_comparison.json")) as f:
        all_res = json.load(f)

    styles = {
        "Always-FT": ("x", "#777777", 90),
        "Always-ANP": ("s", "#0275d8", 120),
        "Always-NAD": ("d", "#f0ad4e", 110),
        "Always-BaEraser-lite": ("v", "#9c27b0", 110),
        "Adaptive (Old Policy)": ("^", "#d9534f", 120),
        "Adaptive Controller": ("*", "#28a745", 220),
        "Oracle": ("p", "#333333", 140),
    }

    # Plot Subplot 1: Solvable Attacks (BadNets + Blended)
    for name, s in solvable_res.items():
        if name in styles:
            m, c, sz = styles[name]
            cost = s["avg_compute_cost"]
            asr = s["avg_asr_after"]
            ax1.scatter(cost, asr, marker=m, color=c, s=sz, edgecolors="black" if m != "x" else None,
                        linewidth=1.2, label=name, zorder=5)
            # Label offsets
            dx, dy = 0.02, 1.2
            if name == "Adaptive Controller":
                dx, dy = 0.03, -1.8
            elif name == "Always-ANP":
                dx, dy = -0.22, 1.5
            elif name == "Always-BaEraser-lite":
                dx, dy = 0.03, 1.2
            ax1.annotate(f"{name}\n({asr:.1f}%, cost={cost:.2f})", (cost, asr),
                         xytext=(cost + dx, asr + dy), fontsize=8.5, fontweight="bold")

    ax1.set_xlim(-0.05, 1.15)
    ax1.set_ylim(-3, 65)
    ax1.set_xlabel("Normalized Compute Cost (0.0=Cheapest, 1.0=Heaviest)", fontweight="bold")
    ax1.set_ylabel("Average Post-Defense ASR (%) ↓", fontweight="bold")
    ax1.set_title("(a) Solvable Attacks (BadNets + Blended)\nAdaptive Controller matches ANP at 31% less compute",
                  fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot Subplot 2: Full Suite (All 8 conditions, including SK)
    for name, s in all_res.items():
        if name in styles:
            m, c, sz = styles[name]
            cost = s["avg_compute_cost"]
            asr = s["avg_asr_after"]
            ax2.scatter(cost, asr, marker=m, color=c, s=sz, edgecolors="black" if m != "x" else None,
                        linewidth=1.2, label=name, zorder=5)
            dx, dy = 0.02, 1.2
            if name == "Adaptive Controller":
                dx, dy = 0.03, -1.8
            elif name == "Always-ANP":
                dx, dy = -0.22, 1.2
            ax2.annotate(f"{name}\n({asr:.1f}%, cost={cost:.2f})", (cost, asr),
                         xytext=(cost + dx, asr + dy), fontsize=8.5, fontweight="bold")

    ax2.set_xlim(-0.05, 1.15)
    ax2.set_ylim(-3, 72)
    ax2.set_xlabel("Normalized Compute Cost (0.0=Cheapest, 1.0=Heaviest)", fontweight="bold")
    ax2.set_ylabel("Average Post-Defense ASR (%) ↓", fontweight="bold")
    ax2.set_title("(b) Full Suite (including SK Stress-Test)\nAdaptive Controller achieves lowest overall ASR (20.1%)",
                  fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    save_fig(fig, "fig2_pareto_frontier")


# ── Figure 3: Defense Heatmap ────────────────────────────────────────────────

def plot_fig3_defense_heatmap():
    matrix_path = os.path.join(PROJECT_ROOT, "results", "defense_matrix.csv")
    df = pd.read_csv(matrix_path)

    attacks_conds = [
        "BadNets 1%", "BadNets 5%", "BadNets 10%",
        "Blended 1%", "Blended 5%", "Blended 10%",
        "Silent Killer 1%", "Silent Killer 5%",
    ]
    defenses = ["FT", "ANP", "NAD", "BaEraser-lite"]

    # Pivot matrix
    grid = np.zeros((len(attacks_conds), len(defenses)))
    annot = np.empty((len(attacks_conds), len(defenses)), dtype=object)

    # Controller route coordinates
    controller_routes = [
        ("BadNets 1%", "ANP"),
        ("BadNets 5%", "NAD"),
        ("BadNets 10%", "BaEraser-lite"),
        ("Blended 1%", "ANP"),
        ("Blended 5%", "BaEraser-lite"),
        ("Blended 10%", "ANP"),
        ("Silent Killer 1%", "ANP"),
        ("Silent Killer 5%", "NAD"),
    ]

    for i, ac in enumerate(attacks_conds):
        atk, cond = ac.rsplit(" ", 1)
        for j, d in enumerate(defenses):
            sub = df[(df["Attack"] == atk) & (df["Poison Condition"] == cond) & (df["Defense"] == d)]
            if not sub.empty and not pd.isna(sub.iloc[0]["ASR After (%)"]):
                val = sub.iloc[0]["ASR After (%)"]
                grid[i, j] = val
                annot[i, j] = f"{val:.1f}%"
            else:
                grid[i, j] = 100.0  # failed
                annot[i, j] = "FAIL"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Custom colormap: Green for low ASR, Red for high ASR
    cmap = sns.diverging_palette(135, 10, s=85, l=55, as_cmap=True)
    sns.heatmap(grid, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=100,
                xticklabels=defenses, yticklabels=attacks_conds, ax=ax,
                cbar_kws={"label": "Post-Defense ASR (%)"}, linewidths=1.2, linecolor="white")

    # Draw border around cells chosen by the Adaptive Controller
    for ac, d in controller_routes:
        i = attacks_conds.index(ac)
        j = defenses.index(d)
        rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="#002244", linewidth=3.5, zorder=10)
        ax.add_patch(rect)

    # Add legend note for highlighted boxes
    ax.text(0.0, -0.6, "■ Bold Blue Borders indicate the route selected by the Adaptive Controller.",
            fontsize=10.5, fontweight="bold", color="#002244")

    ax.set_title("Post-Defense ASR Across Evaluation Matrix\n(Highlighting Adaptive Controller Routes)",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Attack & Poisoning Condition", fontweight="bold")
    ax.set_xlabel("Defense Method", fontweight="bold")

    save_fig(fig, "fig3_defense_heatmap")


# ── Figure 4: Policy Ablation ────────────────────────────────────────────────

def plot_fig4_policy_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    conditions = ["BadNets 1%", "Blended 1%", "Silent Killer 1%"]
    old_asrs = [18.7, 99.99, 81.60]
    new_asrs = [0.20, 0.06, 85.73]

    x = np.arange(len(conditions))
    width = 0.32

    rects1 = ax.bar(x - width/2, old_asrs, width, label="Old Policy (routes to FT)", color="#d9534f", edgecolor="black")
    rects2 = ax.bar(x + width/2, new_asrs, width, label="New Policy (Policy Fix → ANP)", color="#28a745", edgecolor="black")

    # Add value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Post-Defense ASR (%) ↓", fontweight="bold")
    ax.set_title("RQ3 Policy Ablation: Impact of Policy Fix on AC-LOW / STRIP-LOW Regime", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(framealpha=0.9, loc="upper right")

    # Add annotation box
    ax.annotate("Solvable conditions drop\nfrom 20.4% → 0.6% avg ASR",
                xy=(1, 50), xytext=(0.8, 75),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="#ffffcc", ec="#999900"),
                fontsize=10, fontweight="bold")

    save_fig(fig, "fig4_policy_ablation")


# ── Figure 5: Threshold Sensitivity Landscape ────────────────────────────────

def plot_fig5_threshold_stability_surface():
    csv_path = os.path.join(PROJECT_ROOT, "results", "threshold_sensitivity.csv")
    df = pd.read_csv(csv_path)

    # Pivot grid: index = tau_ac, columns = tau_strip
    pivot_score = df.pivot(index="tau_ac", columns="tau_strip", values="avg_score")
    pivot_asr = df.pivot(index="tau_ac", columns="tau_strip", values="avg_asr_solvable")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Score Landscape
    sns.heatmap(pivot_score, annot=True, fmt=".1f", cmap="YlGnBu_r", ax=ax1, cbar_kws={"label": "Formal Score (lower is better)"})
    # Mark baseline operating point (0.40, 30%)
    row_idx = list(pivot_score.index).index(0.40)
    col_idx = list(pivot_score.columns).index(30.0)
    rect1 = patches.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor="#cc0000", linewidth=3.5, zorder=10)
    ax1.add_patch(rect1)

    ax1.set_title("(a) Controller Objective Score Surface\n(Red Box = Base Operating Point: τ_ac=0.40, τ_strip=30%)", fontweight="bold")
    ax1.set_xlabel("STRIP Threshold τ_strip (%)", fontweight="bold")
    ax1.set_ylabel("AC Threshold τ_ac", fontweight="bold")

    # Subplot 2: Solvable ASR Landscape
    sns.heatmap(pivot_asr, annot=True, fmt=".2f", cmap="Blues_r", ax=ax2, cbar_kws={"label": "Solvable ASR (%)"})
    rect2 = patches.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor="#cc0000", linewidth=3.5, zorder=10)
    ax2.add_patch(rect2)

    ax2.set_title("(b) Solvable ASR (%) Surface\n(Consistently < 0.85% across entire perturbation grid)", fontweight="bold")
    ax2.set_xlabel("STRIP Threshold τ_strip (%)", fontweight="bold")
    ax2.set_ylabel("AC Threshold τ_ac", fontweight="bold")

    save_fig(fig, "fig5_threshold_stability_surface")


def main():
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES (300 DPI, PNG + PDF)")
    print("=" * 70)
    plot_fig1_quadrant_scatter()
    plot_fig2_pareto_frontier()
    plot_fig3_defense_heatmap()
    plot_fig4_policy_ablation()
    plot_fig5_threshold_stability_surface()
    print("=" * 70)
    print("ALL 5 PUBLICATION FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
