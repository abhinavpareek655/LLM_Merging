"""
Generates complete Experiment 2 figures from sweep data:
1. Figure 2: Normalized Retention Curves (PRR vs. Alpha & Pareto Frontier)
2. Figure 3: Raw Metric Curves vs. Alpha:
   - Panel (a): SVAMP Test Accuracy (%) vs. Alpha with B0 Baseline (53.33%) and E_math (58.67%)
   - Panel (b): DPO Reward Margin vs. Alpha with B0 Baseline (0.0000) and E_pref (+0.3149)
Copies all generated figures directly into the artifact directory for inline UI preview.
"""
import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT_DIR = r"C:\Users\vedan\.gemini\antigravity\brain\c4af8d62-e4c1-481f-bd8a-d208a5ae2a79"

# Benchmark Baseline & Expert Anchors
B0_SVAMP = 53.3333
EMATH_SVAMP = 58.6667
B0_DPO = 0.0000
EPREF_DPO = 0.31489

def plot_all_figures(results_file="outputs/results/e2_sweep_results.json"):
    if not os.path.exists(results_file):
        results_file = "outputs/results/e2_sweep_progress.json"
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    methods = ["linear", "slerp", "ties", "dare_ties"]
    method_labels = {
        "linear": "Linear Task Arithmetic",
        "slerp": "SLERP",
        "ties": "TIES-Merging",
        "dare_ties": "DARE-TIES"
    }
    colors = {
        "linear": "#1f77b4",
        "slerp": "#ff7f0e",
        "ties": "#2ca02c",
        "dare_ties": "#d62728"
    }
    markers = {
        "linear": "o",
        "slerp": "^",
        "ties": "s",
        "dare_ties": "D"
    }

    alphas = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
    plt.rcParams["font.family"] = "sans-serif"

    # ==========================================
    # 1. FIGURE 2: CR_math vs Alpha & PRR vs Alpha
    # ==========================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: CR_math vs Alpha
    for m in methods:
        pts = []
        for a in alphas:
            k = f"{m}_{a:.1f}"
            if k in data:
                pts.append((a, data[k]["cr_math"]))
        if pts:
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, marker=markers[m], linewidth=2.2, markersize=6, label=method_labels[m], color=colors[m])

    # Reference lines for CR_math
    ax1.axhline(100.0, color="green", linestyle=":", linewidth=1.8, label=r"Specialist $E_{\text{math}}$ (100%)")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.5, label=r"Base Model $B_0$ (0%)")

    ax1.set_title(r"(a) Capability Retention ($\text{CR}_{\text{math}}$) vs. $\alpha$", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Merge Weight $\alpha$ (Preference Anchor)", fontsize=11, fontweight="medium")
    ax1.set_ylabel(r"Capability Retention $\text{CR}_{\text{math}}$ (%)", fontsize=11, fontweight="medium")
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.set_xticks(alphas)
    ax1.legend(frameon=True, fontsize=9.5, loc="best")

    # Panel B: PRR vs Alpha
    for m in methods:
        pts = []
        for a in alphas:
            k = f"{m}_{a:.1f}"
            if k in data:
                pts.append((a, data[k]["prr"]))
        if pts:
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, marker=markers[m], linewidth=2.2, markersize=6, label=method_labels[m], color=colors[m])

    # Reference lines for PRR
    ax2.axhline(100.0, color="purple", linestyle=":", linewidth=1.8, label=r"Specialist $E_{\text{pref}}$ (100%)")
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.5, label=r"Base Model $B_0$ (0%)")

    ax2.set_title(r"(b) Preference Retention Ratio (PRR) vs. $\alpha$", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel(r"Merge Weight $\alpha$ (Preference Anchor)", fontsize=11, fontweight="medium")
    ax2.set_ylabel(r"Preference Retention Ratio PRR (%)", fontsize=11, fontweight="medium")
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.set_xticks(alphas)
    ax2.legend(frameon=True, fontsize=9.5, loc="best")

    os.makedirs("outputs/results", exist_ok=True)
    fig2_path = "outputs/results/figure2_retention_curves.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    # ==========================================
    # 2. FIGURE 3: Raw SVAMP & DPO vs. Alpha with Baselines
    # ==========================================
    fig3, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: SVAMP Accuracy vs Alpha
    for m in methods:
        pts = []
        for a in alphas:
            k = f"{m}_{a:.1f}"
            if k in data:
                pts.append((a, data[k]["svamp_acc"]))
        if pts:
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            bx1.plot(xs, ys, marker=markers[m], linewidth=2.2, markersize=6, label=method_labels[m], color=colors[m])

    # Add Baselines for SVAMP
    bx1.axhline(B0_SVAMP, color="black", linestyle="--", linewidth=1.8, label=f"Base Model $B_0$ ({B0_SVAMP:.2f}%)")
    bx1.axhline(EMATH_SVAMP, color="purple", linestyle=":", linewidth=1.8, label=f"Math Specialist $E_{{\\text{{math}}}}$ ({EMATH_SVAMP:.2f}%)")

    bx1.set_title("(a) SVAMP Test Accuracy (%) vs. $\\alpha$", fontsize=13, fontweight="bold", pad=10)
    bx1.set_xlabel("Merge Weight $\\alpha$ (Preference Anchor)", fontsize=11, fontweight="medium")
    bx1.set_ylabel("Exact Match Accuracy (%)", fontsize=11, fontweight="medium")
    bx1.grid(True, linestyle="--", alpha=0.55)
    bx1.set_xticks(alphas)
    bx1.legend(frameon=True, fontsize=9.5, loc="best")

    # Panel B: DPO Margin vs Alpha
    for m in methods:
        pts = []
        for a in alphas:
            k = f"{m}_{a:.1f}"
            if k in data:
                pts.append((a, data[k]["dpo_margin"]))
        if pts:
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            bx2.plot(xs, ys, marker=markers[m], linewidth=2.2, markersize=6, label=method_labels[m], color=colors[m])

    # Add Baselines for DPO Margin
    bx2.axhline(B0_DPO, color="black", linestyle="--", linewidth=1.8, label=f"Base Model $B_0$ ({B0_DPO:.4f})")
    bx2.axhline(EPREF_DPO, color="purple", linestyle=":", linewidth=1.8, label=f"Preference Specialist $E_{{\\text{{pref}}}}$ (+{EPREF_DPO:.4f})")

    bx2.set_title("(b) DPO Implicit Reward Margin vs. $\\alpha$", fontsize=13, fontweight="bold", pad=10)
    bx2.set_xlabel("Merge Weight $\\alpha$ (Preference Anchor)", fontsize=11, fontweight="medium")
    bx2.set_ylabel("DPO Reward Margin ($\\beta=0.1$)", fontsize=11, fontweight="medium")
    bx2.grid(True, linestyle="--", alpha=0.55)
    bx2.set_xticks(alphas)
    bx2.legend(frameon=True, fontsize=9.5, loc="best")

    plt.tight_layout()
    fig3_path = "outputs/results/figure3_raw_metrics_vs_alpha.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()

    # Copy both plots to artifact directory if on local system
    if os.path.exists(ARTIFACT_DIR):
        try:
            shutil.copy(fig2_path, os.path.join(ARTIFACT_DIR, "figure2_retention_curves.png"))
            shutil.copy(fig3_path, os.path.join(ARTIFACT_DIR, "figure3_raw_metrics_vs_alpha.png"))
        except Exception:
            pass
    print(f"Generated Figure 2 & Figure 3 in outputs/results/ successfully!")

if __name__ == "__main__":
    plot_all_figures()
