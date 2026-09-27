#!/usr/bin/env python3
# ============================================================
#  MicroGen Decision Engine — Paper Visualizations
# ============================================================
#
# Generates publication-ready SVGs/PNGs for the three core
# claims of the MicroGen architecture.
#
# Usage:
# 1. Update the arrays below with your final Kaggle GPU numbers.
# 2. Run: pip install matplotlib seaborn
# 3. Run: python3 benchmarks/generate_paper_plots.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── 0. Configuration ──────────────────────────────────────────
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Real Kaggle T4 GPU data (Qwen/Qwen2.5-1.5B-Instruct) ──────
# Task 7.1 measurements
CANDIDATES     = np.array([2, 4, 6, 10, 20, 50, 77, 100])
P50_BATCHED    = np.array([267, 266, 267, 318, 320, 322, 320, 375])   # ms
P95_BATCHED    = np.array([273, 271, 279, 336, 348, 342, 328, 394])   # ms
P50_SEQUENTIAL = CANDIDATES * 140   # approximated from pre-batched DFS timing (~140ms/candidate)

# Task 7.2 measurements (Banking77, 77 classes)
CALIB_SIZES    = np.array([50, 100, 250, 500, 1000, 2000])
ECE_UNCAL      = 28.26   # %
ECE_CALIBRATED = np.array([33.43, 33.69, 33.27, 33.00, 33.06, 33.01])  # %
TEMPERATURES   = np.array([1.267, 1.286, 1.255, 1.237, 1.241, 1.238])

# Permutation invariance data (from Phase 6 benchmark, N=1000)
DATASETS        = ['SST-2\n(2 classes)', 'AG News\n(4 classes)', 'Emotion\n(6 classes)', 'Banking77\n(77 classes)']
GEN_CHANGED     = [0, 0, 60, 50]   # % predictions flipped on reorder
MICROGEN_CHANGED = [0, 0, 0, 0]

# Set global style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'sans-serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
})

def save_plot(fig, name):
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{name}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved: {png_path} & {svg_path}")

# ============================================================
#  PLOT 1: Latency Scaling (Breaking the Bottleneck)
# ============================================================
def plot_latency_scaling():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(CANDIDATES, P50_SEQUENTIAL, marker='o', linestyle='--', color='#e74c3c',
            linewidth=2.5, label='Sequential DFS Trie (projected)')
    ax.fill_between(CANDIDATES, P50_BATCHED, P95_BATCHED, alpha=0.2, color='#2ecc71')
    ax.plot(CANDIDATES, P50_BATCHED, marker='s', linestyle='-', color='#27ae60',
            linewidth=3.5, label='Batched Trie Frontier — p50 (MicroGen)')
    ax.plot(CANDIDATES, P95_BATCHED, marker='', linestyle=':', color='#27ae60',
            linewidth=1.8, label='Batched Trie Frontier — p95 (MicroGen)')

    # Annotate the key depth transitions
    ax.axvline(x=6.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(7, max(P50_SEQUENTIAL) * 0.6, 'depth\n5→6', ha='left', fontsize=10, color='gray')
    ax.axvline(x=77.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(78, max(P50_SEQUENTIAL) * 0.6, 'depth\n6→7', ha='left', fontsize=10, color='gray')

    ax.set_title('Decision Latency Scaling\nBatched Trie Frontier vs. Sequential DFS')
    ax.set_xlabel('Number of Candidate Classes')
    ax.set_ylabel('p50 / p95 Latency (ms)')
    ax.set_ylim(0, max(P50_SEQUENTIAL) * 1.1)
    ax.legend(loc='upper left', frameon=True, shadow=True)

    save_plot(fig, "1_latency_scaling")


# ============================================================
#  PLOT 2: Permutation Invariance (Prompt-Order Bias)
# ============================================================
def plot_permutation_invariance():
    x = np.arange(len(DATASETS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    rects1 = ax.bar(x - width/2, GEN_CHANGED,      width, label='Free-Form Generation Baseline',
                    color='#e74c3c', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, MICROGEN_CHANGED, width, label='MicroGen (Candidate Scoring)',
                    color='#3498db', edgecolor='black', linewidth=0.8)

    ax.set_title('Robustness to Candidate Reordering\n(Engine permutation invariance test, 10 permutations, N=1000)')
    ax.set_ylabel('Predictions Changed on Reorder (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=True, shadow=True)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            label = f'{height:.0f}%' if height > 0 else '0%'
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=11)

    save_plot(fig, "2_permutation_invariance")


# ============================================================
#  PLOT 3: Calibration Curve (Learning to be Honest)
# ============================================================
def plot_calibration_curve():
    """Two-panel plot: ECE vs N_calib (left) and Temperature vs N_calib (right).

    The honest result: temperature scaling raises ECE on Banking77 because
    a T>1 spreads an already-thin 77-class softmax further.
    Temperature converges after N=50, showing data sparsity is NOT the cause.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Calibration Scaling on Banking77 (77 classes)', fontweight='bold')

    # ── Left: ECE ──────────────────────────────────────────────
    ax1.axhline(y=ECE_UNCAL, color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Uncalibrated ECE = {ECE_UNCAL:.1f}%')
    ax1.plot(CALIB_SIZES, ECE_CALIBRATED, marker='o', linestyle='-',
             color='#9b59b6', linewidth=3, markersize=8, label='Temperature-Scaled ECE')
    ax1.fill_between(CALIB_SIZES, ECE_UNCAL, ECE_CALIBRATED,
                     where=(ECE_CALIBRATED > ECE_UNCAL),
                     alpha=0.15, color='#e74c3c', label='Calibration degradation')

    ax1.set_xscale('log')
    ax1.set_xticks(CALIB_SIZES)
    ax1.set_xticklabels(CALIB_SIZES)
    ax1.set_xlabel('Calibration Set Size ($N_{calib}$)')
    ax1.set_ylabel('ECE (%)')
    ax1.set_title('ECE vs. Calibration Set Size')
    ax1.set_ylim(0, 50)
    ax1.legend(frameon=True, fontsize=10)

    # ── Right: Temperature ──────────────────────────────────────
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='T = 1 (no scaling)')
    ax2.plot(CALIB_SIZES, TEMPERATURES, marker='s', linestyle='-',
             color='#e67e22', linewidth=3, markersize=8, label='Fitted Temperature')

    ax2.set_xscale('log')
    ax2.set_xticks(CALIB_SIZES)
    ax2.set_xticklabels(CALIB_SIZES)
    ax2.set_xlabel('Calibration Set Size ($N_{calib}$)')
    ax2.set_ylabel('Fitted Temperature (T)')
    ax2.set_title('Temperature Convergence')
    ax2.set_ylim(0.9, 1.5)
    ax2.legend(frameon=True, fontsize=10)

    save_plot(fig, "3_calibration_curve")


if __name__ == "__main__":
    print("Generating Paper Plots...")
    plot_latency_scaling()
    plot_permutation_invariance()
    plot_calibration_curve()
    print("Done! Check the results/plots/ directory.")
