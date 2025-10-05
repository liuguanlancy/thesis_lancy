#!/usr/bin/env python3
"""
Generate cross-dataset transfer heatmap for thesis.
Builds the 4B heatmap directly from experimental_results markdown tables
to avoid hardcoded mistakes and keep in sync.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(__file__))
from heatmap_transfer_utils import build_rows_for_size

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300


def main() -> None:
    train_labels, eval_labels, data = build_rows_for_size("4b")

    # Compute average over valid evaluations (excluding NaN and inf)
    averages = []
    for row in data:
        valid_mask = ~(np.isnan(row) | np.isinf(row))
        if valid_mask.sum() > 0:
            averages.append(row[valid_mask].mean())
        else:
            averages.append(np.nan)

    # Add average column to data
    data = np.column_stack([data, averages])
    eval_labels.append("Average")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(train_labels) + 4)))

    # Mask NaNs (missing evals like WikiText for Mixed Fin) and log-transform
    mask = np.isnan(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_data = np.log10(data)

    sns.heatmap(
        log_data,
        mask=mask,
        annot=np.where(mask, np.nan, data),  # show actual values when present
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0.6,
        vmax=2.0,
        xticklabels=eval_labels,
        yticklabels=train_labels,
        cbar_kws={"label": "Perplexity (log₁₀ scale)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Evaluation Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Configuration", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cross-Dataset Transfer at 4B Model Size\n(Lower perplexity = better transfer)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "heatmap_transfer.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
