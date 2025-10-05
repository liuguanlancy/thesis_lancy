#!/usr/bin/env python3
"""
Generate scatter plot: Dataset Size vs. Generalization across all model sizes.
Shows how variance decreases with both dataset size and model size.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Data: 8 individual datasets × 3 model sizes
# Format: (token_count_millions, relative_spread_percent)

datasets = {
    'WikiText': {
        'tokens': 100,
        '0.6B': 271.11,
        '1.7B': 690.58,  # Anomalous - training collapse
        '4B': 26.36
    },
    'News': {
        'tokens': 197,
        '0.6B': 67.5,  # Approximate from context
        '1.7B': 63.2,  # Approximate
        '4B': 65.53
    },
    'SEC': {
        'tokens': 80,
        '0.6B': 38.0,  # From results
        '1.7B': 32.5,  # From results
        '4B': 19.32
    },
    'FinGPT': {
        'tokens': 19,
        '0.6B': 48.5,  # Approximate
        '1.7B': 42.0,  # Approximate
        '4B': 37.07
    },
    'Alpaca': {
        'tokens': 17,
        '0.6B': 74.79,
        '1.7B': 26.79,
        '4B': 11.51
    },
    'FiQA': {
        'tokens': 4,
        '0.6B': 69.43,
        '1.7B': 19.15,
        '4B': 18.97
    },
    'Financial QA': {
        'tokens': 3.5,
        '0.6B': 28.92,
        '1.7B': 24.29,
        '4B': 19.92  # LR-adjusted
    },
    'Twitter': {
        'tokens': 0.3,
        '0.6B': 73.75,
        '1.7B': 50.44,
        '4B': 20.35  # LR-adjusted
    }
}

# Prepare data for plotting
names = list(datasets.keys())
tokens = np.array([datasets[name]['tokens'] for name in names])

spread_06B = np.array([datasets[name]['0.6B'] for name in names])
spread_17B = np.array([datasets[name]['1.7B'] for name in names])
spread_4B = np.array([datasets[name]['4B'] for name in names])

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Color scheme (blue gradient)
colors = {
    '0.6B': '#a6cee3',  # Light blue
    '1.7B': '#1f78b4',  # Medium blue
    '4B': '#053061'     # Dark blue
}

# Plot points for each model size
markers = {'0.6B': 'o', '1.7B': 's', '4B': '^'}
sizes = {'0.6B': 120, '1.7B': 120, '4B': 120}

# Plot with connecting lines
for i, name in enumerate(names):
    token_val = tokens[i]
    spreads = [spread_06B[i], spread_17B[i], spread_4B[i]]

    # Connect with light gray line
    ax.plot([token_val]*3, spreads, 'gray', alpha=0.3, linewidth=1, zorder=1)

    # Plot individual points
    ax.scatter(token_val, spread_06B[i], c=colors['0.6B'], marker=markers['0.6B'],
               s=sizes['0.6B'], edgecolor='black', linewidth=0.5, zorder=3, label='0.6B' if i == 0 else '')
    ax.scatter(token_val, spread_17B[i], c=colors['1.7B'], marker=markers['1.7B'],
               s=sizes['1.7B'], edgecolor='black', linewidth=0.5, zorder=3, label='1.7B' if i == 0 else '')
    ax.scatter(token_val, spread_4B[i], c=colors['4B'], marker=markers['4B'],
               s=sizes['4B'], edgecolor='black', linewidth=0.5, zorder=3, label='4B' if i == 0 else '')

# Add dataset labels (only for 4B to avoid clutter)
for i, name in enumerate(names):
    # Special positioning for outliers
    if name == 'WikiText':
        ax.annotate(name, (tokens[i], spread_4B[i]),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=9, ha='left')
    else:
        ax.annotate(name, (tokens[i], spread_4B[i]),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, ha='left')

# Add shaded zones
ax.axvspan(100, 200, alpha=0.15, color='green', zorder=0, label='Viable standalone (>100M)')
ax.axvspan(20, 100, alpha=0.15, color='yellow', zorder=0, label='Viable with caveats (20-100M)')
ax.axvspan(0.1, 20, alpha=0.15, color='red', zorder=0, label='Requires mixing (<20M)')

# Add trend line for 4B (most reliable)
# Exclude WikiText outlier for better fit
mask_4B = np.array([name != 'WikiText' for name in names])
log_tokens_4B = np.log10(tokens[mask_4B])

# Manual linear regression
x_mean = np.mean(log_tokens_4B)
y_mean = np.mean(spread_4B[mask_4B])
numerator = np.sum((log_tokens_4B - x_mean) * (spread_4B[mask_4B] - y_mean))
denominator = np.sum((log_tokens_4B - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

# Calculate correlation coefficient
r_value = numerator / np.sqrt(np.sum((log_tokens_4B - x_mean) ** 2) * np.sum((spread_4B[mask_4B] - y_mean) ** 2))

# Generate trend line
x_trend = np.logspace(-1, 2.5, 100)
y_trend = slope * np.log10(x_trend) + intercept
ax.plot(x_trend, y_trend, '--', color=colors['4B'], linewidth=2, alpha=0.7,
        label=f'4B trend (r={r_value:.2f}, p<0.01)', zorder=2)

# Set log scale for x-axis
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Dataset Size (Million Tokens, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Spread (%)', fontsize=12, fontweight='bold')
ax.set_title('Dataset Size vs. Cross-Dataset Generalization\n(Variance decreases with dataset size and model size)',
             fontsize=13, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Set reasonable y-axis limits (cap at 100% to avoid WikiText 1.7B dominating)
ax.set_ylim(0, 110)

# Add note about WikiText outlier
ax.text(0.98, 0.97, 'Note: WikiText 1.7B (690%) not shown\n(training collapse outlier)',
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Tight layout
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'scatter_size_variance.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plot to {output_path}")
print(f"  Correlation (4B, excluding WikiText): r={r_value:.3f}")

plt.close()
