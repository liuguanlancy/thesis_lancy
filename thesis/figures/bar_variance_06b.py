#!/usr/bin/env python3
"""
Generate variance comparison bar chart for all experiments at 4B.
Shows relative spread % sorted to highlight Mixed Financial's advantage.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Data: relative spread % at 4B for all 10 experiments
# Note: For WikiText, using the spread across financial evaluations (~53% estimated)
# not the 26.36% which is only on WikiText itself

experiments = {
    'Alpaca': {'spread': 74.79, 'tokens': 8.46, 'epochs': 11.8},
    'FiQA': {'spread': 69.43, 'tokens': 3.60, 'epochs': 27.7},
    'SEC Reports': {'spread': 38.0, 'tokens': 8.12, 'epochs': 12.3},
    'Financial QA': {'spread': 28.92, 'tokens': 0.70, 'epochs': 142.7},
    'Twitter': {'spread': 73.75, 'tokens': 0.28, 'epochs': 351.7},
    'FinGPT': {'spread': 48.5, 'tokens': 4.14, 'epochs': 24.2},
    'WikiText': {'spread': 271.11, 'tokens': 123.58, 'epochs': 0.8},  # 0.6B spread on financial evals
    'Mixed Financial': {'spread': 97.75, 'tokens': 219.77, 'epochs': 0.5},
    'Mixed Wiki+Fin': {'spread': 84.05, 'tokens': 343.35, 'epochs': 0.3},
    'News Articles': {'spread': 67.5, 'tokens': 194.47, 'epochs': 0.5},
}

# Sort by spread (ascending)
sorted_exps = sorted(experiments.items(), key=lambda x: x[1]['spread'])
names = [name for name, _ in sorted_exps]
spreads = [data['spread'] for _, data in sorted_exps]
tokens = [data['tokens'] for _, data in sorted_exps]
epochs = [data['epochs'] for _, data in sorted_exps]

# Create labels with token counts and epoch counts
def format_tokens(t):
    if t >= 1:
        return f'{t:.2f}M' if t < 100 else f'{t:.1f}M'
    else:
        return f'{t*1000:.0f}K' if t < 1 else f'{t:.2f}M'

def format_epochs(e):
    if e >= 10:
        return f'{e:.1f}ep'
    elif e >= 1:
        return f'{e:.1f}ep'
    else:
        return f'{e:.1f}ep'

labels = [f'{name} ({format_tokens(tok)}, {format_epochs(ep)})' for name, tok, ep in zip(names, tokens, epochs)]

# Use single color for all bars
single_color = '#4472C4'  # Professional blue

# No special highlighting
edge_colors = ['none' for _ in names]
edge_widths = [0 for _ in names]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create horizontal bars
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, spreads, color=single_color, edgecolor=edge_colors, linewidth=edge_widths)

# Customize
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Relative Spread (%)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Dataset Variance at 0.6B Model Size\n(Lower = more consistent generalization)',
             fontsize=13, fontweight='bold', pad=20)

# Add value labels at end of bars
for i, (spread, bar) in enumerate(zip(spreads, bars)):
    ax.text(spread + 1.5, i, f'{spread:.1f}%',
            va='center', ha='left', fontsize=9)

# Grid
ax.grid(True, axis='x', alpha=0.3, linestyle=':', zorder=0)

# Tight layout
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'bar_variance_06b.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved bar chart to {output_path}")

plt.close()
