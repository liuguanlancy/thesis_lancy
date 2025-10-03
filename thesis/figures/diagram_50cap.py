#!/usr/bin/env python3
"""
Generate 50cap strategy illustration showing token allocation in Mixed Financial dataset.
Shows how News is capped at 50% and others sampled proportionally.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Original dataset sizes (millions of tokens)
original_sizes = {
    'News Articles': 197,
    'SEC Reports': 80,
    'FinGPT Sentiment': 19.1,
    'Finance Alpaca': 17.2,
    'FiQA': 4.3,
    'Financial QA 10K': 3.5,
    'Twitter Sentiment': 0.3
}

# Total original: 321.4M tokens
total_original = sum(original_sizes.values())

# Apply 50cap strategy
# Step 1: News exceeds 50%, cap it at 50%
# Step 2: Remaining datasets get proportionally sampled from other 50%

# News is capped at 50% of final mixture
news_capped = 160.5  # This will be 50% of 321M

# Other datasets total originally: 321.4 - 197 = 124.4M
# They need to fit in the other 50%: 160.5M
# Scaling factor: 160.5 / 124.4 = 1.29

other_datasets_original = total_original - original_sizes['News Articles']
scaling_factor = news_capped / other_datasets_original

# Calculate final allocations
final_sizes = {}
final_sizes['News Articles'] = news_capped

for name, size in original_sizes.items():
    if name != 'News Articles':
        final_sizes[name] = size * scaling_factor

# Verify total
total_final = sum(final_sizes.values())

# Prepare data for plotting
names = list(final_sizes.keys())
sizes = [final_sizes[name] for name in names]
percentages = [(size / total_final) * 100 for size in sizes]

# Sort by size (descending)
sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
names = [names[i] for i in sorted_indices]
sizes = [sizes[i] for i in sorted_indices]
percentages = [percentages[i] for i in sorted_indices]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Color scheme (colorblind-friendly)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

# Left plot: Pie chart
wedges, texts, autotexts = ax1.pie(sizes, labels=names, autopct='%1.1f%%',
                                     colors=colors, startangle=90,
                                     textprops={'fontsize': 9})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax1.set_title('50cap Token Allocation\n(Pie Chart View)',
              fontsize=12, fontweight='bold', pad=15)

# Right plot: Horizontal stacked bar
y_pos = 0
left_edge = 0
bars = []

for i, (name, size, pct) in enumerate(zip(names, sizes, percentages)):
    bar = ax2.barh(y_pos, size, left=left_edge, color=colors[i],
                   edgecolor='white', linewidth=2)
    bars.append(bar)

    # Add label in the middle of the segment
    if size > 15:  # Only show text if segment is large enough
        ax2.text(left_edge + size/2, y_pos, f'{name}\n{pct:.1f}%',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    else:
        ax2.text(left_edge + size/2, y_pos, f'{pct:.1f}%',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    left_edge += size

# Customize right plot
ax2.set_yticks([])
ax2.set_xlabel('Tokens (Millions)', fontsize=11, fontweight='bold')
ax2.set_title('50cap Token Allocation\n(Stacked Bar View)',
              fontsize=12, fontweight='bold', pad=15)
ax2.set_xlim(0, total_final)

# Add total annotation
ax2.text(total_final/2, -0.3, f'Total: {total_final:.1f}M tokens',
         ha='center', fontsize=10, fontweight='bold')

# Add caption explaining 50cap
fig.text(0.5, 0.02,
         'News Articles capped at 50% (reduced from 197M to 160.5M)\n'
         'Other 6 datasets sampled proportionally from remaining 50% (scaled by 1.29×)',
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Tight layout with room for caption
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save
output_path = os.path.join(os.path.dirname(__file__), 'diagram_50cap.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved 50cap diagram to {output_path}")
print(f"  Total tokens: {total_final:.1f}M")
print(f"  News: {final_sizes['News Articles']:.1f}M ({(final_sizes['News Articles']/total_final)*100:.1f}%)")

plt.close()
