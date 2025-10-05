#!/usr/bin/env python3
"""
Generate pie chart for Mixed Financial dataset (7 datasets) with 50cap strategy.
Shows token allocation from 100M budget. Raw corpus: 219.77M tokens.
"""

import matplotlib.pyplot as plt
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Consistent color mapping across all diagrams
# No yellow - all colors work well with white text
DATASET_COLORS = {
    'News Articles': '#e41a1c',      # Red - largest
    'WikiText': '#377eb8',           # Blue - general corpus
    'Finance Alpaca': '#4daf4a',     # Green
    'SEC Reports': '#984ea3',        # Purple
    'FinGPT Sentiment': '#ff7f00',   # Orange
    'FiQA': '#a65628',               # Brown
    'Financial QA': '#f781bf',       # Pink
    'Twitter Sentiment': '#66a61e'   # Olive (replaces yellow)
}

# 100M token budget allocation (from ./count file - 50cap strategy)
# Raw corpus: 219.77M tokens
final_sizes = {
    'News Articles': 50.0,
    'Finance Alpaca': 13.0,
    'SEC Reports': 13.0,
    'FinGPT Sentiment': 9.0,
    'FiQA': 8.5,
    'Financial QA': 4.0,
    'Twitter Sentiment': 2.5
}

# Prepare data for plotting
names = list(final_sizes.keys())
sizes = [final_sizes[name] for name in names]

# Sort by size (descending)
sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
names = [names[i] for i in sorted_indices]
sizes = [sizes[i] for i in sorted_indices]

# Map colors to datasets
colors = [DATASET_COLORS[name] for name in names]

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Pie chart
wedges, texts, autotexts = ax.pie(sizes, labels=names, autopct='%1.1f%%',
                                   colors=colors, startangle=90,
                                   textprops={'fontsize': 10})

# Make percentage text bold and white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

# Save
output_path = os.path.join(os.path.dirname(__file__), 'diagram_50cap.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Mixed Financial pie chart to {output_path}")
print(f"  100M token budget (raw corpus: 219.77M)")
print(f"  News: 50.0M (50.0%), SEC: 13.0M (13.0%), Alpaca: 13.0M (13.0%)")
print(f"  Using consistent colors across all diagrams")

plt.close()
