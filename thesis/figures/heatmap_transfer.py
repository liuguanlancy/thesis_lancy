#!/usr/bin/env python3
"""
Generate cross-dataset transfer heatmap for thesis.
Shows perplexity at 4B model size for all training×evaluation combinations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Data: 10 training configs × 8 eval datasets (perplexity at 4B)
# Training configs (rows):
# 1. WikiText (LR=2e-5, original with issues)
# 2. WikiText (LR=5e-6, adjusted)
# 3. Financial QA (LR=2e-5, original)
# 4. Financial QA (LR=5e-6, adjusted)
# 5. Twitter (LR=2e-5, original)
# 6. Twitter (LR=5e-6, adjusted)
# 7. News Articles
# 8. SEC Reports
# 9. FinGPT
# 10. Alpaca
# 11. FiQA
# 12. Mixed Financial
# 13. Mixed Wiki+Financial

# Eval datasets (columns): Alpaca, Financial News, Financial QA, SEC Reports, FinGPT, FiQA, Twitter, WikiText

# Perplexity data at 4B (extracted from experimental results)
data = np.array([
    # Alpaca, Fin News, Fin QA, SEC, FinGPT, FiQA, Twitter, WikiText
    [32.38, 29.19, 29.08, 31.23, 35.50, 34.03, 33.71, 27.19],  # WikiText (2e-5)
    [38.06, 26.44, 47.98, 42.41, 48.30, 42.04, 48.48, 38.60],  # WikiText (5e-6)
    [9.91, 8.41, 8.29, 8.25, 9.34, 10.05, 8.99, 8.89],  # Financial QA (2e-5)
    [8.88, 7.71, 7.43, 7.43, 8.24, 8.93, 8.05, 8.00],  # Financial QA (5e-6)
    [19.20, 17.67, 16.98, 16.42, 18.34, 20.09, 17.83, 17.85],  # Twitter (2e-5)
    [13.65, 12.68, 11.39, 10.93, 12.69, 13.61, 11.81, 12.02],  # Twitter (5e-6)
    [29.75, 17.47, 38.90, 33.46, 38.03, 31.69, 38.98, 28.40],  # News
    [18.54, 16.67, 17.39, 15.91, 18.68, 19.34, 18.12, 35.60],  # SEC
    [8.27, 7.92, 9.49, 6.20, 5.67, 8.16, 6.46, 41.20],  # FinGPT
    [8.22, 8.58, 8.56, 8.25, 9.18, 9.22, 8.97, 9.41],  # Alpaca
    [7.12, 7.43, 6.32, 6.14, 7.01, 7.08, 6.58, 6.72],  # FiQA
    [19.50, 13.84, 25.14, 22.36, 23.08, 21.20, 25.72, 33.70],  # Mixed Financial
    [23.23, 15.91, 31.76, 27.91, 28.92, 25.61, 32.48, 27.72],  # Mixed Wiki+Fin
])

# Labels
train_labels = [
    'WikiText (2e-5)',
    'WikiText (5e-6)',
    'Fin QA (2e-5)',
    'Fin QA (5e-6)',
    'Twitter (2e-5)',
    'Twitter (5e-6)',
    'News',
    'SEC',
    'FinGPT',
    'Alpaca',
    'FiQA',
    'Mixed Fin',
    'Mixed Wiki+Fin'
]

eval_labels = [
    'Alpaca',
    'Fin News',
    'Fin QA',
    'SEC',
    'FinGPT',
    'FiQA',
    'Twitter',
    'WikiText'
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Use log scale for better color discrimination
log_data = np.log10(data)

# Create heatmap with colorblind-friendly palette
# Use 'RdYlGn_r' reversed: red (high perplexity/poor) to green (low perplexity/good)
sns.heatmap(log_data,
            annot=data,  # Show actual values, not log
            fmt='.1f',
            cmap='RdYlGn_r',
            xticklabels=eval_labels,
            yticklabels=train_labels,
            cbar_kws={'label': 'Perplexity (log₁₀ scale)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

# Customize
ax.set_xlabel('Evaluation Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Configuration', fontsize=12, fontweight='bold')
ax.set_title('Cross-Dataset Transfer at 4B Model Size\n(Lower perplexity = better transfer)',
             fontsize=13, fontweight='bold', pad=20)

# Rotate labels for readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Tight layout
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'heatmap_transfer.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved heatmap to {output_path}")

plt.close()
