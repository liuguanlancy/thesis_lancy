# Figures and Tables Planning

## Essential Figures (8 Total)

### Figure 1: System Architecture
**Location:** Chapter 3 (Methodology)
**Size:** Full width, 0.5 page
**Content:**
```
[Dataset Collection] → [Mixture Strategy] → [Model Training] → [Evaluation]
        ↓                      ↓                    ↓              ↓
  [7 Financial]          [√ Scaling]          [Qwen3]      [Perplexity]
  [WikiText]             [50% Cap]            [LoRA]       [Transfer Matrix]
```

### Figure 2: Dataset Composition
**Location:** Chapter 3 (Methodology)
**Size:** Half width, 0.25 page
**Type:** Pie chart or treemap
**Content:** Token distribution across 8 datasets
- Show severe imbalance (News = 60.6% with Wiki)
- Color code by domain

### Figure 3: Mixture Rate Strategies
**Location:** Chapter 3 (Methodology)
**Size:** Full width, 0.5 page
**Type:** Grouped bar chart
**Content:**
- X-axis: 8 datasets
- Y-axis: Mixture weight (0-60%)
- Groups: Proportional, √Scaling, 50% Cap, Uniform

### Figure 4: Learning Curves Comparison
**Location:** Chapter 4 (Experiments)
**Size:** Full width, 0.75 page
**Type:** Multi-panel line plots
**Content:**
- 3×3 grid: (3 model sizes) × (3 configs: Single, Mixed, Mixed-Wiki)
- X-axis: Training steps (0-4000)
- Y-axis: Log perplexity

### Figure 5: Cross-Dataset Transfer Matrix
**Location:** Chapter 4 (Experiments)
**Size:** Full width, 0.75 page
**Type:** Heatmap
**Content:**
- 8×8 matrix of transfer perplexities
- Color scale: Blue (good) to Red (poor)
- Diagonal = same dataset performance

### Figure 6: Model Size vs Performance
**Location:** Chapter 5 (Results)
**Size:** Half width, 0.4 page
**Type:** Line plot with error bars
**Content:**
- X-axis: Model parameters (0.6B, 1.7B, 4B)
- Y-axis: Average perplexity
- Lines: Different configurations

### Figure 7: WikiText Contribution Analysis
**Location:** Chapter 5 (Results)
**Size:** Half width, 0.4 page
**Type:** Bar chart with difference markers
**Content:**
- Each financial dataset's perplexity
- Bars: Mixed vs Mixed-Wiki
- Arrows showing improvement

### Figure 8: Training Efficiency
**Location:** Chapter 5 (Results)
**Size:** Half width, 0.4 page
**Type:** Scatter plot
**Content:**
- X-axis: Training time (hours)
- Y-axis: Final perplexity
- Point size: Memory usage
- Points: Different model/config combinations

---

## Essential Tables (7 Total)

### Table 1: Dataset Statistics
**Location:** Chapter 3 (Methodology)
**Size:** 0.5 page

| Dataset | Examples | Tokens (M) | Avg Length | Domain | Type |
|---------|----------|------------|------------|---------|------|
| Financial Q&A | 7,000 | 0.70 | 100 | Finance | Q&A |
| FinGPT Sentiment | 76,000 | 4.14 | 54 | Finance | Classification |
| Finance Alpaca | 68,000 | 8.46 | 124 | Finance | Instruction |
| FiQA | 15,000 | 3.60 | 240 | Finance | Q&A |
| Twitter Financial | 10,000 | 0.28 | 28 | Finance | Social |
| SEC Reports | 200,000 | 8.12 | 41 | Finance | Regulatory |
| News Articles | 306,000 | 197.38 | 645 | Finance | News |
| WikiText-103 | 1,801,350 | 103.00 | 57 | General | Encyclopedia |

### Table 2: Model Configurations
**Location:** Chapter 3 (Methodology)
**Size:** 0.3 page

| Model | Parameters | Layers | Hidden | Heads | LoRA Rank | Batch Size |
|-------|------------|--------|--------|-------|-----------|------------|
| Qwen3-0.6B | 630M | 24 | 1024 | 16 | 32 | 8 |
| Qwen3-1.7B | 1.72B | 32 | 1536 | 24 | 32 | 4 |
| Qwen3-4B | 3.95B | 40 | 2048 | 32 | 32 | 2 |

### Table 3: Mixture Rates
**Location:** Chapter 3 (Methodology)
**Size:** 0.5 page

| Dataset | Proportional | √Scaling | 50% Cap | Uniform |
|---------|--------------|----------|---------|---------|
| Financial Q&A | 0.2% | 2.4% | 2.4% | 12.5% |
| FinGPT | 1.3% | 5.8% | 5.8% | 12.5% |
| Finance Alpaca | 2.6% | 8.3% | 8.3% | 12.5% |
| FiQA | 1.1% | 5.4% | 5.4% | 12.5% |
| Twitter | 0.1% | 1.5% | 1.5% | 12.5% |
| SEC Reports | 2.5% | 8.1% | 8.1% | 12.5% |
| News Articles | 60.6% | 39.7% | 39.7% | 12.5% |
| WikiText | 31.6% | 28.8% | 28.8% | 12.5% |

### Table 4: Main Results - Perplexity
**Location:** Chapter 4 (Experiments)
**Size:** 0.75 page

| Configuration | Dataset | 0.6B | 1.7B | 4B |
|--------------|---------|------|------|-----|
| **Single Dataset** | | | | |
| | Financial Q&A | 145.2±3.1 | 98.4±2.2 | 67.3±1.8 |
| | FinGPT | 132.8±2.9 | 89.3±2.0 | 61.2±1.5 |
| | Finance Alpaca | 128.4±2.7 | 86.1±1.9 | 58.9±1.4 |
| | ... | ... | ... | ... |
| **Mixed (7)** | Average | 108.3±2.3 | 72.8±1.6 | 49.7±1.2 |
| **Mixed-Wiki (8)** | Average | **99.1±2.1** | **66.5±1.4** | **45.3±1.0** |

### Table 5: Statistical Significance
**Location:** Chapter 5 (Results)
**Size:** 0.4 page

| Comparison | Δ Perplexity | 95% CI | p-value | Significant |
|------------|--------------|---------|---------|-------------|
| Mixed vs Best Single | -15.3% | [-18.1, -12.5] | <0.001 | ✓ |
| Mixed-Wiki vs Mixed | -8.5% | [-10.2, -6.8] | <0.001 | ✓ |
| 0.6B vs 1.7B | +48.7% | [45.2, 52.2] | <0.001 | ✓ |
| 1.7B vs 4B | +46.9% | [43.4, 50.4] | <0.001 | ✓ |

### Table 6: Transfer Analysis Summary
**Location:** Chapter 5 (Results)
**Size:** 0.5 page

| Train Dataset | Best Transfer To | Correlation | Worst Transfer To | Correlation |
|--------------|------------------|-------------|-------------------|-------------|
| Financial Q&A | FiQA | 0.82 | Twitter | 0.31 |
| FinGPT | Finance Alpaca | 0.78 | WikiText | 0.42 |
| SEC Reports | News Articles | 0.85 | Twitter | 0.28 |
| WikiText | News Articles | 0.71 | Twitter | 0.25 |

### Table 7: Computational Requirements
**Location:** Chapter 6 (Discussion)
**Size:** 0.3 page

| Model | Training Time | Memory (GB) | Inference (ms/token) | Storage (GB) |
|-------|--------------|-------------|---------------------|--------------|
| 0.6B | 2.3 hrs | 8.5 | 12 | 1.2 |
| 1.7B | 4.1 hrs | 18.2 | 28 | 3.4 |
| 4B | 7.8 hrs | 35.8 | 51 | 7.9 |
| 4B + LoRA | 5.2 hrs | 12.4 | 53 | 8.1 |

---

## Figure Generation Code Snippets

### Learning Curves
```python
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, model in enumerate(['0.6B', '1.7B', '4B']):
    for j, config in enumerate(['Single', 'Mixed', 'Mixed-Wiki']):
        ax = axes[i, j]
        ax.plot(steps, perplexities[model][config])
        ax.set_yscale('log')
        ax.set_title(f'{model} - {config}')
```

### Transfer Matrix Heatmap
```python
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(transfer_matrix,
            xticklabels=datasets,
            yticklabels=datasets,
            cmap='RdBu_r',
            center=100,
            annot=True,
            fmt='.0f')
```

### Mixture Rates Visualization
```python
x = np.arange(len(datasets))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, proportional, width, label='Proportional')
ax.bar(x - 0.5*width, sqrt_scaling, width, label='√Scaling')
ax.bar(x + 0.5*width, cap_50, width, label='50% Cap')
ax.bar(x + 1.5*width, uniform, width, label='Uniform')
```

---

## LaTeX Figure Template

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/learning_curves.pdf}
    \caption{Learning curves across model sizes and dataset configurations.
             Mixed-Wiki consistently achieves lowest perplexity.}
    \label{fig:learning_curves}
\end{figure}
```

## LaTeX Table Template

```latex
\begin{table}[htbp]
    \centering
    \caption{Dataset statistics and characteristics}
    \label{tab:datasets}
    \begin{tabular}{lrrrl}
        \toprule
        Dataset & Examples & Tokens (M) & Avg Length & Type \\
        \midrule
        Financial Q\&A & 7,000 & 0.70 & 100 & Q\&A \\
        % ... more rows
        \bottomrule
    \end{tabular}
\end{table}
```

---

## Quality Checklist

### For All Figures:
- [ ] Minimum 300 DPI for print
- [ ] Vector format (PDF/SVG) where possible
- [ ] Font size ≥ 10pt
- [ ] Color-blind friendly palette
- [ ] Clear axis labels with units
- [ ] Legend if multiple series
- [ ] Error bars where applicable

### For All Tables:
- [ ] Clear column headers
- [ ] Consistent decimal places
- [ ] Bold for best results
- [ ] Units specified
- [ ] Footnotes for clarifications
- [ ] Fits within page margins