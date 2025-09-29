# Mixed Financial Datasets Training Results

## Overview
Training performed on a mixture of financial datasets with evaluation across 7 different financial and general datasets (no wikitext eval for mixed training).

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 5.213 | 3.745 | 3.225 |
| fingpt | 5.037 | 3.633 | 3.139 |
| alpaca | 4.536 | 3.385 | 2.970 |
| fiqa | 4.630 | 3.461 | 3.054 |
| twitter | 5.207 | 3.759 | 3.247 |
| financial_repor | 4.939 | 3.579 | 3.107 |
| financial_news | 4.032 | 3.054 | 2.627 |
| **Average** | **4.799** | **3.517** | **3.053** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 183.72 | 42.30 | 25.14 |
| fingpt | 153.94 | 37.82 | 23.08 |
| alpaca | 93.35 | 29.53 | 19.50 |
| fiqa | 102.47 | 31.85 | 21.20 |
| twitter | 182.63 | 42.91 | 25.72 |
| financial_repor | 139.62 | 35.83 | 22.36 |
| financial_news | 56.35 | 21.19 | 13.84 |
| **Average** | **130.30** | **34.49** | **21.55** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 127.37 | 21.71 | 11.89 |
| Std Deviation | 44.48 | 7.10 | 3.72 |
| Relative Spread (%) | 97.75 | 62.95 | 55.16 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 5.46 | 5.48 | 5.43 |
| Final Eval Loss | 4.799 | 3.517 | 3.053 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **36.4%**
- **Perplexity reduction** from 0.6B to 4B: **83.5%**
- Excellent scaling across all model sizes

### Model Stability
- Dramatic improvement in stability with scale:
  - 0.6B: 97.75% relative spread (very high variance)
  - 1.7B: 62.95% relative spread (improved)
  - 4B: 55.16% relative spread (further improved)

### Dataset-Specific Performance
- **Best generalization**: financial_news dataset (consistently lowest perplexity)
- **Most challenging**: financial_qa and twitter datasets
- **Balanced performance**: Mixed training provides reasonable performance across all datasets

### Scaling Benefits
1. **0.6B → 1.7B**: Major improvement (~27% loss reduction, ~74% perplexity reduction)
2. **1.7B → 4B**: Continued improvement (~13% loss reduction, ~38% perplexity reduction)
3. **Efficient training**: Only ~5.5 epochs needed for convergence
4. **Mixture advantage**: More robust than single-dataset training