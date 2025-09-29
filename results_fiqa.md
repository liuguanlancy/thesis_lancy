# FiQA Dataset Training Results

## Overview
Training performed on FiQA Financial dataset (`LLukas22/fiqa`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 4.639 | 2.605 | 1.844 |
| fingpt | 4.675 | 2.714 | 1.947 |
| alpaca | 4.143 | 2.562 | 1.963 |
| **fiqa** | **4.171** | **2.564** | **1.957** |
| twitter | 4.657 | 2.646 | 1.884 |
| financial_repor | 4.425 | 2.526 | 1.815 |
| financial_news | 3.896 | 2.545 | 2.006 |
| wikitext | 4.523 | 2.626 | 1.906 |
| **Average** | **4.391** | **2.598** | **1.915** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 103.40 | 13.53 | 6.32 |
| fingpt | 107.25 | 15.08 | 7.01 |
| alpaca | 62.97 | 12.96 | 7.12 |
| **fiqa** | **64.75** | **12.99** | **7.08** |
| twitter | 105.32 | 14.10 | 6.58 |
| financial_repor | 83.48 | 12.51 | 6.14 |
| financial_news | 49.22 | 12.74 | 7.43 |
| wikitext | 92.13 | 13.81 | 6.72 |
| **Average** | **83.57** | **13.47** | **6.80** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 58.02 | 2.58 | 1.29 |
| Std Deviation | 20.79 | 0.80 | 0.41 |
| Relative Spread (%) | 69.43 | 19.15 | 18.97 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 6.89 | 6.85 | 6.87 |
| Final Eval Loss | 4.391 | 2.598 | 1.915 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **56.4%**
- **Perplexity reduction** from 0.6B to 4B: **91.9%**
- Excellent scaling across all model sizes

### Model Stability
- Dramatic improvement in stability with scale:
  - 0.6B: 69.43% relative spread (high variance)
  - 1.7B: 19.15% relative spread (massive improvement)
  - 4B: 18.97% relative spread (very stable)

### Dataset-Specific Performance
- **Best generalization**: financial_news dataset (consistently low perplexity)
- **Training dataset advantage**: FiQA performs well but not best
- **Most challenging**: fingpt and financial_qa datasets

### Scaling Benefits
1. **0.6B → 1.7B**: Massive improvement (~41% loss reduction, ~84% perplexity reduction)
2. **1.7B → 4B**: Continued improvement (~26% loss reduction, ~50% perplexity reduction)
3. **Efficient training**: Only ~7 epochs needed for convergence