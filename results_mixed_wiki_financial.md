# Mixed Wiki + Financial Datasets Training Results

## Overview
Training performed on a mixture of WikiText and financial datasets with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 4.580 | 3.870 | 3.458 |
| fingpt | 4.436 | 3.750 | 3.365 |
| alpaca | 4.070 | 3.478 | 3.145 |
| fiqa | 4.144 | 3.556 | 3.243 |
| twitter | 4.586 | 3.880 | 3.481 |
| financial_repor | 4.351 | 3.693 | 3.329 |
| financial_news | 3.655 | 3.126 | 2.767 |
| wikitext | 4.408 | 3.737 | 3.322 |
| **Average** | **4.279** | **3.636** | **3.264** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 97.49 | 47.94 | 31.76 |
| fingpt | 84.43 | 42.50 | 28.92 |
| alpaca | 58.56 | 32.38 | 23.23 |
| fiqa | 63.03 | 35.04 | 25.61 |
| twitter | 98.13 | 48.42 | 32.48 |
| financial_repor | 77.57 | 40.17 | 27.91 |
| financial_news | 38.68 | 22.79 | 15.91 |
| wikitext | 82.10 | 41.95 | 27.72 |
| **Average** | **75.00** | **38.90** | **26.69** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 59.45 | 25.63 | 16.56 |
| Std Deviation | 19.10 | 8.01 | 4.95 |
| Relative Spread (%) | 79.26 | 65.89 | 62.05 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 5.64 | 5.65 | 5.62 |
| Final Eval Loss | 4.279 | 3.636 | 3.264 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **23.7%**
- **Perplexity reduction** from 0.6B to 4B: **64.4%**
- Consistent improvement across all model sizes

### Model Stability
- Gradual improvement in stability:
  - 0.6B: 79.26% relative spread (high variance)
  - 1.7B: 65.89% relative spread (improved)
  - 4B: 62.05% relative spread (most stable)

### Dataset-Specific Performance
- **Best generalization**: financial_news dataset (lowest perplexity across all models)
- **WikiText inclusion benefit**: Better wikitext performance than pure financial training
- **Balanced results**: Mixed training provides robust performance across domains

### Scaling Benefits
1. **0.6B → 1.7B**: Significant improvement (~15% loss reduction, ~48% perplexity reduction)
2. **1.7B → 4B**: Moderate improvement (~10% loss reduction, ~31% perplexity reduction)
3. **Domain diversity**: Wiki+Financial mixture improves generalization
4. **Efficient training**: Only ~5.6 epochs needed, similar to pure financial mixture