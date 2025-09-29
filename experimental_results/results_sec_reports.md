# SEC Financial Reports Dataset Training Results

## Overview
Training performed on SEC Financial Reports dataset (`JanosAudran/financial-reports-sec`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 3.898 | 3.081 | 2.856 |
| fingpt | 3.974 | 3.153 | 2.927 |
| alpaca | 3.864 | 3.137 | 2.920 |
| fiqa | 3.855 | 3.142 | 2.962 |
| twitter | 3.938 | 3.130 | 2.897 |
| **financial_repor** | **3.716** | **2.963** | **2.767** |
| financial_news | 3.710 | 3.075 | 2.814 |
| wikitext | 3.892 | 3.101 | 2.875 |
| **Average** | **3.856** | **3.098** | **2.877** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 49.30 | 21.77 | 17.39 |
| fingpt | 53.18 | 23.41 | 18.68 |
| alpaca | 47.65 | 23.04 | 18.54 |
| fiqa | 47.22 | 23.15 | 19.34 |
| twitter | 51.30 | 22.86 | 18.12 |
| **financial_repor** | **41.12** | **19.36** | **15.91** |
| financial_news | 40.85 | 21.65 | 16.67 |
| wikitext | 49.02 | 22.21 | 17.72 |
| **Average** | **47.46** | **22.18** | **17.80** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 12.33 | 4.05 | 3.44 |
| Std Deviation | 4.14 | 1.23 | 1.05 |
| Relative Spread (%) | 25.99 | 18.24 | 19.32 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 24.46 | 24.32 | 24.51 |
| Final Eval Loss | 3.856 | 3.098 | 2.877 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **25.4%**
- **Perplexity reduction** from 0.6B to 4B: **62.5%**
- Consistent improvement across all model sizes

### Model Stability
- Good stability across all models:
  - 0.6B: 25.99% relative spread
  - 1.7B: 18.24% relative spread (best)
  - 4B: 19.32% relative spread

### Dataset-Specific Performance
- **Best performance**: financial_repor (training dataset) and financial_news
- **Most challenging**: fingpt and financial_qa datasets
- **Strong domain advantage**: SEC reports model generalizes well to financial news

### Scaling Benefits
1. **0.6B → 1.7B**: Significant improvement (~20% loss reduction, ~53% perplexity reduction)
2. **1.7B → 4B**: Moderate improvement (~7% loss reduction, ~20% perplexity reduction)
3. **Moderate training**: 24 epochs shows good balance without overfitting