# FinGPT Sentiment Dataset Training Results

## Overview
Training performed on FinGPT Sentiment dataset (`FinGPT/fingpt-sentiment-train`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 3.663 | 2.384 | 1.831 |
| **fingpt** | **3.490** | **2.258** | **1.735** |
| alpaca | 3.571 | 2.548 | 2.113 |
| fiqa | 3.574 | 2.549 | 2.099 |
| twitter | 3.677 | 2.403 | 1.866 |
| financial_repor | 3.526 | 2.315 | 1.824 |
| financial_news | 3.358 | 2.449 | 2.069 |
| wikitext | 3.656 | 2.439 | 1.986 |
| **Average** | **3.564** | **2.418** | **1.940** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 38.96 | 10.85 | 6.24 |
| **fingpt** | **32.78** | **9.56** | **5.67** |
| alpaca | 35.55 | 12.78 | 8.27 |
| fiqa | 35.64 | 12.79 | 8.16 |
| twitter | 39.54 | 11.05 | 6.46 |
| financial_repor | 33.97 | 10.12 | 6.20 |
| financial_news | 28.72 | 11.58 | 7.92 |
| wikitext | 38.70 | 11.46 | 7.29 |
| **Average** | **35.48** | **11.27** | **7.03** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 10.82 | 3.23 | 2.60 |
| Std Deviation | 3.43 | 1.07 | 0.95 |
| Relative Spread (%) | 30.49 | 28.63 | 37.07 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 29.63 | 29.56 | 29.56 |
| Final Eval Loss | 3.564 | 2.418 | 1.940 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **45.6%**
- **Perplexity reduction** from 0.6B to 4B: **80.2%**
- Consistent improvement with model size, unlike financial_qa

### Model Stability
- Mixed stability patterns:
  - 0.6B: 30.49% relative spread
  - 1.7B: 28.63% relative spread (improved)
  - 4B: 37.07% relative spread (less stable despite better performance)

### Dataset-Specific Performance
- **Best performance**: financial_news dataset (lowest perplexity for 0.6B)
- **Training advantage**: FinGPT shows strong performance as expected
- **Most challenging**: twitter and financial_qa for smaller models

### Scaling Benefits
1. **0.6B → 1.7B**: Significant improvement (~32% loss reduction, ~68% perplexity reduction)
2. **1.7B → 4B**: Continued strong improvement (~20% loss reduction, ~38% perplexity reduction)
3. **Linear scaling**: More consistent scaling benefits compared to financial_qa dataset