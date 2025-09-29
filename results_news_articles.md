# Financial News Articles Dataset Training Results

## Overview
Training performed on Financial News Articles dataset with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 5.113 | 3.903 | 3.661 |
| fingpt | 5.081 | 3.903 | 3.638 |
| alpaca | 4.568 | 3.609 | 3.393 |
| fiqa | 4.618 | 3.655 | 3.456 |
| twitter | 5.107 | 3.910 | 3.663 |
| financial_repor | 4.850 | 3.730 | 3.510 |
| **financial_news** | **3.956** | **3.131** | **2.861** |
| wikitext | 4.947 | 3.810 | 3.536 |
| **Average** | **4.780** | **3.706** | **3.465** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 166.10 | 49.53 | 38.90 |
| fingpt | 160.92 | 49.56 | 38.03 |
| alpaca | 96.31 | 36.92 | 29.75 |
| fiqa | 101.32 | 38.68 | 31.69 |
| twitter | 165.22 | 49.88 | 38.98 |
| financial_repor | 127.73 | 41.68 | 33.46 |
| **financial_news** | **52.25** | **22.91** | **17.47** |
| wikitext | 140.71 | 45.17 | 34.33 |
| **Average** | **126.32** | **41.79** | **32.82** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 113.85 | 26.98 | 21.51 |
| Std Deviation | 37.94 | 8.57 | 6.62 |
| Relative Spread (%) | 90.13 | 64.56 | 65.53 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 2.85 | 2.85 | 2.86 |
| Final Eval Loss | 4.780 | 3.706 | 3.465 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **27.5%**
- **Perplexity reduction** from 0.6B to 4B: **74.0%**
- Consistent improvement with model scaling

### Model Stability
- High variance in smaller models:
  - 0.6B: 90.13% relative spread (very high variance)
  - 1.7B: 64.56% relative spread (improved but still high)
  - 4B: 65.53% relative spread (similar to 1.7B)

### Dataset-Specific Performance
- **Best performance**: financial_news (training dataset) shows excellent results
- **Strong domain transfer**: Good performance on alpaca and fiqa
- **Most challenging**: financial_qa and twitter datasets

### Scaling Benefits
1. **0.6B → 1.7B**: Major improvement (~22% loss reduction, ~67% perplexity reduction)
2. **1.7B → 4B**: Moderate improvement (~7% loss reduction, ~21% perplexity reduction)
3. **Quick training**: Only ~3 epochs needed, indicating efficient learning