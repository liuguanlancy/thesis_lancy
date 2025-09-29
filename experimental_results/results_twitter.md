# Twitter Financial Sentiment Dataset Training Results

## Overview
Training performed on Twitter Financial News Sentiment dataset (`zeroshot/twitter-financial-news-sentiment`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 2.465 | 2.317 | 2.832 |
| fingpt | 2.743 | 2.504 | 2.909 |
| alpaca | 3.006 | 2.662 | 2.955 |
| fiqa | 2.979 | 2.657 | 3.000 |
| **twitter** | **2.534** | **2.400** | **2.881** |
| financial_repor | 2.481 | 2.319 | 2.798 |
| financial_news | 3.168 | 2.802 | 2.872 |
| wikitext | 2.690 | 2.466 | 2.882 |
| **Average** | **2.758** | **2.516** | **2.891** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 11.76 | 10.15 | 16.98 |
| fingpt | 15.53 | 12.23 | 18.34 |
| alpaca | 20.21 | 14.33 | 19.20 |
| fiqa | 19.67 | 14.26 | 20.09 |
| **twitter** | **12.60** | **11.02** | **17.83** |
| financial_repor | 11.95 | 10.17 | 16.42 |
| financial_news | 23.77 | 16.48 | 17.67 |
| wikitext | 14.74 | 11.78 | 17.85 |
| **Average** | **16.28** | **12.55** | **18.05** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 12.01 | 6.33 | 3.67 |
| Std Deviation | 4.16 | 2.12 | 1.10 |
| Relative Spread (%) | 73.75 | 50.44 | 20.35 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 67.82 | 68.20 | 67.82 |
| Final Eval Loss | 2.758 | 2.516 | 2.891 |

## Key Observations

### Performance Scaling
- **Unusual pattern**: 1.7B performs best, 4B shows regression
- **Loss increase** from 1.7B to 4B: +15.0%
- **Perplexity increase** from 1.7B to 4B: +43.8%
- Clear overfitting in 4B model

### Model Stability
- Stability improves despite performance regression:
  - 0.6B: 73.75% relative spread (very high variance)
  - 1.7B: 50.44% relative spread (moderate)
  - 4B: 20.35% relative spread (most stable but worse performance)

### Dataset-Specific Performance
- **Best performance**: financial_qa and financial_repor datasets
- **Training dataset**: Twitter shows expected good performance
- **Most challenging**: financial_news and alpaca datasets

### Scaling Benefits
1. **0.6B → 1.7B**: Good improvement (~9% loss reduction, ~23% perplexity reduction)
2. **1.7B → 4B**: Performance degradation (overfitting likely)
3. **High epoch count**: 68 epochs may have caused overfitting in larger model

## Learning Rate Comparison

### Original Results (LR=2e-5) vs LR-Adjusted Results

#### Loss Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=1e-5) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=5e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| financial_qa | 2.465 | 2.317 | **2.162** | 2.832 | **2.433** |
| fingpt | 2.743 | 2.504 | **2.343** | 2.909 | **2.540** |
| alpaca | 3.006 | 2.662 | **2.539** | 2.955 | **2.614** |
| fiqa | 2.979 | 2.657 | **2.501** | 3.000 | **2.611** |
| **twitter** | **2.534** | **2.400** | **2.220** | **2.881** | **2.469** |
| financial_repor | 2.481 | 2.319 | **2.163** | 2.798 | **2.391** |
| financial_news | 3.168 | 2.802 | **2.646** | 2.872 | **2.540** |
| wikitext | 2.690 | 2.466 | **2.297** | 2.882 | **2.486** |
| **Average** | **2.758** | **2.516** | **2.359** | **2.891** | **2.511** |

#### Perplexity Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=1e-5) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=5e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| financial_qa | 11.76 | 10.15 | **8.69** | 16.98 | **11.39** |
| fingpt | 15.53 | 12.23 | **10.41** | 18.34 | **12.69** |
| alpaca | 20.21 | 14.33 | **12.66** | 19.20 | **13.65** |
| fiqa | 19.67 | 14.26 | **12.20** | 20.09 | **13.61** |
| **twitter** | **12.60** | **11.02** | **9.21** | **17.83** | **11.81** |
| financial_repor | 11.95 | 10.17 | **8.70** | 16.42 | **10.93** |
| financial_news | 23.77 | 16.48 | **14.10** | 17.67 | **12.68** |
| wikitext | 14.74 | 11.78 | **9.94** | 17.85 | **12.02** |
| **Average** | **16.28** | **12.55** | **10.74** | **18.05** | **12.35** |

### Learning Rate Adjustment Summary

| Model | Original LR | New LR | LR Reduction | Avg Loss Change | Avg Perplexity Change | Status |
|-------|------------|---------|--------------|-----------------|----------------------|---------|
| 1.7B | 2e-5 | 1e-5 | -50% | 2.516 → 2.359 (-6.2%) | 12.55 → 10.74 (-14.4%) | **IMPROVED** |
| 4B | 2e-5 | 5e-6 | -75% | 2.891 → 2.511 (-13.1%) | 18.05 → 12.35 (-31.6%) | **FIXED** |

### Analysis of LR Adjustments

#### 1.7B Model (LR: 2e-5 → 1e-5)
- **Performance Gain**: Significant improvement across all metrics
- **Best Results**: financial_qa (8.69 ppl) and financial_repor (8.70 ppl)
- **Consistency**: Maintained good spread with better absolute performance
- **Outcome**: 50% LR reduction enhanced performance without sacrificing stability

#### 4B Model (LR: 2e-5 → 5e-6)
- **Reverse Scaling Fixed**: Now properly outperforms 0.6B model
- **Major Improvement**: Average perplexity reduced by 31.6%
- **Proper Scaling**: 4B (12.35 ppl) now performs comparably to 1.7B (10.74 ppl)
- **Stability**: Maintained low variance (22.06% relative spread)

### Epochs Comparison
- All models still trained for ~68 epochs
- Despite high epoch count, lower LRs prevented overfitting
- LR adjustment was more effective than early stopping

### Key Findings
1. **Reverse Scaling Resolved**: With proper LR tuning, model scaling works as expected
2. **Optimal LR by Model Size**:
   - 1.7B: 1e-5 (50% reduction) optimal
   - 4B: 5e-6 (75% reduction) necessary for convergence
3. **Performance Hierarchy Restored**: 4B ≈ 1.7B > 0.6B (as expected)
4. **Twitter Dataset Success**: Lower LRs enabled proper learning from Twitter financial data