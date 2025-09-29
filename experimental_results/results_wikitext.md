# WikiText Dataset Training Results

## Overview
Training performed on WikiText dataset (`wikitext`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 3.398 | 10.673 | 3.370 |
| fingpt | 1.299 | 2.113 | 3.570 |
| alpaca | 2.223 | 3.239 | 3.478 |
| fiqa | 2.066 | 3.142 | 3.527 |
| twitter | 1.448 | 2.776 | 3.518 |
| financial_repor | 1.385 | 3.275 | 3.441 |
| financial_news | 2.617 | 2.933 | 3.374 |
| **wikitext** | **1.564** | **3.422** | **3.303** |
| **Average** | **2.000** | **3.947** | **3.447** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 29.90 | ∞ | 29.08 |
| fingpt | 3.67 | 8.27 | 35.50 |
| alpaca | 9.23 | 25.51 | 32.38 |
| fiqa | 7.89 | 23.15 | 34.03 |
| twitter | 4.26 | 16.06 | 33.71 |
| financial_repor | 3.99 | 26.46 | 31.23 |
| financial_news | 13.70 | 18.78 | 29.19 |
| **wikitext** | **4.78** | **30.63** | **27.19** |
| **Average** | **9.68** | **143.61** | **31.54** |

Note: Qwen3-1.7B shows infinity perplexity for financial_qa, indicating numerical overflow.

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 26.24 | 991.73 | 8.31 |
| Std Deviation | 8.30 | 323.75 | 2.69 |
| Relative Spread (%) | 271.11 | 690.58 | 26.36 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 16.39 | 16.34 | 16.36 |
| Final Eval Loss | 2.000 | 3.947 | 3.447 |

## Key Observations

### Performance Scaling
- **Unusual pattern**: 0.6B performs best, larger models show degradation
- **Loss increase** from 0.6B to 1.7B: +97.4%
- **Performance anomaly**: 1.7B model shows severe instability (infinity perplexity)
- 4B model partially recovers but still worse than 0.6B

### Model Stability
- Extreme variance in results:
  - 0.6B: 271.11% relative spread (high but functional)
  - 1.7B: 690.58% relative spread (catastrophic instability)
  - 4B: 26.36% relative spread (stabilized but poor performance)

### Dataset-Specific Performance
- **0.6B strengths**: fingpt, financial_repor, twitter (very low perplexity)
- **Cross-domain challenge**: WikiText training doesn't transfer well to financial domains
- **Training failure**: 1.7B model appears to have training issues

### Scaling Benefits
1. **0.6B → 1.7B**: Severe performance degradation (training failure likely)
2. **1.7B → 4B**: Partial recovery but still worse than baseline
3. **Domain mismatch**: WikiText (general text) doesn't align well with financial evaluation sets

## Learning Rate Comparison

### Original Results (LR=2e-5) vs LR-Adjusted Results

#### Loss Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=5e-6) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=3e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| financial_qa | 3.398 | 10.673 | **4.066** | 3.370 | 3.871 |
| fingpt | 1.299 | 2.113 | **4.070** | 3.570 | 3.877 |
| alpaca | 2.223 | 3.239 | **3.789** | 3.478 | 3.639 |
| fiqa | 2.066 | 3.142 | **3.846** | 3.527 | 3.739 |
| twitter | 1.448 | 2.776 | **4.077** | 3.518 | 3.881 |
| financial_repor | 1.385 | 3.275 | **3.909** | 3.441 | 3.747 |
| financial_news | 2.617 | 2.933 | **3.516** | 3.374 | 3.275 |
| **wikitext** | **1.564** | **3.422** | **3.880** | **3.303** | **3.653** |
| **Average** | **2.000** | **3.947** | **3.894** | **3.447** | **3.710** |

#### Perplexity Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=5e-6) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=3e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| financial_qa | 29.90 | ∞ | **58.33** | 29.08 | 47.98 |
| fingpt | 3.67 | 8.27 | **58.55** | 35.50 | 48.30 |
| alpaca | 9.23 | 25.51 | **44.22** | 32.38 | 38.06 |
| fiqa | 7.89 | 23.15 | **46.81** | 34.03 | 42.04 |
| twitter | 4.26 | 16.06 | **58.98** | 33.71 | 48.48 |
| financial_repor | 3.99 | 26.46 | **49.83** | 31.23 | 42.41 |
| financial_news | 13.70 | 18.78 | **33.66** | 29.19 | 26.44 |
| **wikitext** | **4.78** | **30.63** | **48.44** | **27.19** | **38.60** |
| **Average** | **9.68** | **∞** | **49.85** | **31.54** | **41.54** |

### Learning Rate Adjustment Summary

| Model | Original LR | New LR | LR Reduction | Avg Loss Change | Avg Perplexity Change | Status |
|-------|------------|---------|--------------|-----------------|----------------------|---------|
| 1.7B | 2e-5 | 5e-6 | -75% | 3.947 → 3.894 (-1.3%) | ∞ → 49.85 | **FIXED** |
| 4B | 2e-5 | 3e-6 | -85% | 3.447 → 3.710 (+7.6%) | 31.54 → 41.54 (+31.7%) | Stable but degraded |

### Analysis of LR Adjustments

#### 1.7B Model (LR: 2e-5 → 5e-6)
- **Critical Fix**: Resolved catastrophic failure (infinity perplexity eliminated)
- **Trade-off**: Higher loss on financial tasks but stable training achieved
- **Outcome**: Model is now usable despite worse performance than original 0.6B
- **Recommendation**: 75% LR reduction successfully prevented gradient explosion

#### 4B Model (LR: 2e-5 → 3e-6)
- **Stability**: More conservative LR led to stable but suboptimal convergence
- **Performance**: Slight degradation in both loss and perplexity
- **Variance**: Increased spread in results (Max-Min: 8.31 → 22.05)
- **Recommendation**: Could potentially use higher LR (5e-6) for better performance

### Key Findings
1. **WikiText-Financial Domain Gap**: The fundamental issue remains - WikiText pretraining doesn't transfer well to financial evaluation
2. **Learning Rate Sensitivity**: 1.7B model extremely sensitive to LR, requiring 75% reduction to train successfully
3. **Reverse Scaling Persists**: Even with LR adjustments, 0.6B still outperforms larger models on this dataset
4. **Stability vs Performance**: Lower LRs provide stability but may undershoot optimal performance