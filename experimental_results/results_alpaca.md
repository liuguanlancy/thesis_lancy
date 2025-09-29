# Alpaca Dataset Training Results

## Overview
Training performed on Finance Alpaca dataset (`gbharti/finance-alpaca`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 4.766 | 2.950 | 2.148 |
| fingpt | 4.715 | 2.988 | 2.217 |
| **alpaca** | **4.155** | **2.748** | **2.106** |
| fiqa | 4.292 | 2.870 | 2.221 |
| twitter | 4.777 | 2.987 | 2.194 |
| financial_repor | 4.539 | 2.849 | 2.110 |
| financial_news | 3.920 | 2.712 | 2.149 |
| wikitext | 4.629 | 2.936 | 2.183 |
| **Average** | **4.474** | **2.880** | **2.166** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| financial_qa | 117.40 | 19.11 | 8.56 |
| fingpt | 111.65 | 19.85 | 9.18 |
| **alpaca** | **63.73** | **15.61** | **8.22** |
| fiqa | 73.12 | 17.63 | 9.22 |
| twitter | 118.74 | 19.82 | 8.97 |
| financial_repor | 93.56 | 17.26 | 8.25 |
| financial_news | 50.40 | 15.05 | 8.58 |
| wikitext | 102.41 | 18.85 | 8.88 |
| **Average** | **91.37** | **17.90** | **8.73** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 68.34 | 4.79 | 1.00 |
| Std Deviation | 24.34 | 1.72 | 0.37 |
| Relative Spread (%) | 74.79 | 26.79 | 11.51 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 12.26 | 12.26 | 12.24 |
| Final Eval Loss | 4.474 | 2.880 | 2.166 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 4B: **51.6%**
- **Perplexity reduction** from 0.6B to 4B: **90.4%**
- Best performance on the training dataset (alpaca) across all model sizes

### Model Stability
- Larger models show significantly better consistency across datasets:
  - 0.6B: 74.79% relative spread (high variance)
  - 1.7B: 26.79% relative spread (moderate variance)
  - 4B: 11.51% relative spread (low variance, most stable)

### Dataset-Specific Performance
- **Best generalization**: financial_news dataset (lowest perplexity for 0.6B)
- **Most challenging**: twitter and financial_qa datasets (highest perplexities)
- **Training dataset advantage**: Alpaca consistently shows better metrics than average

### Scaling Benefits
1. **0.6B → 1.7B**: Massive perplexity drop (~80%), significant loss improvement (~35%)
2. **1.7B → 4B**: Further perplexity reduction (~51%), moderate loss improvement (~25%)
3. **Diminishing returns**: Larger jumps in performance from 0.6B to 1.7B than from 1.7B to 4B