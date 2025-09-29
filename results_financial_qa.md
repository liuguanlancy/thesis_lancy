# Financial QA Dataset Training Results

## Overview
Training performed on Financial QA 10K dataset (`virattt/financial-qa-10K`) with evaluation across 8 different financial and general datasets.

## Loss Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| **financial_qa** | **2.115** | **2.006** | **2.115** |
| fingpt | 2.306 | 2.154 | 2.235 |
| alpaca | 2.381 | 2.231 | 2.293 |
| fiqa | 2.399 | 2.246 | 2.307 |
| twitter | 2.213 | 2.101 | 2.196 |
| financial_repor | 2.106 | 2.002 | 2.110 |
| financial_news | 2.361 | 2.172 | 2.130 |
| wikitext | 2.241 | 2.108 | 2.186 |
| **Average** | **2.265** | **2.128** | **2.196** |

## Perplexity Metrics

| Eval Dataset | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------------|------------|------------|----------|
| **financial_qa** | **8.29** | **7.44** | **8.29** |
| fingpt | 10.04 | 8.62 | 9.34 |
| alpaca | 10.82 | 9.31 | 9.91 |
| fiqa | 11.02 | 9.45 | 10.05 |
| twitter | 9.14 | 8.18 | 8.99 |
| financial_repor | 8.21 | 7.40 | 8.25 |
| financial_news | 10.60 | 8.78 | 8.41 |
| wikitext | 9.41 | 8.23 | 8.89 |
| **Average** | **9.69** | **8.42** | **9.02** |

## Model Consistency Metrics

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Max-Min Difference | 2.80 | 2.05 | 1.80 |
| Std Deviation | 1.03 | 0.72 | 0.66 |
| Relative Spread (%) | 28.92 | 24.29 | 19.92 |

## Training Details

| Metric | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|--------|------------|------------|----------|
| Epochs | 249.12 | 249.12 | 249.12 |
| Final Eval Loss | 2.265 | 2.128 | 2.196 |

## Key Observations

### Performance Scaling
- **Loss reduction** from 0.6B to 1.7B: **6.0%**, but slight increase to 4B
- **Perplexity reduction** from 0.6B to 1.7B: **13.1%**, but slight increase to 4B
- Unusual pattern: 1.7B performs best, 4B shows slight regression

### Model Stability
- All models show relatively good consistency:
  - 0.6B: 28.92% relative spread
  - 1.7B: 24.29% relative spread (best stability)
  - 4B: 19.92% relative spread (most stable)

### Dataset-Specific Performance
- **Best performance**: financial_repor and financial_qa (training dataset)
- **Most challenging**: fiqa and alpaca datasets
- **Notable**: Very high training epochs (249) indicate extensive fine-tuning

### Scaling Benefits
1. **0.6B → 1.7B**: Consistent improvements across all metrics
2. **1.7B → 4B**: Mixed results - stability improves but performance slightly degrades
3. **Potential overfitting**: 4B model may be overfitted given the high epoch count

## Learning Rate Comparison

### Original Results (LR=2e-5) vs LR-Adjusted Results

#### Loss Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=1e-5) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=5e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| **financial_qa** | **2.115** | **2.006** | **2.115** | **2.115** | **2.006** |
| fingpt | 2.306 | 2.154 | 2.252 | 2.235 | 2.109 |
| alpaca | 2.381 | 2.231 | 2.295 | 2.293 | 2.184 |
| fiqa | 2.399 | 2.246 | 2.313 | 2.307 | 2.189 |
| twitter | 2.213 | 2.101 | 2.208 | 2.196 | 2.086 |
| financial_repor | 2.106 | 2.002 | 2.103 | 2.110 | 2.006 |
| financial_news | 2.361 | 2.172 | 2.225 | 2.130 | 2.043 |
| wikitext | 2.241 | 2.108 | 2.206 | 2.186 | 2.080 |
| **Average** | **2.265** | **2.128** | **2.215** | **2.196** | **2.088** |

#### Perplexity Metrics Comparison

| Eval Dataset | Qwen3-0.6B (LR=2e-5) | Qwen3-1.7B (LR=2e-5) | Qwen3-1.7B (LR=1e-5) | Qwen3-4B (LR=2e-5) | Qwen3-4B (LR=5e-6) |
|--------------|---------------------|---------------------|---------------------|-------------------|-------------------|
| **financial_qa** | **8.29** | **7.44** | **8.29** | **8.29** | **7.43** |
| fingpt | 10.04 | 8.62 | 9.51 | 9.34 | 8.24 |
| alpaca | 10.82 | 9.31 | 9.92 | 9.91 | 8.88 |
| fiqa | 11.02 | 9.45 | 10.10 | 10.05 | 8.93 |
| twitter | 9.14 | 8.18 | 9.10 | 8.99 | 8.05 |
| financial_repor | 8.21 | 7.40 | 8.19 | 8.25 | 7.43 |
| financial_news | 10.60 | 8.78 | 9.25 | 8.41 | 7.71 |
| wikitext | 9.41 | 8.23 | 9.08 | 8.89 | 8.00 |
| **Average** | **9.69** | **8.42** | **9.18** | **9.02** | **8.09** |

### Learning Rate Adjustment Summary

| Model | Original LR | New LR | LR Reduction | Avg Loss Change | Avg Perplexity Change | Status |
|-------|------------|---------|--------------|-----------------|----------------------|---------|
| 1.7B | 2e-5 | 1e-5 | -50% | 2.128 → 2.215 (+4.1%) | 8.42 → 9.18 (+9.0%) | Slightly degraded |
| 4B | 2e-5 | 5e-6 | -75% | 2.196 → 2.088 (-4.9%) | 9.02 → 8.09 (-10.3%) | **IMPROVED** |

### Analysis of LR Adjustments

#### 1.7B Model (LR: 2e-5 → 1e-5)
- **Slight Degradation**: Performance slightly worse with reduced LR
- **Stability Trade-off**: Lower LR may have prevented optimal convergence
- **Still Good**: Results remain competitive (9.18 avg perplexity)
- **Epochs Impact**: 249 epochs with lower LR may have underfit slightly

#### 4B Model (LR: 2e-5 → 5e-6)
- **Clear Improvement**: Best overall performance achieved
- **Proper Scaling**: Now outperforms both smaller models as expected
- **Reduced Overfitting**: Lower LR helped generalization despite 249 epochs
- **Best Results**: Achieved lowest average perplexity (8.09)

### Epochs Analysis
- All models still trained for 249.12 epochs
- Original high LR (2e-5) caused overfitting in 4B model
- Lower LRs helped, especially for 4B model:
  - 1.7B: May have benefited from slightly higher LR (e.g., 1.5e-5)
  - 4B: 5e-6 was optimal for this dataset

### Key Findings
1. **4B Model Fixed**: Reverse scaling resolved with 75% LR reduction
2. **Dataset Characteristics**: Financial QA requires careful LR tuning due to small dataset size (7K examples)
3. **Optimal LR by Model**:
   - 1.7B: Original 2e-5 was actually better
   - 4B: 5e-6 optimal (75% reduction necessary)
4. **Performance Hierarchy**: With LR adjustments: 4B (8.09) > 1.7B original (8.42) > 1.7B adjusted (9.18) > 0.6B (9.69)