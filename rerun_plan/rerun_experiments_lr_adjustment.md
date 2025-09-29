# Learning Rate Adjustment Experiments

## Overview
Rerun experiments for Twitter, Financial QA, and WikiText datasets with adjusted learning rates to address reverse scaling and overfitting issues observed in initial experiments.

## Experiment Configuration Table (Grouped by Dataset)

### Financial QA (Experiment ID: 1)
| Model | Original LR | New LR | Change | Priority | Issue to Fix |
|-------|------------|---------|---------|----------|--------------|
| 1.7B | 2e-5 | **1e-5** | -50% | **4** | Maintain performance, prevent overfitting |
| 4B | 2e-5 | **5e-6** | -75% | **2** | Fix severe overfitting (249 epochs!) |

### Twitter Financial (Experiment ID: 5)
| Model | Original LR | New LR | Change | Priority | Issue to Fix |
|-------|------------|---------|---------|----------|--------------|
| 1.7B | 2e-5 | **1e-5** | -50% | **5** | Optimize mid-size performance |
| 4B | 2e-5 | **5e-6** | -75% | **3** | Fix 4B regression / overfitting |

### WikiText (Experiment ID: 8)
| Model | Original LR | New LR | Change | Priority | Issue to Fix |
|-------|------------|---------|---------|----------|--------------|
| 1.7B | 2e-5 | **5e-6** | -75% | **1** | **Fix infinity perplexity / catastrophic failure** |
| 4B | 2e-5 | **3e-6** | -85% | **6** | Prevent instability |

## Script Execution Commands

### Using the Created Scripts

The experiments are implemented as shell scripts that call `phase2b_financial_pretraining.sh` directly:

#### Financial QA Experiments
```bash
./scripts/lr_adjust_financial_qa_experiments.sh
```
- Runs 1.7B model with LR=1e-5
- Then runs 4B model with LR=5e-6
- 5-minute interval between experiments

#### Twitter Financial Experiments
```bash
./scripts/lr_adjust_twitter_experiments.sh
```
- Runs 1.7B model with LR=1e-5
- Then runs 4B model with LR=5e-6
- 5-minute interval between experiments

#### WikiText Experiments (Priority - Fix Catastrophic Failure)
```bash
./scripts/lr_adjust_wikitext_experiments.sh
```
- Runs 1.7B model with LR=5e-6 (75% reduction for stability)
- Then runs 4B model with LR=3e-6 (85% reduction)
- 5-minute interval between experiments

### Key Parameters Used
All scripts use the same settings as original experiments:
- **Batch sizes**: 1.7B uses 4, 4B uses 2
- **Gradient accumulation**: 1.7B uses 2, 4B uses 4
- **Eval/Save steps**: 1000
- **Save total limit**: 2
- **Warmup**: Automatic (10% of max steps)
- **Token budget**: 100M per experiment


## Expected Outcomes

### WikiText
- **1.7B**: Fix infinity perplexity, achieve stable training
- **4B**: Maintain or improve upon current performance (3.447 loss)

### Financial QA
- **1.7B**: Maintain good performance (2.128 loss) with better generalization
- **4B**: Improve from 2.196 loss, reduce overfitting

### Twitter
- **1.7B**: Maintain good performance (2.516 loss)
- **4B**: Improve from 2.891 loss, should now outperform 1.7B

## Monitoring Metrics

Track these metrics during training:
1. **Eval loss convergence** - should be smoother
2. **Gradient norm** - should stay below 1.0
3. **Learning rate schedule** - verify warmup is working
4. **Early stopping** - should trigger before max epochs for overfitting cases
5. **Perplexity stability** - no infinity values

## Success Criteria

| Dataset | Model | Current Loss | Target Loss | Current Perplexity | Target Perplexity |
|---------|-------|--------------|-------------|-------------------|-------------------|
| WikiText | 1.7B | 3.947 | < 3.5 | ∞ | < 50 |
| WikiText | 4B | 3.447 | < 3.2 | 31.54 | < 25 |
| Financial QA | 1.7B | 2.128 | ~2.1 | 8.42 | ~8.4 |
| Financial QA | 4B | 2.196 | < 2.1 | 9.02 | < 8.5 |
| Twitter | 1.7B | 2.516 | ~2.5 | 12.55 | ~12.5 |
| Twitter | 4B | 2.891 | < 2.5 | 18.05 | < 12.5 |

## Execution Notes

### Running Order (by Priority)
1. **WikiText 1.7B** - Most critical (infinity perplexity fix)
2. **Financial QA 4B** - Severe overfitting (249 epochs)
3. **Twitter 4B** - Performance regression
4. **Financial QA 1.7B** - Performance maintenance
5. **Twitter 1.7B** - Performance optimization
6. **WikiText 4B** - Stability improvement

### Important Considerations
- **Backup original results** before running (same output directories)
- Scripts call `phase2b_financial_pretraining.sh` directly (not via phase2b_rtx4090.sh wrapper)
- All experiments use exact same settings as originals, only LR is changed
- Results will overwrite original experiment directories in `runs/`
- Monitor first few eval steps to ensure stable training (especially WikiText 1.7B)