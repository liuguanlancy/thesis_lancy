# RTX4090 Token Budget Configuration

## 100M Token Budget Per Experiment

This document explains how the RTX4090 script manages a 100M (0.1B) token budget for EACH experiment in Phase 2B financial pretraining.

## Token Consumption Formula

```
tokens_per_step = batch_size × max_length
total_tokens = tokens_per_step × num_steps
```

Where:
- **batch_size** = actual batch size per GPU
- **max_length** = maximum sequence length
- **Each experiment gets a 100M token budget**

**Note on Packing**: Sequence packing does NOT change tokens per step. It improves efficiency by replacing padding tokens with actual content, but the total token count remains the same.

## Default Configuration

### RTX4090 Defaults (Optimized for 1024+ sequence length)
- **Sequence Length**: 1024 tokens (minimum)
- **Batch Size**: 8 (optimized for 24GB VRAM)
- **Packing**: ENABLED (reduces padding waste, not token count)
- **Tokens per step**: 8 × 1024 = 8,192 tokens
- **Max steps**: 100M ÷ 8,192 ≈ 12,207 steps

### Automatic Batch Configuration

The script automatically adjusts batch size based on sequence length:

| Sequence Length | Batch Size | Tokens/Step | Max Steps |
|----------------|------------|-------------|-----------|
| 1024 | 8 | 8,192 | 12,207 |
| 1536 | 6 | 9,216 | 10,850 |
| 2048 | 4 | 8,192 | 12,207 |

## Dataset Coverage with 100M Tokens

### Small Datasets (Multiple Epochs)

| Dataset | Size | Steps@1024 | Approx Epochs |
|---------|------|------------|---------------|
| 1. Financial Q&A | 7.1K samples | 1,220 | ~0.17 |
| 5. Twitter | 1.1K samples | 1,220 | ~1.1 |
| 4. FiQA | 17.4K samples | 1,220 | ~0.07 |

### Medium Datasets (Partial Epoch)

| Dataset | Size | Steps@1024 | Approx Coverage |
|---------|------|------------|-----------------|
| 2. FinGPT Sentiment | 76.8K samples | 1,220 | ~1.6% |
| 3. Finance Alpaca | 68.9K samples | 1,220 | ~1.8% |

### Large Datasets (Token-Based Coverage)

| Dataset | Token Size | Steps@1024 | Token Coverage |
|---------|------------|------------|----------------|
| 6. SEC Reports | 80M tokens | 1,220 | 100M budget |
| 7. News Articles | 197M tokens | 1,220 | 100M budget |
| mixed | 207M tokens | 1,220 | 100M budget |

## Adjusting for Different Configurations

### Custom Batch Size (Manual Override)
```bash
# Override batch size
./scripts/phase2b_rtx4090.sh --experiments 1 --batch-size 16
# 16 × 1024 = 16,384 tokens/step → 6,103 steps for 100M tokens
```

### Longer Sequences (Better Context)
```bash
# 2048 sequence length - auto-adjusts batch to 4
./scripts/phase2b_rtx4090.sh --experiments 6 --max-length 2048
# BS=4, 4 × 2048 = 8,192 tokens/step → 12,207 steps for 100M tokens
```

### Without Packing (Less Efficient)
```bash
# Disable packing - same steps but more padding waste
./scripts/phase2b_rtx4090.sh --experiments 2 --no-packing
# 8 × 1024 = 8,192 tokens/step → 12,207 steps (but many tokens are padding)
```

## Monitoring Token Usage

The script automatically:
1. Calculates token consumption based on configuration
2. Shows percentage of 100M budget used
3. Warns if configuration deviates from 100%
4. Auto-adjusts batch size for sequence length

Example output:
```
Batch Size: 8
Max Length: 1024 tokens
Sequence Packing: true
Max Steps: 12207
Tokens per Step: ~8192
Total Token Budget: ~99999744 (99% of 100M)
```

## Best Practices

1. **Always use packing** for better efficiency (reduces padding waste)
2. **Use minimum 1024 sequence length** for financial documents
3. **Let script auto-adjust** batch size for memory optimization
4. **Monitor convergence** - model may converge before using full 100M
5. **Save checkpoints frequently** (every 100 steps) to resume if needed
6. **Use eval-on-start** to get baseline metrics

## Quick Commands

```bash
# Standard run with 100M tokens per experiment
./scripts/phase2b_rtx4090.sh --experiments 1

# Mixed corpus with 50% cap strategy
./scripts/phase2b_rtx4090.sh --experiments mixed --strategy 50cap

# Longer context for document understanding
./scripts/phase2b_rtx4090.sh --experiments 6 --max-length 2048

# Quick test (10% of budget)
./scripts/phase2b_rtx4090.sh --experiments 2 --max-steps 100

# All experiments sequentially (700M tokens total)
for i in {1..7}; do
  ./scripts/phase2b_rtx4090.sh --experiments $i
done
```

## Training Strategy Recommendations

### For Small Datasets (Twitter, FiQA, Financial Q&A)
- **100M budget allows limited epochs**
- Twitter: ~1.1 epochs (reasonable coverage)
- Financial Q&A: ~0.17 epochs (limited coverage)
- FiQA: ~0.07 epochs (very limited coverage)

### For Medium Datasets (FinGPT, Finance Alpaca)
- **100M budget gives minimal epoch**
- ~1.6-1.8% dataset coverage
- Focus on diversity over completion
- Consider early stopping if converged

### For Large Datasets (SEC Reports, News Articles)
- **100M tokens provides limited but useful training**
- Covers diverse content from large corpora
- May not see all samples but gets good representation

### For Mixed Corpus
- **100M budget with balanced mixture**
- Use 50cap or sqrt strategy to prevent large dataset dominance
- Benefits from diverse financial content

## Memory Requirements

### RTX4090 24GB VRAM Configurations

| Seq Length | Batch Size | Model Memory | Activation Memory | Safe for Training |
|------------|------------|--------------|-------------------|-------------------|
| 1024 | 4 | ~8GB | ~12GB | ✅ Yes (20GB total) |
| 1536 | 3 | ~8GB | ~14GB | ✅ Yes (22GB total) |
| 2048 | 2 | ~8GB | ~15GB | ✅ Yes (23GB total) |

## Notes

- **Each experiment uses 100M tokens** independently
- **Evaluation tokens** are not counted toward budget
- **Early stopping** recommended if loss plateaus
- **Mixed precision (BF16)** enabled for memory efficiency
- Consider using **wandb/tensorboard** to monitor token efficiency