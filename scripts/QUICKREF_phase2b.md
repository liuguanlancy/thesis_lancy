# Phase 2B Quick Reference Card

## Most Common Commands

### Basic Training
```bash
# Default (all experiments, Qwen3-0.6B)
./scripts/phase2b_financial_pretraining.sh

# Mixed corpus only (recommended)
./scripts/phase2b_financial_pretraining.sh --experiments mixed

# Specific experiments
./scripts/phase2b_financial_pretraining.sh --experiments 1,2,3
```

### Mixing Strategies (for experiment 8/mixed)
```bash
# Default: 50% News cap with square root
./scripts/phase2b_financial_pretraining.sh --experiments mixed

# Pure square root scaling
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy sqrt

# Uniform (equal weights)
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy uniform

# Custom rates (must sum to 1.0)
./scripts/phase2b_financial_pretraining.sh --experiments mixed \
    --mixing-strategy custom \
    --custom-mixture-rates "0.04 0.09 0.13 0.085 0.025 0.13 0.50"
```

### Multi-Dataset Evaluation
```bash
# Train on one, eval on all
./scripts/phase2b_financial_pretraining.sh --experiments 1 --eval-all-datasets

# Traditional single eval (default)
./scripts/phase2b_financial_pretraining.sh --experiments 1 --no-eval-all-datasets
```

### Platform-Specific
```bash
# Mac/Apple Silicon (auto-detects and sets batch-size=8)
./scripts/phase2b_financial_pretraining.sh \
    --precision fp32 \
    --attn-implementation eager \
    --eval-batches 10

# RTX 4090
./scripts/phase2b_financial_pretraining.sh \
    --batch-size 128 \
    --precision bf16 \
    --use-packing

# Memory-constrained (12-16GB)
./scripts/phase2b_financial_pretraining.sh \
    --batch-size 8 \
    --use-lora --lora-rank 16 \
    --max-length 512
```

### Testing & Development
```bash
# Dry run (test config without training)
./scripts/phase2b_financial_pretraining.sh --dry-run

# Quick test (minimal steps)
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 \
    --batch-size 2 \
    --max-steps 10 \
    --eval-steps 5

# List available experiments
./scripts/phase2b_financial_pretraining.sh --list-experiments
```

## Key Options

| Option | Purpose | Default |
|--------|---------|---------|
| `--experiments` | Which to run (1-8, mixed, all) | all |
| `--mixing-strategy` | How to mix datasets | 50cap |
| `--eval-all-datasets` | Eval on all 7 datasets | false |
| `--batch-size` | Training batch size | 8 (Mac) / 256 (GPU) |
| `--max-steps` | Training steps | 4000 |
| `--use-lora` | Enable LoRA | true |
| `--precision` | bf16, fp16, fp32 | bf16 |
| `--use-packing` | Sequence packing | true |

## Dataset IDs

1. `financial_qa` - SEC filings Q&A (0.7M tokens)
2. `fingpt` - FinGPT sentiment (4.1M tokens)
3. `alpaca` - Finance Alpaca (8.5M tokens)
4. `fiqa` - FiQA Q&A (3.6M tokens)
5. `twitter` - Twitter sentiment (0.3M tokens)
6. `sec_reports` - SEC reports (8.1M tokens)
7. `news_articles` - Financial news (197M tokens)
8. `mixed` - All combined (223M tokens)

## Mixing Strategy Rates

### 50cap (Default)
```
News: 50%, Alpaca: 13%, SEC: 13%, FinGPT: 9%, FiQA: 8.5%, Q&A: 4%, Twitter: 2.5%
```

### Square Root
```
News: 56%, Alpaca: 11.6%, SEC: 11.3%, FinGPT: 8.1%, FiQA: 7.6%, Q&A: 3.3%, Twitter: 2.1%
```

### Uniform
```
All datasets: 14.3% each
```

## Hardware Quick Settings

| Hardware | Settings |
|----------|----------|
| **M1/M2 Mac** | `--batch-size 4 --precision fp32 --attn-implementation eager` |
| **RTX 3060** | `--batch-size 16 --use-lora --precision fp16` |
| **RTX 4090** | `--batch-size 128 --precision bf16` |
| **A100** | `--batch-size 256 --no-lora --precision bf16` |

## Monitoring

```bash
# TensorBoard
tensorboard --logdir ./runs/phase2b_financial_*

# Logs
tail -f ./runs/phase2b_financial_*/*/logs/training.log
```