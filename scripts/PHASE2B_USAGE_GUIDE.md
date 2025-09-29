# Phase 2B Training Scripts Usage Guide

This guide covers the usage of `phase2b_m1max.sh` and `phase2b_rtx4090.sh` scripts for running Phase 2B financial pretraining experiments with sequential execution support.

## Overview

Both scripts support:
- **Single experiment execution**: Run one experiment at a time
- **Sequential execution**: Run multiple experiments automatically with 5-minute breaks
- **100M token budget**: Each experiment uses 100M tokens
- **Device-optimized settings**: Automatic configuration for M1 Max or RTX 4090

## Quick Start

### Single Experiment
```bash
# M1 Max
./scripts/phase2b_m1max.sh --experiments 1

# RTX 4090
./scripts/phase2b_rtx4090.sh --experiments 1
```

### Multiple Experiments (Sequential)
```bash
# M1 Max: Run experiments 1, 3, and mixed sequentially
./scripts/phase2b_m1max.sh --experiments "1 3 mixed" --save-steps 500 --eval-steps 500

# RTX 4090: Run all individual datasets then mixed
./scripts/phase2b_rtx4090.sh --experiments "1 2 3 4 5 6 7 mixed"
```

## Experiment Numbers

| Exp # | Dataset | Samples | Description |
|-------|---------|---------|-------------|
| 1 | Financial Q&A | 7.1K | Question-answering pairs |
| 2 | FinGPT Sentiment | 76.8K | Financial sentiment analysis |
| 3 | Finance Alpaca | 68.9K | Instruction-following dataset |
| 4 | FiQA | 17.4K | Financial Q&A dataset |
| 5 | Twitter | 1.1K | Financial tweets (will repeat ~88-177x) |
| 6 | SEC Reports | 54.3K | SEC filings (80M tokens) |
| 7 | News Articles | 300K | Financial news (197M tokens) |
| mixed | All datasets | - | Combined with configurable strategy |

## Sequential Execution Features

### How It Works
1. First experiment runs to completion
2. 5-minute break (shows next start time)
3. Next experiment starts automatically
4. Process repeats until all complete

### Example Timeline
```bash
./scripts/phase2b_rtx4090.sh --experiments "1 3 mixed"
```
- Hour 0-7: Experiment 1 (Financial Q&A)
- 5 min break
- Hour 7-14: Experiment 3 (Finance Alpaca)
- 5 min break
- Hour 14-21: Mixed experiment

### Order Matters
Experiments run in the exact order specified:
```bash
# This runs: mixed → 3 → 1
./scripts/phase2b_m1max.sh --experiments "mixed 3 1"
```

## Common Usage Patterns

### 1. Quick Test Run
```bash
# Test with minimal steps
./scripts/phase2b_m1max.sh --experiments 1 --max-steps 100 --dry-run
```

### 2. Overnight Training
```bash
# Run 3 experiments overnight with 500-step checkpoints
./scripts/phase2b_rtx4090.sh --experiments "1 3 mixed" \
    --save-steps 500 \
    --eval-steps 500
```

### 3. Full Suite (RTX 4090)
```bash
# Run all experiments (will take ~35-49 hours on A100)
./scripts/phase2b_rtx4090.sh --experiments "1 2 3 4 5 6 7 mixed"
```

### 4. Memory-Optimized (M1 Max)
```bash
# Shorter sequences for M1 Max memory constraints
./scripts/phase2b_m1max.sh --experiments "1 2" \
    --max-length 256 \
    --batch-size 16
```

### 5. Custom Mixing Strategy
```bash
# Different mixing strategies for combined dataset
./scripts/phase2b_rtx4090.sh --experiments mixed --strategy sqrt
./scripts/phase2b_rtx4090.sh --experiments mixed --strategy proportional
./scripts/phase2b_rtx4090.sh --experiments mixed --strategy uniform
```

## Configuration Options

### Required
- `--experiments <exp>`: Single (1-7, mixed) or multiple ("1 3 mixed")

### Optional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | 50cap | Mixing strategy (50cap/sqrt/proportional/uniform) |
| `--max-steps` | Auto | Override calculated steps (normally ~12-24K) |
| `--batch-size` | 8 | Training batch size |
| `--max-length` | 512/1024 | Sequence length (M1:512, RTX:1024) |
| `--eval-steps` | 100-200 | Evaluation frequency |
| `--save-steps` | 100-200 | Checkpoint frequency |
| `--eval-batches` | 50-100 | Batches per evaluation |
| `--warmup-steps` | 10% | Warmup period |
| `--no-packing` | - | Disable sequence packing |
| `--no-eval-on-start` | - | Skip initial evaluation |
| `--dry-run` | - | Preview commands without executing |
| `--local` | - | Use locally cached datasets |

## Device-Specific Settings

### M1 Max (32GB)
- **Default sequence length**: 512 tokens
- **Batch size**: 8 (auto-adjusts based on sequence length)
- **Precision**: FP32 (for stability)
- **Attention**: Eager (avoids MPS NaN issues)
- **Throughput**: ~450 tokens/sec

### RTX 4090 (24GB)
- **Default sequence length**: 1024 tokens
- **Batch size**: 8 (auto-adjusts)
- **Precision**: BF16
- **Attention**: Flash Attention 2
- **Throughput**: ~2000 tokens/sec

## Resource Estimates

### Token Budget (100M per experiment)
| Config | Batch | Seq Len | Steps | Time (RTX) | Time (M1) |
|--------|-------|---------|-------|------------|-----------|
| Default | 8 | 1024 | 12,207 | ~14 hrs | - |
| Default | 8 | 512 | 24,414 | - | ~60 hrs |
| Memory-opt | 4 | 2048 | 12,207 | ~14 hrs | - |
| Speed-opt | 16 | 256 | 24,414 | ~10 hrs | ~50 hrs |

### Storage Requirements
- **Checkpoint size**: ~1.2GB (BF16) or ~2.5GB (FP32)
- **With 500-step saves**: ~24 checkpoints = ~30-60GB per experiment
- **With 100-step saves**: ~122 checkpoints = ~150-300GB per experiment

## Monitoring Progress

### During Training
```bash
# Watch training logs
tail -f runs/*/logs/training.log

# Monitor GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Monitor system resources (M1 Max)
sudo powermetrics --samplers gpu_power -i 1000
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 4

# Reduce sequence length
--max-length 256

# Enable gradient checkpointing (in underlying script)
```

### Training Too Slow
```bash
# Increase batch size if memory allows
--batch-size 16

# Reduce evaluation frequency
--eval-steps 1000 --save-steps 1000

# Disable multi-dataset evaluation
--no-eval-all
```

### Interrupted Training
```bash
# Resume from latest checkpoint
# Check runs/ directory for latest checkpoint
# Manually restart with --resume_from_checkpoint in underlying script
```

## Advanced Usage

### Dry Run First
Always test with dry-run before long training:
```bash
./scripts/phase2b_rtx4090.sh --experiments "1 2 3" --dry-run
```

### Custom Token Budget
Override the 100M token budget:
```bash
# Use only 10M tokens (quick test)
./scripts/phase2b_m1max.sh --experiments 1 --max-steps 2441

# Use 200M tokens (double budget)
./scripts/phase2b_rtx4090.sh --experiments 1 --max-steps 24414
```

### Chaining Strategies
Test different mixing strategies sequentially:
```bash
./scripts/phase2b_rtx4090.sh --experiments "mixed mixed mixed" \
    --strategy "50cap sqrt proportional"  # Note: would need script modification
```

## Example Workflows

### 1. Development Testing
```bash
# Quick 1000-step test on each dataset
for exp in 1 2 3 4 5 6 7; do
    ./scripts/phase2b_m1max.sh --experiments $exp \
        --max-steps 1000 \
        --eval-steps 100 \
        --save-steps 500
done
```

### 2. Production Training
```bash
# Full training with optimal settings
./scripts/phase2b_rtx4090.sh \
    --experiments "1 2 3 4 5 6 7 mixed" \
    --save-steps 500 \
    --eval-steps 500 \
    --eval-batches 100
```

### 3. Comparison Study
```bash
# Compare different sequence lengths
./scripts/phase2b_m1max.sh --experiments 1 --max-length 256
./scripts/phase2b_m1max.sh --experiments 1 --max-length 512  
./scripts/phase2b_m1max.sh --experiments 1 --max-length 1024
```

## Tips and Best Practices

1. **Start with dry-run**: Always verify commands before executing
2. **Monitor first checkpoint**: Ensure training is stable before leaving unattended
3. **Use screen/tmux**: For remote sessions that may disconnect
4. **Save frequently early**: Use smaller save_steps initially, increase later
5. **Document experiments**: Keep notes on configurations and results
6. **Check disk space**: Ensure sufficient space for checkpoints
7. **Gradual scaling**: Start with single experiments before running all

## Script Locations

- Main scripts: `scripts/phase2b_m1max.sh`, `scripts/phase2b_rtx4090.sh`
- Underlying script: `scripts/phase2b_financial_pretraining.sh`
- Test script: `test_sequential_m1max.sh`
- Logs: `runs/*/logs/`
- Checkpoints: `runs/*/checkpoints/`
- TensorBoard: `runs/*/tensorboard/`

## Support

For issues or questions:
1. Check the dry-run output first
2. Verify dataset availability
3. Ensure sufficient memory/disk space
4. Check the underlying script logs
5. Test with minimal configuration

Remember: Each experiment uses 100M tokens and takes 5-60 hours depending on hardware!