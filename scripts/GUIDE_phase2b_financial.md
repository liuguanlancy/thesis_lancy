# Phase 2B Financial Pretraining: Comprehensive Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Hardware Requirements](#hardware-requirements)
4. [Dataset Information](#dataset-information)
5. [Mixing Strategies](#mixing-strategies)
6. [Dataset Mixture Sampling Implementation](#dataset-mixture-sampling-implementation)
7. [Multi-Dataset Evaluation](#multi-dataset-evaluation)
8. [Complete Options Reference](#complete-options-reference)
9. [Platform-Specific Optimizations](#platform-specific-optimizations)
10. [Example Commands Gallery](#example-commands-gallery)
11. [Performance Tuning](#performance-tuning)
12. [Integration with Pipeline](#integration-with-pipeline)
13. [Troubleshooting](#troubleshooting)

## Overview

The `phase2b_financial_pretraining.sh` script performs domain-specific pretraining on financial datasets. It adapts any HuggingFace language model to the financial domain through continued pretraining on seven curated financial corpora.

### Key Features
- **Model-agnostic**: Works with any HuggingFace causal language model
- **Flexible training**: Supports both LoRA and full fine-tuning
- **7 financial datasets**: Plus mixed training with multiple strategies
- **Dataset mixing strategies**: 50% cap, square root, proportional, uniform, or custom
- **Multi-dataset evaluation**: Evaluate on all datasets even when training on one
- **Memory efficient**: Auto-detects hardware and optimizes settings
- **Sequence packing**: ~2.5x speedup for short sequences
- **Platform optimized**: Special handling for Apple Silicon/MPS

### Training Pipeline Context
```
Phase 1: Base Model Selection
    ↓
Phase 2B: Financial Domain Pretraining (THIS SCRIPT)
    ↓
Phase 2C: Analytical Capabilities
    ↓
Phase 3: Task-Specific Fine-tuning
    ↓
Deployment
```

## Quick Start

### Basic Usage (All Defaults)
```bash
# Run all experiments with Qwen3-0.6B model
./scripts/phase2b_financial_pretraining.sh
```

### Common Configurations
```bash
# Use a different model
./scripts/phase2b_financial_pretraining.sh --model meta-llama/Llama-3.2-1B

# Run specific experiments
./scripts/phase2b_financial_pretraining.sh --experiments 1,2,3

# Mixed corpus with default 50% News cap strategy
./scripts/phase2b_financial_pretraining.sh --experiments mixed

# Mixed corpus with different strategy
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy sqrt

# Memory-constrained setup
./scripts/phase2b_financial_pretraining.sh --batch-size 8 --use-lora --lora-rank 16

# Test configuration without running
./scripts/phase2b_financial_pretraining.sh --dry-run
```

## Hardware Requirements

### GPU Memory Requirements

| Configuration | GPU Memory | Recommended Settings |
|--------------|------------|---------------------|
| Minimum | 12GB | `--batch-size 8 --use-lora --lora-rank 16` |
| Standard | 16-24GB | `--batch-size 32 --use-lora` |
| Optimal | 40GB+ | `--batch-size 128` (default) |
| High-end | 80GB | `--batch-size 256 --no-lora` |

### Apple Silicon Memory Requirements

| Configuration | Memory | Auto-Detected Settings |
|--------------|--------|------------------------|
| M1/M2 (8-16GB) | 8-16GB | `--batch-size 8` (automatic) |
| M1/M2 Pro/Max | 32GB | `--batch-size 8` (automatic) |
| M1/M2 Ultra | 64GB+ | `--batch-size 8` (automatic) |

**Note**: Script automatically detects Darwin/Mac and sets batch size to 8 for memory safety.

## Dataset Information

### Dataset Statistics (Exact Token Counts)

Using Qwen3-0.6B-Base tokenizer on full datasets:

| Dataset | Examples | Avg Tokens | Total Tokens | 100M Training | Status |
|---------|----------|------------|--------------|-------------|--------|
| Financial Q&A | 7,000 | 100.1 | 0.70M | 142.7 epochs | ⚠️ High overtraining |
| FinGPT Sentiment | 76,772 | 53.9 | 4.14M | 24.2 epochs | 🟡 Moderate overtraining |
| Finance Alpaca | 68,912 | 122.8 | 8.46M | 11.8 epochs | 🟡 Some overtraining |
| FiQA | 14,511 | 248.4 | 3.60M | 27.8 epochs | 🟡 Moderate overtraining |
| Twitter Sentiment | 9,543 | 29.8 | 0.28M | 357.1 epochs | ❌ Extreme overtraining |
| SEC Reports | 200,000 | 40.6 | 8.12M | 12.3 epochs | 🟡 Some overtraining |
| News Articles | 306,242 | 644.5 | 197.38M | 0.51 epochs | ✅ Good coverage |
| **Mixed Corpus** | ~700,000 | ~318 | **222.69M** | **0.45 epochs** | ✅ Partial but balanced |

### Available Experiments

| ID | Name | Command | Focus |
|----|------|---------|-------|
| 1 | financial_qa | `--experiments 1` | SEC filings Q&A |
| 2 | fingpt | `--experiments 2` | Financial sentiment |
| 3 | alpaca | `--experiments 3` | Instruction following |
| 4 | fiqa | `--experiments 4` | General finance Q&A |
| 5 | twitter | `--experiments 5` | Social media finance |
| 6 | sec_reports | `--experiments 6` | SEC regulatory filings |
| 7 | news_articles | `--experiments 7` | Financial news |
| 8 | mixed | `--experiments mixed` | All combined (recommended) |

## Mixing Strategies

### Overview

When training on mixed corpus (experiment 8), you can choose different dataset mixing strategies to handle the 705x size imbalance between smallest (Twitter, 0.28M) and largest (News, 197.38M) datasets.

### Available Strategies

#### 1. 50% Cap Strategy (Default) - `--mixing-strategy 50cap`

Caps News Articles at 50% and uses square root scaling for others:

| Dataset | Mix Rate | 100M Tokens | Epochs | Zone |
|---------|----------|-----------|--------|------|
| News Articles | 50.0% | 50M | 0.25 | 🟢 Good |
| Finance Alpaca | 13.0% | 13M | 1.54 | 🟢 Good |
| SEC Reports | 13.0% | 13M | 1.6 | 🟢 Good |
| FinGPT | 9.0% | 9M | 2.17 | 🟢 Good |
| FiQA | 8.5% | 8.5M | 2.36 | 🟢 Good |
| Financial Q&A | 4.0% | 4M | 5.71 | 🟡 Acceptable |
| Twitter | 2.5% | 2.5M | 8.93 | 🟡 Acceptable |

```bash
./scripts/phase2b_financial_pretraining.sh --experiments mixed  # Uses 50cap by default
```

#### 2. Square Root Scaling - `--mixing-strategy sqrt`

Pure square root scaling without cap:

| Dataset | Mix Rate | Description |
|---------|----------|-------------|
| News Articles | 56.0% | Gets more weight |
| Twitter | 2.1% | Less overtraining than uniform |

```bash
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy sqrt
```

#### 3. Proportional - `--mixing-strategy proportional`

Based on raw dataset sizes (not recommended):

| Dataset | Mix Rate | Issue |
|---------|----------|-------|
| News Articles | 88.7% | Dominates training |
| Twitter | 0.1% | Barely represented |

```bash
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy proportional
```

#### 4. Uniform - `--mixing-strategy uniform`

Equal weight for all datasets:

| Dataset | Mix Rate | Issue |
|---------|----------|-------|
| All datasets | 14.3% | Twitter: 510 epochs! |

```bash
./scripts/phase2b_financial_pretraining.sh --experiments mixed --mixing-strategy uniform
```

#### 5. Custom - `--mixing-strategy custom`

Provide your own mixture rates (must sum to 1.0):

```bash
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --mixing-strategy custom \
    --custom-mixture-rates "0.1 0.15 0.15 0.1 0.05 0.2 0.25"
    # Order: financial_qa, fingpt, alpaca, fiqa, twitter, sec, news
```

### Training Zones Explained

- 🟢 **Green (1-20 epochs)**: Optimal learning, no overtraining
- 🟡 **Yellow (20-50 epochs)**: Acceptable with monitoring
- 🟠 **Orange (50-100 epochs)**: Significant repetition, use regularization
- 🔴 **Red (>100 epochs)**: Severe memorization risk, avoid

## Dataset Mixture Sampling Implementation

### Overview

When training with mixed datasets (`--experiments mixed`), the pipeline creates a static mixture of all datasets before training begins. Understanding how this sampling works is crucial for reproducibility and optimization.

### Sampling Process

#### 1. Pre-Training Dataset Creation
The mixture is created **once before training starts**:

```python
# For each dataset in the mixture:
for dataset, rate in zip(datasets, mixture_rates):
    target_size = int(total_size * rate)
    
    if len(dataset) >= target_size:
        # Large datasets: Sample WITHOUT replacement
        indices = random.sample(range(len(dataset)), target_size)
    else:
        # Small datasets: Sample WITH replacement (may repeat examples)
        indices = [random.randint(0, len(dataset)-1) for _ in range(target_size)]
```

#### 2. Static Mixture Throughout Training
- **No dynamic resampling**: The same mixture is used for all epochs
- **Fixed order**: After shuffling with `seed=42`, the order remains constant
- **Deterministic iteration**: Each epoch sees exactly the same examples

### Determinism and Reproducibility

#### Deterministic Aspects ✅
- **Final shuffle**: Always uses `seed=42` for the concatenated dataset
- **Training iteration**: Same example order every epoch
- **HuggingFace defaults**: TrainingArguments uses `seed=42` by default

#### Non-Deterministic Aspects ⚠️
- **Initial sampling**: Python's `random.sample()` and `random.randint()` are not seeded
- **Between runs**: Different examples may be selected from each dataset
- **Model initialization**: May vary without explicit seeding

#### Making Fully Deterministic
To ensure complete reproducibility, add at the start of your script:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Sampling Strategy by Dataset Size

| Dataset Type | Sampling Method | Implications |
|--------------|-----------------|-------------|
| **Large datasets** (News, SEC) | Without replacement | Each example appears at most once |
| **Small datasets** (Twitter, FiQA) | With replacement | Examples may repeat multiple times |
| **Mixed corpus** | Concatenated + shuffled | All sampled examples mixed randomly |

### Example: 50% Cap Strategy with 100M Tokens

With `--mixing-strategy 50cap` and 100M token budget:

1. **News Articles** (197M tokens available):
   - Gets 50% = 50M tokens
   - Samples ~77,500 examples without replacement
   - Sees only 25% of available data

2. **Twitter** (0.28M tokens available):
   - Gets 2.5% = 2.5M tokens
   - Needs ~84,000 examples but only has 9,543
   - Each example repeated ~8.8 times (with replacement)

3. **Final mixture**:
   - All sampled examples concatenated
   - Shuffled once with `seed=42`
   - This fixed mixture used for entire training

### Performance Implications

1. **Memory Efficiency**: Static mixture loaded once, no resampling overhead
2. **Training Speed**: No dynamic sampling during training iterations
3. **Convergence**: Small datasets may overfit due to repetition
4. **Evaluation**: Validation uses original datasets, not the mixture

## Multi-Dataset Evaluation

### Overview

The script supports evaluating on all 7 financial datasets even when training on just one, helping monitor cross-dataset generalization.

### Usage

#### Enable Multi-Dataset Evaluation
```bash
# Train on Financial Q&A but evaluate on all datasets
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 \
    --eval-all-datasets
```

#### Disable Multi-Dataset Evaluation (Default)
```bash
# Traditional single dataset training and evaluation
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 \
    --no-eval-all-datasets  # Or just omit the flag
```

### Benefits

1. **Monitor Generalization**: See how training on one dataset affects others
2. **Early Stopping Insights**: Detect overfitting across domains
3. **Dataset Similarity**: Understand which datasets are most related
4. **Optimal Strategy**: Compare single vs mixed training effectiveness

### How It Works

- Training happens on the selected dataset (e.g., Financial Q&A)
- During evaluation, the model is tested on all 7 datasets separately
- Each dataset's metrics are logged independently
- Uses `--separate_mixture_eval` internally for per-dataset metrics

## Complete Options Reference

### Dataset Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--local` | Use locally cached datasets | false | `--local` |
| `--cache-dir PATH` | Directory for cached datasets | ./datasets/phase2b | `--cache-dir /data/cache` |
| `--experiments LIST` | Which experiments to run | all | `--experiments 1,2,3` or `mixed` |
| `--list-experiments` | Show available experiments | - | `--list-experiments` |

### Model Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--model NAME` | HuggingFace model | Qwen/Qwen3-0.6B-Base | `--model gpt2-large` |
| `--model-short NAME` | Short name for dirs | auto-generated | `--model-short my_model` |
| `--attn-implementation` | Attention type | auto | `--attn-implementation eager` |

### LoRA Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--no-lora` | Disable LoRA | false | `--no-lora` |
| `--lora-rank N` | LoRA rank | 32 | `--lora-rank 16` |
| `--lora-alpha N` | LoRA alpha | 64 | `--lora-alpha 32` |
| `--lora-modules` | Target modules | q_proj k_proj v_proj o_proj | `--lora-modules "q_proj v_proj"` |

### Training Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--batch-size N` | Training batch size | 8 (Mac) / 256 (GPU) | `--batch-size 32` |
| `--max-length N` | Max sequence length | 1024 | `--max-length 512` |
| `--max-steps N` | Training steps | 4000 | `--max-steps 1000` |
| `--learning-rate LR` | Learning rate | 2e-5 | `--lr 1e-5` |
| `--lr-scheduler TYPE` | Scheduler | cosine | `--lr-scheduler linear` |
| `--warmup-steps N` | Warmup steps | 400 | `--warmup-steps 100` |
| `--weight-decay W` | Weight decay | 0.01 | `--weight-decay 0.001` |

### Mixed Precision Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--precision MODE` | bf16, fp16, fp32 | bf16 | `--precision fp32` |

### Evaluation Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--eval-batches N` | Max eval batches | 100 | `--eval-batches 50` |
| `--eval-steps N` | Eval frequency | 1000 | `--eval-steps 500` |
| `--eval-on-start` | Eval before training | false | `--eval-on-start` |
| `--eval-all-datasets` | Eval on all datasets | false | `--eval-all-datasets` |
| `--save-steps N` | Save frequency | 500 | `--save-steps 1000` |

### Packing Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--use-packing` | Enable sequence packing | true | `--use-packing` |
| `--packing-max-length N` | Max packed length | uses --max-length | `--packing-max-length 4096` |

### Mixing Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--mixing-strategy NAME` | Dataset mixing strategy | 50cap | `--mixing-strategy sqrt` |
| `--custom-mixture-rates` | Custom rates (7 floats) | - | `--custom-mixture-rates "0.1 0.1 0.2 0.1 0.05 0.2 0.25"` |

### Output Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--output-base-dir PATH` | Output directory | ./runs/phase2b_financial_{model} | `--output-base-dir ./my_runs` |
| `--dry-run` | Test without running | false | `--dry-run` |

## Platform-Specific Optimizations

### Apple Silicon (M1/M2/M3)

The script automatically detects Mac/Darwin and optimizes settings:

#### Auto-Applied Settings on Mac
- Batch size: 8 (vs 256 on GPU)
- Recommended: `--precision fp32` (MPS doesn't support bf16)
- Recommended: `--attn-implementation eager` (prevents NaN losses)

#### M1/M2 Pro/Max (32GB) Recommended
```bash
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --batch-size 4 \
    --precision fp32 \
    --attn-implementation eager \
    --max-length 512 \
    --use-lora --lora-rank 16 \
    --eval-batches 10
```

#### M1/M2 Base (8-16GB) Recommended
```bash
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 \
    --batch-size 2 \
    --precision fp32 \
    --attn-implementation eager \
    --max-length 256 \
    --use-lora --lora-rank 8 \
    --eval-batches 5
```

### NVIDIA GPUs

#### RTX 4090 (24GB)
```bash
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --batch-size 128 \
    --precision bf16 \
    --use-packing \
    --eval-steps 500
```

#### RTX 3060 (12GB)
```bash
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --batch-size 16 \
    --precision fp16 \
    --use-lora --lora-rank 16 \
    --max-length 512
```

#### A100 (80GB)
```bash
./scripts/phase2b_financial_pretraining.sh \
    --model meta-llama/Llama-2-7b \
    --experiments mixed \
    --no-lora \
    --batch-size 512 \
    --precision bf16
```

## Example Commands Gallery

### Quick Test Run
```bash
# Test with minimal steps
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 \
    --batch-size 2 \
    --max-steps 10 \
    --eval-steps 5 \
    --save-steps 10
```

### Production Training
```bash
# Recommended production settings
./scripts/phase2b_financial_pretraining.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --experiments mixed \
    --mixing-strategy 50cap \
    --batch-size 256 \
    --precision bf16 \
    --max-steps 20000 \
    --eval-steps 1000 \
    --save-steps 5000
```

### Memory-Constrained Training
```bash
# For 12-16GB GPUs
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --batch-size 8 \
    --use-lora --lora-rank 8 \
    --max-length 512 \
    --precision fp16 \
    --eval-batches 10
```

### Multi-Dataset Evaluation Example
```bash
# Train on FinGPT, evaluate on all
./scripts/phase2b_financial_pretraining.sh \
    --experiments 2 \
    --eval-all-datasets \
    --eval-steps 100 \
    --eval-batches 20
```

### Custom Mixing Example
```bash
# 30% News, 20% each for Alpaca/SEC, rest distributed
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed \
    --mixing-strategy custom \
    --custom-mixture-rates "0.05 0.10 0.20 0.10 0.05 0.20 0.30"
```

### Development/Testing
```bash
# Fast iteration
./scripts/phase2b_financial_pretraining.sh \
    --model distilgpt2 \
    --experiments 5 \
    --batch-size 4 \
    --max-steps 100 \
    --eval-steps 25 \
    --save-steps 50 \
    --output-base-dir ./test_runs
```

### Hyperparameter Sweep
```bash
# Test different LoRA ranks
for rank in 8 16 32; do
    ./scripts/phase2b_financial_pretraining.sh \
        --experiments 1 \
        --lora-rank $rank \
        --lora-alpha $((rank * 2)) \
        --max-steps 500 \
        --output-base-dir ./runs/lora_r${rank}
done
```

## Performance Tuning

### Expected Training Times

#### RTX 4090 (24GB) with LoRA

| Model | Batch Size | 100M Tokens | 200M Tokens |
|-------|------------|-------------|-------------|
| 0.6B | 256 | 0.8-1h | 1.5-1.9h |
| 1.7B | 128 | 1-1.4h | 2-2.8h |
| 4B | 64 | 1.4-1.9h | 2.8-3.9h |

#### Apple M1 Max (32GB) with LoRA

| Model | Batch Size | 100M Tokens | 200M Tokens |
|-------|------------|-------------|-------------|
| 0.6B | 8 | 2.4-3h | 4.8-6h |
| 1.7B | 4 | 3.6-4.8h | 7.2-9.6h |

### Memory Optimization Tips

1. **Use LoRA**: Reduces memory by 50-75%
2. **Enable Packing**: Better GPU utilization
3. **Reduce Max Length**: `--max-length 512` saves memory
4. **Use Mixed Precision**: `--precision bf16` (except on MPS)
5. **Gradient Accumulation**: Simulate larger batches

### Speed Optimization Tips

1. **Enable Packing**: `--use-packing` for 2.5x speedup
2. **Larger Batch Size**: If memory allows
3. **Reduce Eval Frequency**: `--eval-steps 5000`
4. **Limit Eval Batches**: `--eval-batches 10`
5. **Use BF16**: Better than FP16 for stability

## Integration with Pipeline

### Using Phase 2B Checkpoints

1. **Complete Phase 2B**:
```bash
./scripts/phase2b_financial_pretraining.sh --experiments mixed
```

2. **Find Best Checkpoint**:
```bash
ls ./runs/phase2b_financial_*/mixed_financial/checkpoints/
# Look for checkpoint-20000 or similar
```

3. **Use in Phase 2C**:
```bash
# Edit phase2c script with checkpoint path
CHECKPOINT_PATH="./runs/phase2b_financial_qwen3_0.6b/mixed_financial/checkpoint-20000"
```

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir ./runs/phase2b_financial_qwen3_0.6b

# Watch logs
tail -f ./runs/phase2b_financial_*/*/logs/training.log

# Check GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Check MPS usage (Mac)
# Open Activity Monitor → Window → GPU History
```

## Troubleshooting

### Common Issues

#### NaN Loss on MPS
```bash
# Solution: Use eager attention
--attn-implementation eager --precision fp32
```

#### Out of Memory
```bash
# Solutions (try in order):
--batch-size 4
--use-lora --lora-rank 8
--max-length 512
--precision fp16  # or fp32 on MPS
```

#### Slow Training on Mac
```bash
# Optimizations:
--batch-size 2  # Smaller but more frequent updates
--eval-steps 5000  # Less frequent evaluation
--eval-batches 5  # Quick evaluations
--use-packing  # Better efficiency
```

#### Dataset Download Fails
```bash
# Use local datasets:
python scripts/download_phase2b_datasets.py  # Download once
./scripts/phase2b_financial_pretraining.sh --local  # Use cached
```

#### Wrong Python Environment
```bash
# Specify Python path:
./scripts/phase2b_financial_pretraining.sh \
    --python-cmd /path/to/conda/envs/lancy/bin/python
```

### Performance Issues

| Issue | Solution |
|-------|----------|
| Training too slow | Enable packing, increase batch size, reduce eval frequency |
| Loss not decreasing | Lower learning rate, increase warmup, check data |
| Unstable training | Use FP32, reduce learning rate, enable gradient clipping |
| High memory usage | Enable LoRA, reduce batch size, use gradient accumulation |

## Summary

The Phase 2B financial pretraining script provides:

1. **7 Financial Datasets**: Comprehensive coverage of financial domains
2. **5 Mixing Strategies**: Handle 705x size imbalance effectively
3. **Multi-Dataset Evaluation**: Monitor cross-domain generalization
4. **Platform Optimization**: Auto-detects and optimizes for hardware
5. **Memory Efficiency**: LoRA support, packing, mixed precision
6. **Flexible Configuration**: Extensive options for customization

### Quick Reference

```bash
# Most common command (recommended)
./scripts/phase2b_financial_pretraining.sh --experiments mixed

# Memory-constrained Mac
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed --batch-size 4 --max-length 512

# Test new mixing strategy
./scripts/phase2b_financial_pretraining.sh \
    --experiments mixed --mixing-strategy sqrt --dry-run

# Multi-dataset evaluation
./scripts/phase2b_financial_pretraining.sh \
    --experiments 1 --eval-all-datasets
```

---

*Version: 3.0 - Comprehensive Guide*  
*Last Updated: November 2024*  
*Script Features: Mixing strategies, multi-dataset eval, Mac optimization*