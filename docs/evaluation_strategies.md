# Evaluation Strategies Guide

This guide explains how to control when and how evaluation runs during training.

## Overview

The training pipeline supports three evaluation strategies:
- **Epoch-based**: Evaluate at the end of each epoch
- **Step-based**: Evaluate every N training steps  
- **No evaluation**: Disable evaluation during training

## Primary Control: `--eval_strategy`

The `--eval_strategy` parameter is the primary control for evaluation behavior:

```bash
--eval_strategy epoch  # Evaluate at the end of each epoch (default)
--eval_strategy steps  # Evaluate every N steps (requires --eval_steps)
--eval_strategy no     # Disable evaluation during training
```

## Evaluation Options

### 1. Epoch-Based Evaluation (Default)

Evaluate at the end of each training epoch:

```bash
# Default behavior - evaluates after each epoch
python train.py --model gpt2 --dataset wikitext --mode pretrain --num_train_epochs 3

# Explicit epoch evaluation
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_strategy epoch --num_train_epochs 3

# With evaluation at start
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_strategy epoch --eval_on_start --num_train_epochs 3
```

### 2. Step-Based Evaluation

Evaluate every N training steps:

```bash
# Method 1: Auto-set (convenience feature)
# When you specify --eval_steps alone, it automatically sets eval_strategy='steps'
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_steps 100

# Method 2: Explicit specification
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_strategy steps --eval_steps 100

# With evaluation at start (step 0)
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_steps 100 --eval_on_start
```

### 3. No Evaluation

Disable evaluation during training (only train):

```bash
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_strategy no
```

## Additional Evaluation Parameters

### `--eval_on_start`
Run evaluation at step 0 before training begins:

```bash
# Works with any evaluation strategy
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_strategy epoch --eval_on_start

python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_steps 100 --eval_on_start
```

### `--eval_max_batches`
Limit evaluation to N batches for faster (but approximate) evaluation:

```bash
# Evaluate on only 10 batches (faster but less accurate)
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_steps 50 --eval_max_batches 10

# Full evaluation (default, -1 means all batches)
python train.py --model gpt2 --dataset wikitext --mode pretrain \
    --eval_steps 100 --eval_max_batches -1
```

## Common Use Cases

### Quick Experimentation
Frequent evaluation with limited batches for fast feedback:
```bash
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \
    --eval_steps 25 --eval_max_batches 5 --eval_on_start
```

### Standard Training
Balanced evaluation frequency:
```bash
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \
    --eval_steps 200 --save_steps 500
```

### Long Pretraining
Infrequent evaluation to minimize overhead:
```bash
python train.py --model Qwen/Qwen3-0.6B --dataset wikitext --mode pretrain \
    --eval_steps 1000 --save_steps 5000 --max_steps 50000
```

### Production Training with Epochs
Train for multiple epochs with per-epoch evaluation:
```bash
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \
    --eval_strategy epoch --num_train_epochs 5 \
    --save_strategy epoch --load_best_model_at_end
```

### Debugging
Maximum visibility with minimal training:
```bash
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \
    --eval_steps 10 --eval_max_batches 3 --eval_on_start --max_steps 50
```

## Important Notes

1. **Auto-setting behavior**: When `--eval_steps` is specified without `--eval_strategy`, the system automatically sets `eval_strategy='steps'` for convenience.

2. **Override auto-setting**: You can still explicitly set `--eval_strategy epoch` even if `--eval_steps` is specified (though this would ignore the eval_steps value).

3. **Evaluation metrics**: During evaluation, the system computes:
   - `eval_loss`: The loss on the evaluation dataset
   - For pretraining: Perplexity (computed from eval_loss)
   - For classification: Accuracy (if applicable)

4. **Checkpointing**: Evaluation and checkpointing are independent:
   - Use `--save_strategy` and `--save_steps` for checkpointing
   - Use `--eval_strategy` and `--eval_steps` for evaluation
   - They can have different frequencies

## Summary

- **Primary control**: Use `--eval_strategy` to choose between epoch/steps/no evaluation
- **Convenience**: `--eval_steps` alone auto-sets step-based evaluation
- **Flexibility**: Combine with `--eval_on_start` and `--eval_max_batches` for fine control
- **Independence**: Evaluation frequency is independent of checkpoint saving frequency