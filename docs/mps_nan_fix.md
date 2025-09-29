# MPS NaN Evaluation Loss Fix

## Root Cause Discovered

The NaN issue on MPS devices is caused by **PyTorch's SDPA (Scaled Dot-Product Attention) implementation on Metal Performance Shaders**. 

### The Problem

When training models on Apple Silicon (MPS) devices, you may encounter NaN values for `eval_loss` during evaluation, while training loss remains valid. This occurs specifically when:
- Using SDPA attention implementation (PyTorch's default optimized attention)
- During evaluation (not training)
- With left-padded sequences (standard in language modeling)
- Using `DataCollatorForLanguageModeling`

## Solutions

### Solution 1: Automatic Attention Switching (Recommended)

The training pipeline now **automatically switches from SDPA to eager attention on MPS devices** to prevent NaN values. This happens automatically when you run training on MPS.

### Solution 2: MPSSafeTrainer (Heavy-handed Alternative)

If the automatic attention switching doesn't work, the pipeline can use `MPSSafeTrainer` that computes the loss on CPU during evaluation. This is enabled by default but can be disabled with `--disable_mps_fix`.

## Usage

### Default Behavior (Recommended)

By default, the MPS fix is **enabled** on Apple Silicon devices:

```bash
python train.py --model gpt2 --dataset wikitext --dataset_config wikitext-2-raw-v1 --mode pretrain
```

Output will show:
```
Using MPSSafeTrainer for MPS device (prevents NaN eval loss)
```

### Disabling the Fix

If you want to use the original HuggingFace Trainer (which may produce NaN eval_loss), use the `--disable_mps_fix` flag:

```bash
python train.py --model gpt2 --dataset wikitext --dataset_config wikitext-2-raw-v1 --mode pretrain --disable_mps_fix
```

You'll see a warning:
```
⚠️  WARNING: MPS device detected but MPS fix disabled via --disable_mps_fix
    You may experience NaN eval_loss during evaluation.
    To enable the fix, remove the --disable_mps_fix flag.
Using standard HuggingFace Trainer
```

## When to Disable the Fix

You might want to disable the fix if:
- You're debugging trainer behavior
- You want to compare performance with/without the fix
- You're using a different evaluation strategy that doesn't trigger the NaN issue
- You need exact compatibility with standard HuggingFace Trainer behavior

## Performance Impact

The MPS fix has minimal performance impact:
- **Training speed**: No impact (fix only applies during evaluation)
- **Evaluation speed**: Slight slowdown due to CPU-GPU data transfer
- **Memory usage**: Minimal additional memory for temporary CPU tensors

## Technical Details

The `MPSSafeTrainer` works by:
1. Detecting when the model is in evaluation mode on MPS
2. Moving loss computation to CPU to avoid MPS numerical instabilities
3. Moving the result back to MPS to continue evaluation

This approach maintains full training performance while ensuring stable evaluation metrics.

## Verification

To verify the fix is working, check your evaluation logs:
- **With fix**: `eval_loss` should be a valid number (e.g., `6.601303`)
- **Without fix**: `eval_loss` may show as `nan`

## Related Files

- `src/training/mps_safe_trainer.py`: Implementation of the MPS-safe trainer
- `src/training/utils.py`: Trainer selection logic
- `src/config/args.py`: Command-line argument definitions