# Attention Implementation Guide

This guide explains how to use different attention implementations in the training pipeline.

## Available Implementations

### 1. **Auto** (Default)
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation auto
```
- Lets the model choose the best available implementation
- Recommended for most users

### 2. **Eager**
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation eager
```
- Vanilla PyTorch implementation
- Most compatible, works on all devices
- Best for debugging (simplest code path)
- Use when investigating NaN issues or other numerical problems

### 3. **SDPA** (Scaled Dot-Product Attention)
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation sdpa
```
- PyTorch's native optimized attention (torch.nn.functional.scaled_dot_product_attention)
- Good balance of performance and compatibility
- Works on CUDA, MPS (Apple Silicon), and CPU
- Recommended for MPS devices

### 4. **Flash Attention 2**
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation flash_attention_2
```
- Memory-efficient exact attention computation
- 2-4x faster than standard attention
- **CUDA only** - requires GPU with compute capability ≥ 8.0
- Requires installation: `pip install flash-attn --no-build-isolation`
- Best for training large models with long sequences

### 5. **Flash Attention 3**
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation flash_attention_3
```
- Latest version of Flash Attention
- Further performance improvements
- **CUDA only** - same requirements as Flash Attention 2

### 6. **Flex Attention**
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation flex_attention
```
- Allows custom attention patterns
- Requires PyTorch 2.5+
- Useful for research and custom attention masks

### 7. **Paged Attention**
```bash
python train.py --model gpt2 --dataset wikitext --attn_implementation paged_attention
```
- Optimized for inference with KV-cache management
- Reduces memory fragmentation
- **CUDA only** - primarily for inference servers

## Device Compatibility

| Implementation | CUDA | MPS (Apple) | CPU |
|---------------|------|-------------|-----|
| eager         | ✅   | ✅          | ✅  |
| sdpa          | ✅   | ⚠️ **See warning below** | ✅  |
| flash_attention_2 | ✅ | ❌ (falls back to eager) | ❌ |
| flash_attention_3 | ✅ | ❌ (falls back to eager) | ❌ |
| flex_attention | ✅  | ✅          | ✅  |
| paged_attention | ✅ | ❌ (falls back to eager) | ❌ |

### ⚠️ Important: SDPA on MPS (Apple Silicon) Issue

**SDPA causes NaN values on MPS devices during language model evaluation.** This is a known issue with PyTorch's `scaled_dot_product_attention` implementation on Metal Performance Shaders.

The issue specifically occurs when:
- Using MPS (Apple Silicon) devices
- During model evaluation (not training)
- With left-padded sequences (common in language modeling)
- Using DataCollatorForLanguageModeling

**Automatic Protection**: The training pipeline automatically detects this condition and switches to `eager` attention when:
- Device is MPS
- Attention implementation is `auto` or `sdpa`

You'll see this warning:
```
⚠️  SDPA attention can cause NaN values on MPS devices during evaluation
   This is a known issue with PyTorch's scaled_dot_product_attention on Metal.
   Automatically using 'eager' attention for numerical stability.
   To suppress this warning, explicitly use --attn_implementation eager
```

## Automatic Fallback

The system automatically falls back to compatible implementations:
- **MPS devices**: CUDA-only implementations fall back to `sdpa`
- **CPU**: CUDA-only implementations fall back to `eager`
- **Missing packages**: If flash-attn is not installed, falls back to `sdpa`

## Use Cases

### Debugging NaN Issues
```bash
# Use eager for simplest implementation
python train.py --model gpt2 --dataset wikitext \
    --attn_implementation eager \
    --disable_mps_fix  # If on MPS and want to reproduce NaN
```

### Production Training on CUDA
```bash
# Use Flash Attention 2 for efficiency
python train.py --model gpt2 --dataset wikitext \
    --attn_implementation flash_attention_2 \
    --max_length 2048  # Can handle longer sequences efficiently
```

### Training on Apple Silicon (M1/M2)
```bash
# Use SDPA for best MPS performance
python train.py --model gpt2 --dataset wikitext \
    --attn_implementation sdpa \
    --device mps
```

### Comparing Implementations
```bash
# Compare performance between implementations
for impl in eager sdpa flash_attention_2; do
    echo "Testing $impl"
    python train.py --model gpt2 --dataset wikitext \
        --attn_implementation $impl \
        --max_steps 100
done
```

## Backward Compatibility

The old `--use_flash_attention` flag is deprecated but still works:
```bash
# Deprecated (but still works)
python train.py --model gpt2 --use_flash_attention

# New way (preferred)
python train.py --model gpt2 --attn_implementation flash_attention_2
```

## Troubleshooting

### Flash Attention Not Working
If you get an error about flash-attn not being installed:
```bash
# Install flash-attn (CUDA only)
pip install flash-attn --no-build-isolation
```

### Wrong PyTorch Version for Flex Attention
If flex_attention fails:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Need PyTorch 2.5+ for flex_attention
```

### NaN Issues During Training
If experiencing NaN losses:
```bash
# Use eager attention for debugging
python train.py --model gpt2 --dataset wikitext \
    --attn_implementation eager \
    --eval_on_start \
    --eval_steps 10
```

## Performance Comparison

Approximate relative performance (may vary by model/hardware):

| Implementation | Speed | Memory Usage | Accuracy |
|---------------|-------|--------------|----------|
| eager         | 1.0x  | Baseline     | Exact    |
| sdpa          | 1.5-2x| ~Baseline    | Exact    |
| flash_attention_2 | 2-4x | 50-70%    | Exact    |
| flash_attention_3 | 3-5x | 50-70%    | Exact    |

Note: Actual performance depends on sequence length, batch size, model size, and hardware.