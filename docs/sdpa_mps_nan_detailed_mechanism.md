# Detailed Mechanism of SDPA NaN Bug on MPS

## Executive Summary

The NaN bug occurs when PyTorch's `scaled_dot_product_attention` (SDPA) is used on Apple Silicon (MPS) devices with left-padded sequences. The bug is triggered by a critical logic flaw in how SDPA determines whether to apply causal masking, combined with numerical instabilities in the Metal Performance Shaders implementation.

## The Complete Bug Mechanism

### 1. Initial Conditions

The bug requires ALL of these conditions to manifest:

- **Device**: Apple Silicon (MPS) using Metal Performance Shaders
- **Attention**: SDPA implementation (`torch.nn.functional.scaled_dot_product_attention`)
- **Padding**: Left padding (`tokenizer.padding_side = 'left'`)
- **Mode**: Evaluation mode (not training)
- **Task**: Causal language modeling with `DataCollatorForLanguageModeling(mlm=False)`

### 2. The Critical Logic Flaw

In `transformers/integrations/sdpa_attention.py`, line 73:

```python
is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
```

This line contains the root cause:
- `is_causal` is only `True` when `attention_mask is None`
- With left padding, we ALWAYS have an `attention_mask` to mask padding tokens
- Therefore, `is_causal` becomes `False` when it should be `True`

### 3. The Numerical Pathway

#### Step 1: Tokenization with Left Padding
```
Input: ["Short text", "This is a longer text that needs less padding"]
After tokenization with left padding:
[PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, "Short", "text"]
["This", "is", "a", "longer", "text", "that", "needs", "less", "padding"]
```

#### Step 2: Attention Mask Creation
```python
attention_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # 0 for padding, 1 for real
# Converted to additive mask:
attention_mask = [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 0, 0]
```

#### Step 3: SDPA Call with Wrong Parameters
```python
# What happens:
F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=attention_mask,  # Only padding mask
    is_causal=False  # WRONG! Should be True
)

# What should happen:
F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=combined_mask,  # Padding + causal mask
    is_causal=True  # Or handle both masks correctly
)
```

#### Step 4: Attention Pattern Corruption

For position 9 ("Short" - first real token after padding):

**Correct attention pattern (with causal mask):**
```
Can attend to: [✗, ✗, ✗, ✗, ✗, ✗, ✗, ✗, ✗, ✓, ✗]
              (padding masked)        (self) (future masked)
```

**Buggy SDPA pattern (without causal mask):**
```
Can attend to: [✗, ✗, ✗, ✗, ✗, ✗, ✗, ✗, ✗, ✓, ✓]
              (padding masked)        (self) (FUTURE VISIBLE!)
```

#### Step 5: Numerical Instability in Metal Kernel

The Metal kernel receives:
1. Attention scores with mixed valid/invalid patterns
2. Some positions have all-masked attention (padding)
3. Some positions have future leakage (breaks causality)
4. The softmax computation becomes unstable

The specific numerical failure:
```
Softmax of attention scores for a corrupted row:
exp([-inf, -inf, -inf, ..., valid_score, leaked_future_score])
= [0, 0, 0, ..., exp(valid), exp(leaked)]

In MPS with lower precision:
- exp(-inf) might not cleanly become 0
- Division by near-zero sum causes overflow
- Result: NaN propagation
```

### 4. Why MPS Specifically?

#### MPS/Metal Characteristics:
- **Lower Precision**: Likely uses fp16 or mixed precision internally
- **Different Inf Handling**: Metal shaders handle infinity differently than CUDA
- **Optimization Assumptions**: May assume certain mask patterns that break here
- **No Fallback**: Unlike CUDA, doesn't fall back to safer implementations

#### CUDA/CPU Don't Fail:
- **CUDA**: Robust handling of edge cases, proper inf/nan handling
- **CPU**: Full precision (fp32/fp64), explicit error checking

### 5. The Cascading Failure

Once NaN appears in attention weights:
1. **Forward Pass**: NaN propagates through all subsequent layers
2. **Loss Computation**: Cross-entropy loss becomes NaN
3. **Backward Pass**: Gradients become NaN (if training)
4. **Metrics**: Evaluation metrics show NaN

### 6. Proof of Mechanism

Our test (`test_exact_nan_conditions.py`) confirms:
- **Eager attention**: No NaN (manually applies both masks correctly)
- **SDPA attention**: NaN at position [0,0,0] (very first output!)

The NaN appears immediately because the first attended position already has corrupted attention patterns due to the missing causal mask.

### 7. The Fix

The implemented fix in `src/models/utils.py`:

```python
if device == 'mps' and attn_implementation in ['auto', 'sdpa']:
    print("⚠️  SDPA attention can cause NaN values on MPS devices...")
    attn_implementation = 'eager'
```

This forces eager attention which:
- Manually applies causal mask with `torch.where`
- Explicitly handles mask combination
- Doesn't rely on the buggy `is_causal` logic

## Detailed Code Analysis

### SDPA Implementation (Buggy)
```python
# In sdpa_attention_forward:
if attention_mask is not None:  # True with padding
    is_causal = False  # BUG: Loses causal masking!

torch.nn.functional.scaled_dot_product_attention(
    query, key, value,
    attn_mask=attention_mask,  # Only padding mask
    is_causal=False,  # No causal mask!
)
```

### Eager Implementation (Correct)
```python
# In eager_attention_forward (GPT2):
# 1. Compute scores
attn_weights = torch.matmul(query, key.transpose(-1, -2))

# 2. Apply causal mask explicitly
causal_mask = self.bias[...]  # Pre-computed causal mask
attn_weights = torch.where(causal_mask, attn_weights, mask_value)

# 3. Apply attention mask for padding
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# 4. Softmax - both masks properly applied
attn_weights = F.softmax(attn_weights, dim=-1)
```

## Root Cause Summary

The bug is fundamentally a **logic error in SDPA's mask handling** that manifests as a **numerical instability on MPS**:

1. **Logic Error**: SDPA incorrectly assumes that presence of `attention_mask` means no causal masking needed
2. **Mask Corruption**: Causal mask is not applied, allowing attention to future tokens
3. **MPS Instability**: Metal kernel can't handle the invalid attention pattern numerically
4. **NaN Result**: Numerical errors in Metal's softmax implementation produce NaN

## Implications

This bug reveals:
- **API Design Issue**: SDPA's `is_causal` flag shouldn't be mutually exclusive with `attn_mask`
- **MPS Maturity**: Metal Performance Shaders need better numerical stability
- **Testing Gap**: This common scenario (left padding + causal LM) wasn't properly tested

## Recommendations

1. **Short-term**: Use eager attention on MPS (current fix)
2. **Medium-term**: PyTorch should fix SDPA to handle both masks correctly
3. **Long-term**: Apple should improve Metal kernel numerical stability