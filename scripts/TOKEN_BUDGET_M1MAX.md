# M1 Max Token Budget Configuration

## 100M Token Budget Per Experiment

This document explains how the M1 Max script manages a 100M (0.1B) token budget for EACH experiment in Phase 2B financial pretraining, optimized for Apple Silicon memory constraints.

## Token Consumption Formula

```
tokens_per_step = batch_size × max_length
total_tokens = tokens_per_step × num_steps
```

Where:
- **batch_size** = actual batch size
- **max_length** = maximum sequence length
- **Each experiment gets a 100M token budget**

**Note on Packing**: Sequence packing does NOT change tokens per step. It improves efficiency by replacing padding tokens with actual content, but the total token count remains the same.

## Default Configuration

### M1 Max Defaults (Optimized for 32GB RAM)
- **Sequence Length**: 512 tokens (balanced default, range: 256-1024)
- **Batch Size**: 8 (conservative for stability)
- **Packing**: ENABLED (reduces padding waste, not token count)
- **Mixed Precision**: DISABLED (FP32 for MPS stability)
- **Device**: MPS (Metal Performance Shaders)
- **Tokens per step**: 8 × 512 = 4,096 tokens
- **Max steps**: 100M ÷ 4,096 ≈ 24,414 steps

### Automatic Batch Configuration

The script automatically adjusts batch size based on sequence length:

| Sequence Length | Batch Size | Tokens/Step | Max Steps |
|----------------|------------|-------------|-----------|
| 256 | 16 | 4,096 | 24,414 |
| 512 | 8 | 4,096 | 24,414 |
| 768 | 6 | 4,608 | 21,701 |
| 1024 | 4 | 4,096 | 24,414 |

## Dataset Coverage with 100M Tokens

### At Default 512 Sequence Length

#### Small Datasets (Multiple Epochs)
| Dataset | Size | Steps@512 | Approx Epochs |
|---------|------|-----------|---------------|
| 1. Financial Q&A | 7.1K samples | 24,414 | ~27.5 |
| 5. Twitter | 1.1K samples | 24,414 | ~177 |
| 4. FiQA | 17.4K samples | 24,414 | ~11.2 |

#### Medium Datasets (Partial Epoch)
| Dataset | Size | Steps@512 | Approx Coverage |
|---------|------|-----------|-----------------|
| 2. FinGPT Sentiment | 76.8K samples | 24,414 | ~2.5 epochs |
| 3. Finance Alpaca | 68.9K samples | 24,414 | ~2.8 epochs |

#### Large Datasets (Token-Based Coverage)
| Dataset | Token Size | Steps@512 | Token Coverage |
|---------|------------|-----------|----------------|
| 6. SEC Reports | 80M tokens | 24,414 | 100M budget |
| 7. News Articles | 197M tokens | 24,414 | 100M budget |
| mixed | 207M tokens | 24,414 | 100M budget |

## M1 Max vs RTX4090 Comparison

| Aspect | M1 Max | RTX4090 |
|--------|--------|---------|
| Default Seq Length | 512 | 1024 |
| Min Seq Length | 256 | 1024 |
| Max Seq Length | 1024 | 2048+ |
| Default Batch Size | 4 | 4 |
| Mixed Precision | FP32 | BF16 |
| Device | MPS | CUDA |
| Eval Max Batches | 50 | 100 |
| Memory | 32GB unified | 24GB VRAM |

## Sequence Length Strategy

### Short Sequences (256 tokens) - Fastest Training
- **Best for**: Twitter sentiment, short financial news
- **Config**: BS=16, ~24.4K steps
- **Benefits**: Same training steps, better for small texts
- **Trade-off**: Less context per sample

### Balanced (512 tokens) - Default
- **Best for**: Most financial datasets
- **Config**: BS=8, ~24.4K steps
- **Benefits**: Good balance of context and training steps
- **Trade-off**: Standard configuration

### Long Sequences (1024 tokens) - Maximum Context
- **Best for**: SEC reports, complex documents
- **Config**: BS=4, ~24.4K steps
- **Benefits**: Full document context
- **Trade-off**: Higher memory usage

## Quick Commands

```bash
# Standard run with balanced settings
./scripts/phase2b_m1max.sh --experiments 1

# Twitter sentiment with short sequences (more epochs)
./scripts/phase2b_m1max.sh --experiments 5 --max-length 256

# SEC Reports with maximum context
./scripts/phase2b_m1max.sh --experiments 6 --max-length 1024

# Mixed corpus with strategy
./scripts/phase2b_m1max.sh --experiments mixed --strategy 50cap

# Quick test (10% of budget)
./scripts/phase2b_m1max.sh --experiments 2 --max-steps 200

# Custom batch configuration
./scripts/phase2b_m1max.sh --experiments 3 --batch-size 2

# All experiments sequentially (700M tokens total)
for i in {1..7}; do
  ./scripts/phase2b_m1max.sh --experiments $i
done
```

## Memory Management Tips

### M1 Max 32GB Unified Memory

1. **Monitor Memory Pressure**: Use Activity Monitor during training
2. **Reduce if OOM**:
   - Lower batch size (minimum 1)
   - Reduce sequence length
   - Disable packing if necessary
3. **MPS Specific**:
   - FP32 is more stable than mixed precision
   - Restart Python between long runs to clear memory
   - Use `--no-packing` if memory issues persist

### Safe Configurations

| Seq Length | Batch Size | Model Memory | Safe for M1 Max 32GB |
|------------|------------|--------------|----------------------|
| 256 | 8 | ~6GB | ✅ Very Safe |
| 512 | 4 | ~10GB | ✅ Safe (default) |
| 768 | 3 | ~14GB | ✅ Safe |
| 1024 | 2 | ~18GB | ⚠️ Monitor carefully |

## Training Strategy Recommendations

### For Small Datasets (Twitter, FiQA)
- **Use 256-512 sequence length**
- Multiple epochs help with limited data
- Monitor for overfitting (especially Twitter with 2.2 epochs)
- Consider early stopping

### For Medium Datasets (FinGPT, Finance Alpaca)
- **Use 512 sequence length (default)**
- 3-4% coverage is limited but useful for testing
- Focus on data quality over quantity

### For Large Datasets (SEC, News)
- **Can use 768-1024 for better context**
- 100M token budget provides limited but useful coverage
- Longer sequences capture document structure

### For Mixed Corpus
- **Use 512 default for balance**
- Apply mixture strategy to prevent imbalance
- Benefits from diverse financial content

## Monitoring and Optimization

1. **Track Training Metrics**:
   - Loss convergence
   - Memory usage (Activity Monitor)
   - Tokens per second

2. **Early Stopping**:
   - Save checkpoints every 500 steps
   - Stop if loss plateaus for 2000 steps
   - Resume from best checkpoint

3. **Performance Tips**:
   - Close other applications
   - Ensure adequate cooling
   - Use local datasets to avoid download overhead

## Comparison with Cloud GPUs

| Metric | M1 Max | RTX 4090 | A100 40GB |
|--------|--------|----------|-----------|
| Tokens/sec @ 512 | ~500 | ~2000 | ~3000 |
| Time for 100M tokens | ~3.3 hours | ~0.8 hours | ~0.55 hours |
| Cost | $0 (local) | ~$24 | ~$40 |
| Availability | Always | Rental | Cloud |

## Notes

- **Each experiment uses 100M tokens** independently
- **MPS optimizations** may vary by macOS version
- **Unified memory** allows flexible allocation
- **Power efficiency** excellent for overnight runs
- Consider using **caffeinate** to prevent sleep during training
- **Sequence packing** critical for efficiency on M1 Max