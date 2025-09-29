# GPU Hours Estimation for Qwen2-0.5B Experiments

## Model Configuration
- **Model**: Qwen/Qwen2-0.5B (500M parameters)
- **LoRA Configuration**: rank=4, alpha=8 (~2.5M trainable parameters, 0.5% of total)
- **Hardware**: RTX 4090 (24GB VRAM)
- **Expected Throughput**: ~200-250 tokens/second with batch_size=8

## Training Speed Estimates

### Base Calculations
- **Tokens per step**: batch_size(8) × max_length(128) = 1,024 tokens/step
- **Steps per hour**: ~1,800-2,000 steps/hour
- **With LoRA**: ~30-40% faster than full fine-tuning

### Empirical Estimates (RTX 4090 with LoRA rank=4)
| Steps | Estimated Time |
|-------|---------------|
| 1,000 | 0.5 hours |
| 5,000 | 2.5 hours |
| 10,000 | 5 hours |
| 20,000 | 10 hours |
| 50,000 | 25 hours |

## Phase 1: Baseline Establishment
**10 experiments × evaluation only**
- Each evaluation: ~10 minutes (0.17 hours)
- **Total: 1.7 hours**

## Phase 2: Pretraining Experiments

### 2A: General Pretraining (9 experiments)
| Experiment | Steps | Time (hours) |
|------------|-------|--------------|
| OpenWebText 10k | 10,000 | 5.0 |
| OpenWebText 25k | 25,000 | 12.5 |
| OpenWebText 50k | 50,000 | 25.0 |
| Two-corpus mix 10k | 10,000 | 5.0 |
| Two-corpus mix 25k | 25,000 | 12.5 |
| Two-corpus mix 50k | 50,000 | 25.0 |
| Three-corpus mix 10k | 10,000 | 5.0 |
| Three-corpus mix 25k | 25,000 | 12.5 |
| Three-corpus mix 50k | 50,000 | 25.0 |
| **Subtotal** | | **127.5 hours** |

### 2B: Domain Continued Pretraining (12 experiments)
| Experiment | Steps | Time (hours) |
|------------|-------|--------------|
| Financial QA 5k | 5,000 | 2.5 |
| Financial QA 10k | 10,000 | 5.0 |
| FinGPT sentiment 10k | 10,000 | 5.0 |
| FinGPT sentiment 20k | 20,000 | 10.0 |
| Finance Alpaca 10k | 10,000 | 5.0 |
| Finance Alpaca 20k | 20,000 | 10.0 |
| Mixed financial 10k | 10,000 | 5.0 |
| Mixed financial 20k | 20,000 | 10.0 |
| FiQA 5k | 5,000 | 2.5 |
| FiQA 10k | 10,000 | 5.0 |
| Twitter sentiment 5k | 5,000 | 2.5 |
| Twitter sentiment 10k | 10,000 | 5.0 |
| **Subtotal** | | **67.5 hours** |

### 2C: Cross-Domain Pretraining (6 experiments)
| Experiment | Steps | Time (hours) |
|------------|-------|--------------|
| GSM8K math | 10,000 | 5.0 |
| DeepMind math | 10,000 | 5.0 |
| BigCodeBench | 10,000 | 5.0 |
| GLUE MNLI | 10,000 | 5.0 |
| MMLU-Pro | 10,000 | 5.0 |
| Math+Code mix | 10,000 | 5.0 |
| **Subtotal** | | **30.0 hours** |

**Phase 2 Total: 225.0 hours**

## Phase 3: Fine-tuning Experiments

### 3A: Direct Fine-tuning (10 experiments)
| Experiment | Steps | Time (hours) |
|------------|-------|--------------|
| 10 tasks × 5,000 steps | 50,000 | 25.0 |
| **Subtotal** | | **25.0 hours** |

### 3B: Multi-task Fine-tuning (10 experiments)
| Experiment | Steps | Time (hours) |
|------------|-------|--------------|
| All sentiment | 10,000 | 5.0 |
| All Q&A | 10,000 | 5.0 |
| Mixed classification | 10,000 | 5.0 |
| Financial + General | 10,000 | 5.0 |
| Math + Financial | 10,000 | 5.0 |
| GLUE multi-task | 10,000 | 5.0 |
| Financial complete | 20,000 | 10.0 |
| Cross-domain | 15,000 | 7.5 |
| Instruction following | 10,000 | 5.0 |
| Advanced reasoning | 15,000 | 7.5 |
| **Subtotal** | | **60.0 hours** |

### 3C: Sequential Fine-tuning (6 sequences)
| Sequence | Total Steps | Time (hours) |
|----------|------------|--------------|
| General→Specific (3 stages) | 6,000 | 3.0 |
| Easy→Hard (4 stages) | 10,000 | 5.0 |
| Pretrain→Finetune (2 stages) | 15,000 | 7.5 |
| Math→Financial (2 stages) | 10,000 | 5.0 |
| Multi-domain (3 stages) | 13,000 | 6.5 |
| GLUE progression (3 stages) | 6,000 | 3.0 |
| **Subtotal** | | **30.0 hours** |

**Phase 3 Total: 115.0 hours**

## Summary

| Phase | Experiments | GPU Hours |
|-------|------------|-----------|
| Phase 1: Baselines | 10 | 1.7 |
| Phase 2: Pretraining | 27 | 225.0 |
| Phase 3: Fine-tuning | 35 | 115.0 |
| **TOTAL** | **72** | **341.7 hours** |

## Cost Estimates

### Cloud GPU Costs (approximate)
- **AWS p3.2xlarge** (V100): $3.06/hour × 341.7 = **$1,046**
- **GCP n1-highmem-8 + V100**: $2.48/hour × 341.7 = **$847**
- **Azure NC6s v3** (V100): $3.06/hour × 341.7 = **$1,046**
- **Lambda Labs** (RTX 4090): $0.59/hour × 341.7 = **$202**

### Local RTX 4090 Estimates
- **Total runtime**: 341.7 hours ≈ **14.2 days** (continuous)
- **With 8 hours/day**: ~43 days
- **Power consumption**: ~450W × 341.7 hours = 154 kWh
- **Electricity cost** (@$0.15/kWh): ~$23

## Optimization Strategies

### To Reduce GPU Hours:

1. **Prioritize High-Impact Experiments**
   - Focus on Phase 3 first (115 hours) for immediate results
   - Skip 50k-step pretraining experiments (save 75 hours)
   - Reduce to 2 data scales instead of 3 (save ~40%)

2. **Efficient Batching**
   - Use gradient accumulation: `--gradient_accumulation_steps 4`
   - Increase batch size if memory allows: `--batch_size 16`
   - Can reduce time by 20-30%

3. **Early Stopping**
   - Monitor validation loss and stop when plateaued
   - Can save 20-40% of training time

4. **Reduced Experiment Set** (Essential only)
   - Phase 1: 10 baselines (1.7 hours)
   - Phase 2: 6 key pretraining (30 hours)
   - Phase 3: 10 key fine-tuning (50 hours)
   - **Reduced Total: ~82 hours**

## Parallel Execution Options

With access to multiple GPUs:
- **2 GPUs**: ~171 hours (7 days)
- **4 GPUs**: ~85 hours (3.5 days)
- **8 GPUs**: ~43 hours (2 days)

## Recommendations

1. **Start with Phase 3** direct fine-tuning for quick results
2. **Use reduced step counts** initially (2k-5k steps) for exploration
3. **Focus on high-value experiments** that directly answer research questions
4. **Consider spot instances** on Lambda Labs for cost-effective computing
5. **Implement checkpointing** to resume interrupted experiments