# Experiment Tracking Table

## Server Assignment and Status Tracking
**Last Updated:** 2025-09-21

### Summary Statistics
- **Total Experiments:** 30 (10 experiments × 3 model sizes)
- **Models:** Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-4B-Base
- **Servers:** Multiple GPU servers accessed via SSH

---

## 📊 Experiment Status Table

| Exp # | Dataset/Config | Model | Server | Status | GPU | Notes |
|:-----:|---------------|:-----:|--------|:------:|:---:|-------|
| **#1** | **Financial Q&A** | | | | | |
| 1.1 | Financial Q&A | 0.6B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 1.2 | Financial Q&A | 1.7B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 1.3 | Financial Q&A | 4B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| **#2** | **FinGPT Sentiment** | | | | | |
| 2.1 | FinGPT Sentiment | 0.6B | ssh ubuntu@129.153.73.125 | ✅ Finished | - | |
| 2.2 | FinGPT Sentiment | 1.7B | ssh ubuntu@129.153.73.125 | ✅ Finished | - | |
| 2.3 | FinGPT Sentiment | 4B | ssh ubuntu@129.153.73.125 | ✅ Finished | - | |
| **#3** | **Finance Alpaca** | | | | | |
| 3.1 | Finance Alpaca | 0.6B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 3.2 | Finance Alpaca | 1.7B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 3.3 | Finance Alpaca | 4B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| **#4** | **FiQA** | | | | | |
| 4.1 | FiQA | 0.6B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| 4.2 | FiQA | 1.7B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| 4.3 | FiQA | 4B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| **#5** | **Twitter Financial** | | | | | |
| 5.1 | Twitter Financial | 0.6B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| 5.2 | Twitter Financial | 1.7B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| 5.3 | Twitter Financial | 4B | ssh ubuntu@104.171.202.235 | ✅ Finished | A6000 | |
| **#6** | **SEC Reports** | | | | | |
| 6.1 | SEC Reports | 0.6B | ssh ubuntu@129.153.73.125 | 🔄 Running | A100 | |
| 6.2 | SEC Reports | 1.7B | ssh ubuntu@129.153.73.125 | 🔄 Running | A100 | |
| 6.3 | SEC Reports | 4B | ssh ubuntu@129.153.73.125| 🔄 Running | A100 | |
| **#7** | **News Articles** | | | | | |
| 7.1 | News Articles | 0.6B | ssh ubuntu@104.171.202.235 | 🔄 Running | 2×A6000 | |
| 7.2 | News Articles | 1.7B | ssh ubuntu@104.171.202.235 | 🔄 Running | 2×A6000 | |
| 7.3 | News Articles | 4B | ssh ubuntu@104.171.202.235 | 🔄 Running | 2×A6000 | |
| **#8** | **WikiText** | | | | | |
| 8.1 | WikiText | 0.6B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 8.2 | WikiText | 1.7B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| 8.3 | WikiText | 4B | ssh ubuntu@129.153.73.125 | ✅ Finished | A100 | |
| **#9** | **Mixed (7 Financial)** | | | | | |
| 9.1 | Mixed Financial | 0.6B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |
| 9.2 | Mixed Financial | 1.7B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |
| 9.3 | Mixed Financial | 4B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |
| **#10** | **Mixed-Wiki (7 Fin + Wiki)** | | | | | |
| 10.1 | Mixed-Wiki | 0.6B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |
| 10.2 | Mixed-Wiki | 1.7B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |
| 10.3 | Mixed-Wiki | 4B | ssh ubuntu@104.171.202.235 | ✅ Finished | 2×A6000 | |

---

## 📋 Status Legend

| Symbol | Status | Description |
|:------:|--------|-------------|
| ⬜ | **Not Started** | Experiment not yet initiated |
| 🔄 | **Running** | Currently executing on server |
| ✅ | **Finished** | Successfully completed |
| ❌ | **Failed** | Encountered error (see Notes) |
| ⏸️ | **Paused** | Temporarily stopped |
| 🔁 | **Restarting** | Being rerun due to issue |

## 📈 Progress Summary

| Category | Total | Finished | Running | Not Started |
|:--------:|:-----:|:--------:|:-------:|:-----------:|
| **Single Datasets** | 21 | 9 (43%) | 6 (29%) | 6 (29%) |
| **Mixed Datasets** | 9 | 6 (67%) | 0 (0%) | 3 (33%) |
| **Total** | **30** | **15 (50%)** | **6 (20%)** | **9 (30%)** |

---

## Server Details

### Available Servers
| Server ID | SSH Command | GPU Type | Memory | Current Load | Notes |
|-----------|------------|----------|--------|--------------|-------|
| Server 1 | `ssh ubuntu@_______________` | | | | |
| Server 2 | `ssh ubuntu@_______________` | | | | |
| Server 3 | `ssh ubuntu@_______________` | | | | |
| Server 4 | `ssh ubuntu@_______________` | | | | |
| Server 5 | `ssh ubuntu@_______________` | | | | |

---

## Execution Scripts

### Single Dataset Experiments (1-8)
- **WikiText Experiments**: `scripts/run_wikitext_experiments.sh`
- **Alpaca Experiments**: `scripts/run_alpaca_experiments.sh`

### Mixed Dataset Experiments (9-10)
- **Mixed (GPU 0)**: `scripts/run_mixed_gpu0.sh`
- **Mixed-Wiki (GPU 1)**: `scripts/run_mixed_wiki_gpu1.sh`

---

## Training Parameters (All Experiments)

```bash
# Common Parameters
--mode pretrain
--batch_size 4
--gradient_accumulation_steps 1
--max_length 1024
--learning_rate 2e-5
--lr_scheduler_type cosine
--warmup_steps 200
--weight_decay 0.01
--bf16
--max_steps 4000
--eval_steps 100
--save_steps 1000
--logging_steps 10
--use_lora
--lora_r 32
--lora_alpha 64
--lora_target_modules q_proj k_proj v_proj o_proj
--attn_implementation flash_attention_2
--eval_max_batches 100
--ddp_find_unused_parameters false
```

---

## Mixture Rates (Experiments 9-10)

### Experiment 9: Mixed Financial (7 datasets)
| Dataset | Token Weight |
|---------|-------------|
| Financial Q&A | 0.024 |
| FinGPT Sentiment | 0.058 |
| Finance Alpaca | 0.083 |
| FiQA | 0.054 |
| Twitter Financial | 0.015 |
| SEC Reports | 0.081 |
| News Articles | 0.685 |

### Experiment 10: Mixed-Wiki (8 datasets)
| Dataset | Token Weight |
|---------|-------------|
| Financial Q&A | 0.024 |
| FinGPT Sentiment | 0.058 |
| Finance Alpaca | 0.083 |
| FiQA | 0.054 |
| Twitter Financial | 0.015 |
| SEC Reports | 0.081 |
| News Articles | 0.397 |
| WikiText | 0.288 |

---

## Estimated Runtime

| Model Size | Steps | Estimated Time | Memory Usage |
|------------|-------|----------------|--------------|
| 0.6B | 4000 | ~2.5 hours | ~8 GB |
| 1.7B | 4000 | ~4 hours | ~16 GB |
| 4B | 4000 | ~8 hours | ~28 GB |

---

## Notes Section

### General Notes
- All experiments use LoRA fine-tuning (rank=32, alpha=64)
- Flash Attention 2 enabled on RTX4090 servers
- Eager attention used on M1 Max machines
- Evaluation runs every 100 steps on all datasets
- Checkpoints saved every 1000 steps

### Issues Encountered
-

### Server-Specific Notes
-

---

## Quick Commands

### Check Experiment Status
```bash
# On server
cd ~/lancy/lancy_thesis/runs/phase2b_financial_qwen3_[model_size]/[dataset]/logs
tail -f training.log
```

### Monitor GPU Usage
```bash
nvidia-smi -l 1  # RTX4090 servers
```

### Resume Failed Experiment
```bash
# Add --resume_from_checkpoint latest to command
```