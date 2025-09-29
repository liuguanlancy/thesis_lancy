# Master's Thesis Experimental Plan: Taxonomy of Domain Adaptation for Small Language Models

## Thesis Title
**Understanding the Training Taxonomy: How Dataset Selection and Training Strategies Enable Small Models to Compete with Large Model Zero-Shot Performance**

## Core Research Questions

### Primary Question
**What is the optimal taxonomy of training strategies (pretraining, continued pretraining, fine-tuning, and their combinations) to enable small models (≤2B parameters) to match large model zero-shot performance on domain-specific tasks?**

### Sub-Questions
1. **RQ1**: Does continued pretraining on domain-specific SFT datasets improve downstream task performance compared to direct fine-tuning?
2. **RQ2**: What is the optimal data mixture and training sequence for domain adaptation?
3. **RQ3**: How does cross-task transfer work in small models with different training strategies?
4. **RQ4**: Can we identify a universal training recipe that works across different task types?
5. **RQ5**: What is the minimum data and compute required to achieve 80% of GPT-4's performance?

## 1. Dataset Taxonomy (26 Available Datasets)

### 1.1 General Pretraining Corpora (5)
Used for general language understanding:
- **Skylion007/openwebtext** - 40GB web text
- **bookcorpus/bookcorpus** - 1B words literature
- **wikitext** - Encyclopedia articles
- **PleIAs/common_corpus** - Multilingual corpus
- **deepmind/math_dataset** - Mathematical language

### 1.2 Financial Domain Texts (8)
Can be used for continued pretraining OR fine-tuning:
- **virattt/financial-qa-10K** - 7K Q&A pairs
- **gbharti/finance-alpaca** - Instruction pairs
- **JanosAudran/financial-reports-sec** - SEC documents
- **LLukas22/fiqa** - Financial Q&A
- **AdaptLLM/finance-tasks** (ConvFinQA) - Conversational
- **FinGPT/fingpt-sentiment-train** - 76K examples
- **yale-nlp/FinanceMath** - Math reasoning
- **sujet-ai/Sujet-Finance-QA-Vision-100k** - Multimodal

### 1.3 Classification Tasks (7)
For supervised fine-tuning and evaluation:
- **takala/financial_phrasebank** - 4.8K sentiment
- **zeroshot/twitter-financial-news-sentiment** - 11.9K tweets
- **zeroshot/twitter-financial-news-topic** - 17K topic classification
- **TheFinAI/fiqa-sentiment-classification** - 1.2K microblogs
- **AdaptLLM/finance-tasks** (FPB, FiQA_SA) - Sentiment variants
- **stanfordnlp/imdb** - General sentiment baseline

### 1.4 Reasoning & Understanding (6)
Complex tasks for evaluation:
- **openai/gsm8k** - Math word problems
- **cais/mmlu** - Multitask understanding
- **TIGER-Lab/MMLU-Pro** - Advanced reasoning
- **glue** - Language understanding suite
- **bigcode/bigcodebench** - Code generation
- **AdaptLLM/finance-tasks** (NER) - Entity recognition

## 2. Comprehensive Experimental Matrix

### 2.1 Base Models (4)
```python
models = {
    "tiny": "gpt2 (124M)",           # Baseline
    "small": "gpt2-medium (355M)",   # Intermediate
    "medium": "Qwen2-0.5B (500M)",   # Modern architecture
    "large": "Gemma-2-2B (2B)",      # Upper bound
}
```

### 2.2 Training Strategies (8)

#### Strategy 1: Baseline (No Training)
```python
zero_shot_baseline = evaluate_without_training(model, task)
```

#### Strategy 2: General Pretraining Only
```python
# Pretrain on general corpora
general_pretrain = pretrain(
    model="gpt2",
    datasets=["openwebtext", "wikitext", "bookcorpus"],
    mixture=[0.5, 0.25, 0.25],
    steps=50000
)
```

#### Strategy 3: Domain Continued Pretraining
```python
# Continue pretraining on financial texts
domain_pretrain = continue_pretrain(
    model="gpt2",
    datasets=["financial-qa-10K", "finance-alpaca", "sec-reports"],
    mode="pretrain",  # Use as text corpus
    steps=20000
)
```

#### Strategy 4: Direct Fine-tuning
```python
# Fine-tune on target task only
direct_finetune = finetune(
    model="gpt2",
    dataset="target_task",
    use_lora=True,
    steps=5000
)
```

#### Strategy 5: Sequential Training (General→Domain→Task)
```python
# Three-stage training
stage1 = pretrain(model, general_corpora, 30000)
stage2 = continue_pretrain(stage1, domain_corpora, 15000)
stage3 = finetune(stage2, target_task, 5000)
```

#### Strategy 6: Multi-task Fine-tuning
```python
# Train on multiple tasks simultaneously
multitask = finetune(
    model="gpt2",
    datasets=["sentiment", "qa", "classification"],
    mixture=[0.4, 0.3, 0.3],
    steps=10000
)
```

#### Strategy 7: Curriculum Learning
```python
# Easy to hard progression
curriculum = [
    ("sentiment_binary", 2000),     # Easy
    ("topic_classification", 3000), # Medium  
    ("financial_qa", 5000),         # Hard
    ("math_reasoning", 5000)        # Hardest
]
```

#### Strategy 8: Mixed Objective Training
```python
# Combine pretraining and fine-tuning objectives
mixed = train(
    model="gpt2",
    pretrain_data=["openwebtext", "financial_texts"],
    finetune_data=["sentiment", "qa"],
    pretrain_weight=0.3,
    finetune_weight=0.7,
    steps=15000
)
```

### 2.3 Evaluation Tasks (10)

#### Core Financial Tasks
1. **Financial Sentiment** (3-class): financial_phrasebank
2. **Tweet Sentiment** (3-class): twitter-financial-sentiment
3. **Topic Classification** (20-class): twitter-financial-topic
4. **Financial Q&A**: financial-qa-10K
5. **Conversational Q&A**: ConvFinQA

#### Transfer Tasks
6. **General Sentiment**: IMDB
7. **Math Reasoning**: GSM8K
8. **General Understanding**: GLUE (SST-2)
9. **Code Generation**: BigCodeBench
10. **Named Entity Recognition**: AdaptLLM/NER

## 3. Detailed Experiment Plan (200+ Experiments)

### 3.1 Phase 1: Baseline Establishment (25 experiments)

#### Zero-shot Baselines
```python
# 4 models × 10 tasks = 40 evaluations
for model in ["gpt2", "gpt2-medium", "qwen2-0.5b", "gemma-2-2b"]:
    for task in evaluation_tasks:
        results[model][task]["zero_shot"] = evaluate(model, task)

# Large model baselines (3 models × 10 tasks × 2 settings)
for model in ["gpt-4", "claude-3", "gemini-pro"]:
    for task in evaluation_tasks:
        results[model][task]["zero_shot"] = api_evaluate(model, task, shots=0)
        results[model][task]["few_shot"] = api_evaluate(model, task, shots=5)
```

### 3.2 Phase 2: Pretraining Experiments (60 experiments)

#### 2A: General Pretraining Effects
```python
# 3 models × 3 corpus combinations × 3 data scales
pretraining_experiments = [
    {
        "model": "gpt2",
        "data": ["openwebtext"],
        "steps": [10k, 25k, 50k],
        "eval": all_tasks
    },
    {
        "model": "gpt2",
        "data": ["openwebtext", "wikitext"],
        "mixture": [0.7, 0.3],
        "steps": [10k, 25k, 50k],
        "eval": all_tasks
    },
    {
        "model": "gpt2",
        "data": ["openwebtext", "wikitext", "bookcorpus"],
        "mixture": [0.5, 0.25, 0.25],
        "steps": [10k, 25k, 50k],
        "eval": all_tasks
    }
]
```

#### 2B: Domain Continued Pretraining
```python
# Using financial datasets as pretraining corpora
# 3 models × 4 dataset combinations × 2 data scales
domain_pretrain_experiments = [
    {
        "base": "gpt2",
        "pretrain_on": "financial-qa-10K",
        "mode": "pretrain",  # Treat Q&A as text
        "steps": [5k, 10k],
        "eval": ["qa_tasks", "sentiment_tasks", "transfer_tasks"]
    },
    {
        "base": "gpt2",
        "pretrain_on": "fingpt-sentiment",
        "mode": "pretrain",  # Treat sentiment as text
        "steps": [10k, 20k],
        "eval": ["sentiment_tasks", "qa_tasks", "transfer_tasks"]
    },
    {
        "base": "gpt2",
        "pretrain_on": ["financial-qa", "fingpt-sentiment", "sec-reports"],
        "mixture": [0.4, 0.3, 0.3],
        "mode": "pretrain",
        "steps": [10k, 20k],
        "eval": all_tasks
    }
]
```

#### 2C: Cross-Domain Pretraining
```python
# Test if math/code pretraining helps financial tasks
cross_domain_experiments = [
    {
        "base": "gpt2",
        "pretrain_on": "gsm8k",
        "mode": "pretrain",
        "steps": 10000,
        "eval": ["financial_math", "financial_qa", "general_math"]
    },
    {
        "base": "gpt2",
        "pretrain_on": "bigcodebench",
        "mode": "pretrain",
        "steps": 10000,
        "eval": ["code_gen", "financial_qa", "reasoning"]
    }
]
```

### 3.3 Phase 3: Fine-tuning Experiments (80 experiments)

#### 3A: Direct Fine-tuning
```python
# 4 models × 5 target tasks × 2 methods (full/LoRA)
direct_finetune_experiments = []
for model in ["gpt2", "gpt2-medium", "qwen2-0.5b", "gemma-2b"]:
    for task in ["sentiment", "qa", "topic", "ner", "math"]:
        for method in ["full", "lora"]:
            exp = {
                "model": model,
                "task": task,
                "method": method,
                "lora_r": 16 if method == "lora" else None,
                "steps": 5000,
                "eval": all_tasks  # Test on all to measure transfer
            }
            direct_finetune_experiments.append(exp)
```

#### 3B: Multi-task Fine-tuning
```python
# Different task combinations
multitask_experiments = [
    {
        "name": "all_sentiment",
        "datasets": ["financial_phrasebank", "twitter_sentiment", "fingpt_sentiment"],
        "mixture": [0.3, 0.3, 0.4],
        "steps": 10000
    },
    {
        "name": "all_qa",
        "datasets": ["financial-qa-10k", "convfinqa", "fiqa"],
        "mixture": [0.4, 0.3, 0.3],
        "steps": 10000
    },
    {
        "name": "mixed_types",
        "datasets": ["sentiment", "qa", "classification", "ner"],
        "mixture": [0.25, 0.25, 0.25, 0.25],
        "steps": 15000
    },
    {
        "name": "financial_complete",
        "datasets": all_financial_datasets,
        "mixture": "uniform",
        "steps": 20000
    }
]
```

#### 3C: Sequential Fine-tuning
```python
# Test different training orders
sequential_experiments = [
    {
        "name": "general_to_specific",
        "sequence": [
            ("imdb", 2000),
            ("financial_sentiment", 2000),
            ("twitter_sentiment", 2000)
        ]
    },
    {
        "name": "easy_to_hard",
        "sequence": [
            ("binary_sentiment", 2000),
            ("3way_sentiment", 2000),
            ("20way_topic", 3000),
            ("financial_qa", 3000)
        ]
    },
    {
        "name": "pretrain_then_finetune",
        "sequence": [
            ("financial_texts", 10000, "pretrain"),
            ("target_task", 5000, "finetune")
        ]
    }
]
```

### 3.4 Phase 4: Combined Strategies (40 experiments)

#### 4A: Full Pipeline Experiments
```python
# Test complete training pipelines
pipeline_experiments = [
    {
        "name": "minimal_pipeline",
        "stages": [
            ("general_pretrain", "openwebtext", 10000),
            ("task_finetune", "target", 2000)
        ]
    },
    {
        "name": "standard_pipeline",
        "stages": [
            ("general_pretrain", "openwebtext+wiki", 20000),
            ("domain_pretrain", "financial_texts", 10000),
            ("task_finetune", "target", 5000)
        ]
    },
    {
        "name": "comprehensive_pipeline",
        "stages": [
            ("general_pretrain", "all_general", 30000),
            ("domain_pretrain", "all_financial", 15000),
            ("multitask_finetune", "related_tasks", 5000),
            ("target_finetune", "target", 2000)
        ]
    }
]
```

#### 4B: Mixture Strategies
```python
# Test different mixture ratios during training
mixture_experiments = [
    {
        "name": "pretrain_heavy",
        "pretrain_data": 0.8,
        "finetune_data": 0.2,
        "steps": 15000
    },
    {
        "name": "balanced",
        "pretrain_data": 0.5,
        "finetune_data": 0.5,
        "steps": 15000
    },
    {
        "name": "finetune_heavy",
        "pretrain_data": 0.2,
        "finetune_data": 0.8,
        "steps": 15000
    }
]
```

### 3.5 Phase 5: Ablation Studies (40 experiments)

#### 5A: Data Quantity Ablations
```python
# How much data is really needed?
data_ablations = []
for dataset in ["financial-qa", "sentiment", "topic_class"]:
    for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
        for strategy in ["pretrain", "finetune"]:
            ablation = {
                "dataset": dataset,
                "fraction": fraction,
                "strategy": strategy,
                "steps": int(5000 * fraction),
                "eval": relevant_tasks
            }
            data_ablations.append(ablation)
```

#### 5B: Model Size Ablations
```python
# When does model size matter?
size_ablations = []
for model_size in [124M, 355M, 500M, 1B, 2B]:
    for training in ["none", "pretrain", "finetune", "both"]:
        ablation = {
            "model": get_model_by_size(model_size),
            "training": training,
            "eval": all_tasks
        }
        size_ablations.append(ablation)
```

#### 5C: Training Duration Ablations
```python
# Diminishing returns analysis
duration_ablations = []
for steps in [100, 500, 1000, 2500, 5000, 10000, 20000]:
    for strategy in ["pretrain", "finetune"]:
        ablation = {
            "steps": steps,
            "strategy": strategy,
            "checkpoint_every": min(100, steps//10),
            "eval": core_tasks
        }
        duration_ablations.append(ablation)
```

#### 5D: LoRA Rank Ablations
```python
# Optimal LoRA configuration
lora_ablations = []
for rank in [1, 2, 4, 8, 16, 32, 64]:
    for alpha_ratio in [0.5, 1.0, 2.0]:
        ablation = {
            "lora_r": rank,
            "lora_alpha": int(rank * alpha_ratio),
            "eval": ["performance", "trainable_params", "memory_usage"]
        }
        lora_ablations.append(ablation)
```

## 4. Evaluation Protocol

### 4.1 Metrics Framework

#### Classification Metrics
```python
classification_metrics = {
    "accuracy": overall_accuracy,
    "f1_macro": macro_averaged_f1,
    "f1_weighted": weighted_f1,
    "precision_recall": per_class_pr,
    "confusion_matrix": full_confusion,
    "calibration": ece_score,
    "robustness": adversarial_accuracy
}
```

#### Generation Metrics
```python
generation_metrics = {
    "exact_match": exact_string_match,
    "f1_token": token_level_f1,
    "bleu": bleu_scores[1,2,3,4],
    "rouge": rouge_l_score,
    "bertscore": semantic_similarity,
    "factuality": fact_checking_score,
    "coherence": perplexity_score
}
```

#### Efficiency Metrics
```python
efficiency_metrics = {
    "inference_speed": tokens_per_second,
    "memory_usage": peak_ram_gb,
    "model_size": parameter_count,
    "training_time": gpu_hours,
    "energy": kwh_consumed,
    "cost": dollar_estimate
}
```

### 4.2 Statistical Analysis

#### Significance Testing
```python
# Ensure results are statistically significant
statistical_tests = {
    "paired_t_test": compare_two_methods,
    "anova": compare_multiple_methods,
    "wilcoxon": non_parametric_comparison,
    "bootstrap_ci": confidence_intervals,
    "effect_size": cohens_d
}
```

#### Variance Analysis
```python
# Multiple runs with different seeds
variance_analysis = {
    "seeds": [42, 1337, 2024, 3141, 9999],
    "report": ["mean", "std", "min", "max", "percentiles"]
}
```

## 5. Implementation Timeline (12 Weeks)

### Weeks 1-2: Infrastructure Setup
- [ ] Implement evaluation framework
- [ ] Set up experiment tracking (Weights & Biases)
- [ ] Create data pipelines for all 26 datasets
- [ ] Establish compute resources (Colab Pro, Kaggle)
- [ ] Baseline measurements (all models, all tasks)

### Weeks 3-4: Pretraining Experiments
- [ ] General pretraining (20 experiments)
- [ ] Domain continued pretraining (20 experiments)
- [ ] Cross-domain pretraining (10 experiments)
- [ ] Analyze pretraining effects

### Weeks 5-6: Fine-tuning Experiments
- [ ] Direct fine-tuning (30 experiments)
- [ ] Multi-task fine-tuning (20 experiments)
- [ ] Sequential fine-tuning (15 experiments)
- [ ] Compare with pretraining results

### Weeks 7-8: Combined Strategies
- [ ] Full pipeline experiments (15 experiments)
- [ ] Mixture strategies (10 experiments)
- [ ] Curriculum learning (10 experiments)
- [ ] Identify optimal combinations

### Weeks 9-10: Ablation Studies
- [ ] Data quantity ablations (15 experiments)
- [ ] Model size ablations (10 experiments)
- [ ] Training duration analysis (10 experiments)
- [ ] LoRA configuration optimization (10 experiments)

### Weeks 11-12: Analysis and Writing
- [ ] Statistical significance testing
- [ ] Create visualization and plots
- [ ] Error analysis and failure modes
- [ ] Write thesis chapters
- [ ] Prepare defense presentation

## 6. Expected Contributions

### 6.1 Scientific Contributions

1. **Comprehensive Taxonomy**: First systematic study of how different training strategies affect small model performance on financial tasks

2. **Pretraining on SFT Data**: Novel approach of using task-specific datasets as pretraining corpora

3. **Transfer Learning Map**: Complete understanding of cross-task and cross-domain transfer in small models

4. **Optimal Training Recipe**: Evidence-based guidelines for training small domain-specific models

5. **Efficiency Analysis**: Quantification of performance/compute tradeoffs

### 6.2 Practical Contributions

1. **Training Playbook**: Step-by-step guide for practitioners
2. **Cost Analysis**: Real costs of achieving target performance
3. **Model Zoo**: Released checkpoints for all trained models
4. **Evaluation Suite**: Comprehensive benchmark for financial NLP
5. **Reproducible Pipeline**: Full code and configurations

## 7. Success Metrics

### Primary Success Criteria
- [ ] At least one training strategy achieves ≥80% of GPT-4 zero-shot performance
- [ ] Demonstrate >50% improvement over zero-shot small models
- [ ] Complete at least 150 experiments with full evaluation
- [ ] Identify clear patterns in training taxonomy

### Secondary Success Criteria
- [ ] Show that continued pretraining improves downstream performance
- [ ] Demonstrate cross-domain transfer benefits
- [ ] Achieve results with <$100 in compute costs
- [ ] Publish findings in ACL/EMNLP/NeurIPS

## 8. Risk Mitigation

### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Compute limitations | High | High | Use efficient training (LoRA), checkpoint frequently |
| Poor performance | High | Medium | Have multiple strategies, extensive hyperparameter search |
| Data quality issues | Medium | Low | Validate datasets early, have backups |
| Reproducibility | Medium | Medium | Set seeds, log everything, version control |

### Timeline Risks
- **Buffer time**: 2 weeks built into schedule
- **Parallel execution**: Run experiments concurrently
- **Early validation**: Test pipeline with small experiments first
- **Incremental writing**: Document as you go

## 9. Detailed Experiment Tracking

### Experiment Logging Schema
```json
{
    "experiment_id": "uuid",
    "timestamp": "2024-01-15T10:30:00",
    "model": {
        "name": "gpt2",
        "size": 124000000,
        "checkpoint": "path/to/checkpoint"
    },
    "training": {
        "strategy": "pretrain|finetune|both",
        "datasets": ["dataset1", "dataset2"],
        "mixture_rates": [0.6, 0.4],
        "steps": 10000,
        "hyperparameters": {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation": 4,
            "warmup_steps": 500,
            "lora_config": {...}
        }
    },
    "evaluation": {
        "tasks": ["task1", "task2"],
        "metrics": {
            "task1": {"accuracy": 0.85, "f1": 0.83},
            "task2": {"bleu": 0.45, "rouge": 0.52}
        }
    },
    "resources": {
        "gpu_hours": 2.5,
        "peak_memory_gb": 12.3,
        "cost_usd": 0.0
    },
    "notes": "Observations and insights"
}
```

### Results Aggregation
```python
# Aggregate results across all experiments
results_matrix = pd.DataFrame({
    "model": [...],
    "training_strategy": [...],
    "pretrain_data": [...],
    "finetune_data": [...],
    "task": [...],
    "metric": [...],
    "value": [...],
    "std": [...],
    "compute_hours": [...],
})

# Generate comprehensive reports
generate_latex_tables(results_matrix)
generate_plots(results_matrix)
generate_statistical_tests(results_matrix)
```

## 10. Thesis Outline

### Chapter 1: Introduction
- Motivation: Small models for practical deployment
- Research questions and hypotheses
- Contributions and thesis structure

### Chapter 2: Background and Related Work
- Language model pretraining and fine-tuning
- Domain adaptation techniques
- Financial NLP tasks and benchmarks
- Small model optimization

### Chapter 3: Methodology
- Dataset taxonomy and selection
- Training strategies framework
- Evaluation protocol
- Experimental design

### Chapter 4: Baseline Experiments
- Zero-shot performance analysis
- Large model baselines
- Performance gaps identification

### Chapter 5: Pretraining Studies
- General pretraining effects
- Domain continued pretraining
- Novel: SFT data as pretraining corpus
- Cross-domain transfer

### Chapter 6: Fine-tuning Studies
- Direct fine-tuning analysis
- Multi-task learning
- Sequential training
- Curriculum learning

### Chapter 7: Combined Strategies
- Full training pipelines
- Mixture strategies
- Optimal combinations
- Cost-benefit analysis

### Chapter 8: Ablation Studies
- Data quantity requirements
- Model size effects
- Training duration analysis
- Hyperparameter sensitivity

### Chapter 9: Discussion
- Key findings synthesis
- Training taxonomy guidelines
- Practical recommendations
- Limitations and future work

### Chapter 10: Conclusion
- Research questions answered
- Contributions summary
- Broader implications
- Future research directions

## Conclusion

This comprehensive experimental plan explores the full taxonomy of training strategies for small language models, with 200+ carefully designed experiments that will provide definitive answers about how to effectively adapt small models for domain-specific tasks. The systematic approach, extensive ablations, and rigorous evaluation will contribute both scientific understanding and practical guidelines for the field.

The key innovation is treating this as a taxonomy problem: understanding not just whether small models can compete with large ones, but exactly which training strategies, data combinations, and sequences enable this performance, providing a complete map of the training landscape for resource-constrained domain adaptation.