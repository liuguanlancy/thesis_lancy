# Experimental Plan: Small Model Performance Through Domain-Specific Training

## Research Question

**Can pretraining or fine-tuning small models (≤2B parameters) on specific datasets help them match large models' zero/few-shot performance?**

This research investigates whether domain-specific training can bridge the performance gap between small, specialized models and large general-purpose models like GPT-4 and Claude.

## Experimental Design Overview

### Core Hypothesis
Small models (GPT-2, Qwen2-0.5B) trained on domain-specific data can compete with large model zero/few-shot performance through:
1. **Continued pretraining** on SFT datasets (using them as domain corpora)
2. **Fine-tuning with LoRA** for parameter-efficient adaptation
3. **Dataset mixture training** for multi-task robustness

### Research Framework
```
Small Model + Domain Training → Performance ≈ Large Model Zero/Few-shot
```

## Available Datasets (26 Total)

### Classification Tasks (9 datasets)
1. **IMDB** (`stanfordnlp/imdb`) - Binary sentiment
2. **GLUE** (`glue`) - 8 configs: cola, sst2, mrpc, qqp, mnli, qnli, rte, wnli
3. **Financial PhraseBank** (`takala/financial_phrasebank`) - 3-class sentiment
4. **Twitter Financial** (`zeroshot/twitter-financial-news-sentiment`) - 3-class sentiment  
5. **Twitter Financial Topics** (`zeroshot/twitter-financial-news-topic`) - 20-class topics
6. **FiQA Sentiment** (`TheFinAI/fiqa-sentiment-classification`) - 3-class sentiment
7. **FinGPT Sentiment** (`FinGPT/fingpt-sentiment-train`) - Instruction-based sentiment
8. **MMLU** (`cais/mmlu`) - 57 subjects (if accessible)
9. **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`) - Advanced reasoning

### Generative/Q&A Tasks (8 datasets)
10. **FiQA Q&A** (`LLukas22/fiqa`) - Financial Q&A
11. **Finance Alpaca** (`gbharti/finance-alpaca`) - Instruction following
12. **Financial QA 10-K** (`virattt/financial-qa-10K`) - 10-K filings Q&A
13. **SEC Reports** (`JanosAudran/financial-reports-sec`) - Financial documents
14. **GSM8K** (`openai/gsm8k`) - Grade school math
15. **FinanceMath** (`yale-nlp/FinanceMath`) - Financial mathematics (if accessible)
16. **DeepMind Math** (`deepmind/math_dataset`) - Mathematical reasoning
17. **BigCodeBench** (`bigcode/bigcodebench`) - Code generation

### AdaptLLM Finance Suite (4 configs)
18. **AdaptLLM ConvFinQA** (`AdaptLLM/finance-tasks` config: ConvFinQA)
19. **AdaptLLM FPB** (`AdaptLLM/finance-tasks` config: FPB)
20. **AdaptLLM FiQA_SA** (`AdaptLLM/finance-tasks` config: FiQA_SA)
21. **AdaptLLM NER** (`AdaptLLM/finance-tasks` config: NER)

### Pretraining Corpora (5 datasets)
22. **Common Corpus** (`PleIAs/common_corpus`) - 2T tokens multilingual
23. **OpenWebText** (`Skylion007/openwebtext`) - Web content
24. **BookCorpus** (`bookcorpus/bookcorpus`) - Literary texts
25. **OpenWebMath** (`open-web-math/open-web-math`) - Mathematical web content
26. **WikiText** (`wikitext`) - Wikipedia articles

## Experimental Design

### Phase 1: Baseline Establishment
**Goal**: Establish performance baselines for comparison

#### 1.1 Large Model Baselines (External)
- **Zero-shot**: GPT-4, Claude-3.5-Sonnet on key evaluation tasks
- **Few-shot**: Same models with 1-5 examples per task
- **Task Selection**: Focus on 8-10 representative tasks across domains

#### 1.2 Small Model Baselines  
- **Models**: GPT-2 (124M), GPT-2-medium (355M), Qwen2-0.5B
- **Evaluation**: Zero-shot performance on all classification tasks
- **Command**: `python main.py --model gpt2 --dataset [DATASET] --mode sft --max_steps 0 --evaluation_only`

### Phase 2: Continued Pretraining Experiments
**Goal**: Test whether using SFT datasets as pretraining corpora improves performance

#### 2.1 Domain-Specific Continued Pretraining
**Hypothesis**: Continued pretraining on domain data improves downstream performance

**Financial Domain Pretraining**:
```bash
# Pretrain on financial datasets
python main.py --model gpt2 --dataset gbharti/finance-alpaca --mode pretrain --max_steps 2000
python main.py --model gpt2 --dataset LLukas22/fiqa --mode pretrain --max_steps 1000
python main.py --model gpt2 --dataset JanosAudran/financial-reports-sec --mode pretrain --max_steps 1500
```

**Mathematical Domain Pretraining**:
```bash
# Pretrain on mathematical content
python main.py --model gpt2 --dataset open-web-math/open-web-math --mode pretrain --max_steps 2000
python main.py --model gpt2 --dataset openai/gsm8k --mode pretrain --max_steps 800
```

**General Domain Pretraining**:
```bash
# Pretrain on general text for control
python main.py --model gpt2 --dataset wikitext --dataset_config wikitext-103-raw-v1 --mode pretrain --max_steps 2000
```

#### 2.2 Mixed Domain Pretraining
```bash
# Mix financial datasets for pretraining
python main.py --model gpt2 --datasets gbharti/finance-alpaca LLukas22/fiqa --dataset_configs None None --mixture_rates 0.7 0.3 --mode pretrain --max_steps 2000

# Mix mathematical datasets
python main.py --model gpt2 --datasets openai/gsm8k open-web-math/open-web-math --dataset_configs main None --mixture_rates 0.4 0.6 --mode pretrain --max_steps 2000
```

### Phase 3: Fine-tuning Experiments  
**Goal**: Compare fine-tuning approaches with pretraining results

#### 3.1 Standard Fine-tuning
**Individual Task Fine-tuning**:
```bash
# Financial sentiment analysis
python main.py --model gpt2 --dataset zeroshot/twitter-financial-news-sentiment --mode sft --max_steps 1000

# Mathematical reasoning  
python main.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode sft --max_steps 800

# General sentiment
python main.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --max_steps 1000
```

#### 3.2 Multi-task Fine-tuning
```bash
# Financial multi-task
python main.py --model gpt2 --datasets zeroshot/twitter-financial-news-sentiment takala/financial_phrasebank --dataset_configs None None --mixture_rates 0.5 0.5 --mode sft --max_steps 1200

# GLUE multi-task  
python main.py --model gpt2 --datasets glue glue glue --dataset_configs sst2 cola mrpc --mixture_rates 0.4 0.3 0.3 --mode sft --max_steps 1200
```

#### 3.3 Sequential Training (Pretrain → SFT)
```bash
# First pretrain on domain, then fine-tune on specific task
# 1. Domain pretraining
python main.py --model gpt2 --dataset gbharti/finance-alpaca --mode pretrain --max_steps 1000 --output_dir ./checkpoints/gpt2_finance_pretrain

# 2. Task fine-tuning  
python main.py --model ./checkpoints/gpt2_finance_pretrain --dataset zeroshot/twitter-financial-news-sentiment --mode sft --max_steps 500
```

### Phase 4: Evaluation Framework

#### 4.1 Task-Specific Evaluation
**Financial Tasks**:
- Sentiment: Financial PhraseBank, Twitter Financial, FiQA Sentiment
- Q&A: Financial QA 10-K, AdaptLLM ConvFinQA
- Classification: Twitter Financial Topics (20-class)

**Mathematical Tasks**:
- Problem Solving: GSM8K, DeepMind Math
- Advanced Reasoning: MMLU-Pro mathematical sections

**General Tasks**:
- Sentiment: IMDB, GLUE SST-2
- Grammar: GLUE CoLA
- Paraphrase: GLUE MRPC

#### 4.2 Cross-Domain Transfer
**Transfer Learning Tests**:
```bash
# Train on one domain, test on another
# Financial → General
python main.py --model gpt2 --dataset gbharti/finance-alpaca --mode pretrain --max_steps 1000
# Then evaluate on IMDB

# Mathematical → General
python main.py --model gpt2 --dataset openai/gsm8k --mode pretrain --max_steps 800  
# Then evaluate on GLUE tasks
```

### Phase 5: Ablation Studies

#### 5.1 Dataset Size Effects
- Compare training with 25%, 50%, 75%, 100% of available data
- Measure performance scaling with dataset size

#### 5.2 Model Size Comparison
- GPT-2 (124M) vs GPT-2-medium (355M) vs Qwen2-0.5B
- Analyze parameter efficiency vs performance trade-offs

#### 5.3 Training Approach Comparison
**Direct Comparison Matrix**:
| Approach | Financial | Mathematical | General | Cross-Domain |
|----------|-----------|--------------|---------|--------------|
| Zero-shot baseline | ✓ | ✓ | ✓ | ✓ |
| Continued pretraining | ✓ | ✓ | ✓ | ✓ |  
| Direct fine-tuning | ✓ | ✓ | ✓ | ✓ |
| Sequential training | ✓ | ✓ | ✓ | ✓ |
| Multi-task training | ✓ | ✓ | ✓ | ✓ |

## Resource Constraints & Optimization

### Hardware Requirements
- **Primary**: Free Google Colab GPUs (T4, V100 when available)
- **Secondary**: Local CPU training for smaller experiments
- **Backup**: Free Kaggle GPU hours

### Training Efficiency
```bash
# Small batch sizes for memory efficiency
--batch_size 4 --gradient_accumulation_steps 4

# Shorter sequences for speed
--max_length 256

# Limited training steps for resource conservation  
--max_steps 1000

# Efficient checkpointing
--save_steps 200 --save_total_limit 3
```

### Model Selection for Resource Efficiency
- **Primary**: GPT-2 (124M) - fastest training
- **Secondary**: GPT-2-medium (355M) - balance of size/performance
- **Tertiary**: Qwen2-0.5B - modern architecture comparison

## Expected Outcomes & Metrics

### Success Criteria
1. **Domain Matching**: Small model performance on domain tasks ≥ 80% of large model zero-shot
2. **Transfer Learning**: Cross-domain performance ≥ 60% of large model zero-shot  
3. **Efficiency**: Training time < 2 hours per experiment on free GPUs

### Key Metrics
- **Classification**: Accuracy, F1-score, precision, recall
- **Generation**: BLEU, ROUGE, semantic similarity (where applicable)
- **Efficiency**: Training time, memory usage, parameters/performance ratio

### Research Questions to Answer
1. Does continued pretraining on domain data outperform direct fine-tuning?
2. Which domains benefit most from specialized training?
3. How does model size affect the benefits of domain-specific training?
4. Can small models achieve competitive performance with proper training?
5. What is the optimal balance between pretraining and fine-tuning steps?

## Implementation Timeline

### Week 1-2: Infrastructure & Baselines
- Set up evaluation framework
- Establish large model baselines (external API calls)
- Run small model zero-shot baselines

### Week 3-4: Continued Pretraining Experiments  
- Domain-specific pretraining experiments
- Mixed-domain pretraining
- Initial evaluation and comparison

### Week 5-6: Fine-tuning Experiments
- Standard fine-tuning across tasks
- Multi-task fine-tuning experiments
- Sequential training (pretrain → SFT)

### Week 7-8: Analysis & Ablation
- Cross-domain transfer experiments
- Dataset size and model size ablations
- Performance analysis and comparison

### Week 9-10: Documentation & Reporting
- Comprehensive results analysis
- Research paper draft
- Code documentation and reproducibility

## Dataset Usage Examples

### High-Priority Experiments
```bash
# Core financial sentiment experiment
python main.py --model gpt2 --dataset zeroshot/twitter-financial-news-sentiment --mode pretrain --max_steps 1000
python main.py --model gpt2 --dataset zeroshot/twitter-financial-news-sentiment --mode sft --max_steps 500

# Core mathematical reasoning experiment  
python main.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode pretrain --max_steps 800
python main.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode sft --max_steps 400

# Core general sentiment experiment
python main.py --model gpt2 --dataset stanfordnlp/imdb --mode pretrain --max_steps 1000  
python main.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --max_steps 500
```

### Multi-Domain Mixture Experiments
```bash
# Financial domain mixture
python main.py --model gpt2 --datasets zeroshot/twitter-financial-news-sentiment takala/financial_phrasebank FinGPT/fingpt-sentiment-train --dataset_configs None None None --mixture_rates 0.4 0.3 0.3 --mode sft

# Mathematical + Financial mixture  
python main.py --model gpt2 --datasets openai/gsm8k LLukas22/fiqa --dataset_configs main None --mixture_rates 0.6 0.4 --mode sft

# GLUE multi-task baseline
python main.py --model gpt2 --datasets glue glue glue --dataset_configs sst2 cola mrpc --mixture_rates 0.5 0.3 0.2 --mode sft
```

## Hypothesis Validation Plan

### Primary Hypothesis: Domain-Specific Training Effectiveness
**H1**: Small models with domain-specific pretraining will outperform zero-shot small models
- **Test**: Compare pretrained vs zero-shot performance on financial/mathematical tasks
- **Metric**: Accuracy improvement ≥ 20%

### Secondary Hypothesis: Training Method Comparison  
**H2**: Continued pretraining will outperform direct fine-tuning
- **Test**: Compare pretrain → SFT vs direct SFT on same tasks
- **Metric**: Performance difference and training stability

### Tertiary Hypothesis: Cross-Domain Transfer
**H3**: Domain pretraining will improve performance even on out-of-domain tasks
- **Test**: Train on financial data, evaluate on general tasks
- **Metric**: Transfer performance ≥ baseline + 10%

## Success Metrics

### Quantitative Goals
1. **Domain Performance**: Small model reaches ≥80% of GPT-4 zero-shot performance
2. **Training Efficiency**: Results achieved within resource constraints (≤2 hours/experiment)
3. **Generalization**: Cross-domain transfer shows ≥10% improvement over baseline
4. **Reproducibility**: All experiments documented with exact commands and results

### Qualitative Insights
1. Identification of which domains benefit most from specialized training
2. Understanding of optimal training strategies for small models
3. Analysis of parameter efficiency vs performance trade-offs
4. Documentation of practical considerations for domain adaptation

This experimental plan provides a systematic approach to testing whether small models can compete with large models through domain-specific training, using only the 26 datasets available in the repository and realistic resource constraints.