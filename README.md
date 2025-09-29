# Lancy Thesis - HuggingFace Training Pipeline

A modular, flexible training pipeline for HuggingFace models supporting multiple datasets, tasks, and training modes with a **generative approach** to classification tasks.

## 📚 Table of Contents

- [🚀 Key Features](#-key-features)
- [🎯 Generative Approach](#-generative-approach)
- [📋 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📊 Comprehensive Dataset Support](#-comprehensive-dataset-support)
- [🔧 Configuration Options](#-configuration-options)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [💾 Checkpointing and Resumption](#-checkpointing-and-resumption)
- [🎯 Examples](#-examples)
- [🔍 How It Works](#-how-it-works)
- [🔄 Dataset Mixture Training](#-dataset-mixture-training)
- [🚀 Advanced Usage](#-advanced-usage)
- [📈 Monitoring](#-monitoring)
- [🖥️ Multi-GPU Training](#️-multi-gpu-training)
- [🤖 Reinforcement Learning with GRPO](#-reinforcement-learning-with-grpo)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Key Features

- **MOE Model Support**: Full support for Mixture of Experts models (Mixtral, Switch Transformers, DeepSeek-MOE) with automatic detection and optimization
- **FlashAttention 2**: 2-3x faster attention computation on CUDA GPUs with automatic fallback on MPS/CPU
- **Sequence Packing**: 3-5x training speedup on short sequences by efficiently packing multiple examples
- **LoRA (Low-Rank Adaptation)**: Memory-efficient fine-tuning reducing trainable parameters by 90-99%
- **Enhanced TensorBoard Logging**: Comprehensive metadata tracking including token budget, parameter counts, and actual attention implementation
- **Generative Classification**: All classification tasks are handled using text generation instead of classification heads
- **Dataset Mixture Training**: Mix multiple datasets with configurable mixture rates for multi-task learning
- **Schema Harmonization**: Automatically handles different dataset formats and combines them seamlessly
- **Multi-GPU Support**: Single-node multi-GPU training with DistributedDataParallel (DDP)
- **Multi-dataset support**: 25+ datasets including IMDB, GLUE, finance datasets, mathematical reasoning, code generation, and large pretraining corpora
- **Flexible training modes**: Pretraining, supervised fine-tuning (SFT), and reinforcement learning (RL) with GRPO 
- **Automatic task detection**: Smart detection of task types based on dataset
- **Attention Implementation Resolution**: Automatic fallback to stable attention on MPS devices with proper logging
- **Argument logging**: All training arguments automatically saved to JSON/text for reproducibility
- **Modular architecture**: Clean separation of concerns for easy extension
- **Comprehensive checkpointing**: Automatic checkpoint management and resumption
- **Device flexibility**: Automatic detection and support for CUDA, MPS (Apple Silicon), and CPU
- **Quantization Support**: 4-bit and 8-bit quantization for large models on limited hardware

## 🎯 Generative Approach

This pipeline uses a **generative approach** for all classification tasks, treating them as text generation problems instead of using classification heads. This approach:

- ✅ **Works with**: Decoder-only models (GPT, Qwen, Llama) and Encoder-Decoder models (T5, BART)
- ❌ **Incompatible with**: Encoder-only models (BERT, RoBERTa, DeBERTa)
- 🎯 **Benefits**: More flexible, instruction-following, better zero-shot capabilities

### Example: Classification as Generation

**Traditional approach** (with classification head):
```
Input: "This movie is great!"
Output: [0.1, 0.9] (logits for negative/positive)
```

**Generative approach** (this pipeline):
```
Input: "Classify the sentiment of this movie review as either 'positive' or 'negative'.\n\nReview: This movie is great!\nSentiment:"
Output: " positive"
```

## 📋 Quick Start

### Installation

```bash
pip install torch transformers datasets accelerate
```

### Basic Usage

```bash
# IMDB sentiment classification with generative approach
python train.py --model gpt2 --dataset stanfordnlp/imdb --task auto --mode sft

# GLUE SST-2 sentiment classification
python train.py --model gpt2 --dataset glue --dataset_config sst2 --task auto --mode sft

# GLUE MRPC paraphrase detection
python train.py --model gpt2 --dataset glue --dataset_config mrpc --task auto --mode sft

# Dataset mixture training (NEW FEATURE)
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.7 0.3 --mode sft
```

## 🏗️ Architecture

### 🤖 Comprehensive Model Support

The pipeline supports a wide range of models, including standard transformers and advanced MOE (Mixture of Experts) architectures.

#### Model Categories

| Category | Compatible | Architecture Type | Use Cases |
|----------|-----------|------------------|-----------|
| **Standard Decoder-only** | ✅ | Autoregressive | Text generation, classification via generation |
| **MOE Models** | ✅ | Sparse Mixture of Experts | Large-scale training, specialized domains |
| **Encoder-Decoder** | ✅ | Seq2Seq | Translation, summarization, generation |
| **Encoder-only** | ❌ | Bidirectional | Not supported (requires classification heads) |

#### 📊 Supported Models - Complete List

##### Standard Language Models

| Model | Size | Parameters | Memory (FP32/FP16) | HuggingFace ID | Notes |
|-------|------|------------|-------------------|----------------|-------|
| **GPT-2** | Small | 124M | 0.5GB / 0.25GB | `gpt2` | Classic autoregressive |
| **GPT-2 Medium** | Medium | 355M | 1.4GB / 0.7GB | `gpt2-medium` | Larger GPT-2 variant |
| **GPT-2 Large** | Large | 774M | 3.1GB / 1.5GB | `gpt2-large` | Powerful generation |
| **GPT-2 XL** | XL | 1.5B | 6GB / 3GB | `gpt2-xl` | Largest GPT-2 |
| **DialoGPT** | Small-Large | 117M-762M | 0.5GB-3GB / 0.25GB-1.5GB | `microsoft/DialoGPT-[small/medium/large]` | Conversational |
| **Qwen2** | 0.5B-7B | 0.5B-7B | 2GB-28GB / 1GB-14GB | `Qwen/Qwen2-[0.5B/1.5B/7B]` | Multilingual |
| **Qwen2.5** | 0.5B-3B | 0.5B-3B | 2GB-12GB / 1GB-6GB | `Qwen/Qwen2.5-[0.5B/1.5B/3B]`, `Qwen/Qwen2.5-[0.5B/1.5B/3B]-Instruct` | Latest Qwen2 series, 128K context |
| **Qwen3** | 0.6B-4B | 0.6B-4B | 2.4GB-16GB / 1.2GB-8GB | Base: `Qwen/Qwen3-[0.6B/1.7B/4B]-Base`, Post-trained: `Qwen/Qwen3-[0.6B/1.7B/4B]`, Instruct: `Qwen/Qwen3-4B-Instruct-2507` | Advanced reasoning, thinking mode |
| **Llama 3.2** | 1B/3B | 1.2B/3.2B | 4.8GB/12.8GB / 2.4GB/6.4GB | `meta-llama/Llama-3.2-[1B/3B]` | Edge optimized, 128K context |
| **T5** | Small-3B | 60M-3B | 0.24GB-12GB / 0.12GB-6GB | `t5-[small/base/large/3b]` | Encoder-decoder |
| **BART** | Base/Large | 140M-400M | 0.56GB-1.6GB / 0.28GB-0.8GB | `facebook/bart-[base/large]` | Denoising seq2seq |

##### MOE (Mixture of Experts) Models

| Model | Experts | Active Experts | Total Parameters | Memory Requirements | HuggingFace ID | Device Compatibility |
|-------|---------|----------------|------------------|-------------------|----------------|----------------------|
| **Mixtral 8x7B** | 8 | 2 | 46.7B | ~93GB (FP32) / ~47GB (FP16) / ~13GB (4-bit) | `mistralai/Mixtral-8x7B-v0.1` | CUDA, MPS* with quantization |
| **Mixtral 8x22B** | 8 | 2 | 141B | ~282GB (FP32) / ~141GB (FP16) / ~35GB (4-bit) | `mistralai/Mixtral-8x22B` | Multi-GPU CUDA |
| **Switch-Base-8** | 8 | 1 | 0.2B | ~0.8GB (FP32) / ~0.4GB (FP16) | `google/switch-base-8` | All devices |
| **Switch-Base-16** | 16 | 1 | 0.3B | ~1.2GB (FP32) / ~0.6GB (FP16) | `google/switch-base-16` | All devices |
| **Switch-Base-32** | 32 | 1 | 0.5B | ~2GB (FP32) / ~1GB (FP16) | `google/switch-base-32` | All devices |
| **Switch-Base-128** | 128 | 1 | 1.6B | ~6.4GB (FP32) / ~3.2GB (FP16) | `google/switch-base-128` | CUDA, MPS |
| **DeepSeek-MOE 16B** | 64 | 6 | 16.4B | ~66GB (FP32) / ~33GB (FP16) / ~8GB (4-bit) | `deepseek-ai/deepseek-moe-16b-base` | CUDA, MPS* with quantization |
| **Arctic Base** | 128 | 2 | 480B | ~960GB (FP32) / ~480GB (FP16) | `snowflake/arctic-base` | Multi-node CUDA cluster |
| **Qwen3-235B-A22B** | 16 | 2 | 235B | ~940GB (FP32) / ~470GB (FP16) / ~59GB (4-bit) | `Qwen/Qwen3-235B-A22B` | Multi-GPU CUDA |
| **Qwen3-30B-A3B** | 8 | 1 | 30B | ~120GB (FP32) / ~60GB (FP16) / ~8GB (4-bit) | `Qwen/Qwen3-30B-A3B` | CUDA, MPS* with quantization |

*MPS = Apple Silicon Metal Performance Shaders

#### 💾 Memory Requirements by Device

##### Apple M1/M2 Max (32GB-96GB RAM)

| Configuration | Recommended Models | Notes |
|--------------|-------------------|-------|
| **32GB RAM** | GPT-2 (all), Qwen2-0.5B/1.5B, T5-small/base, Switch-Base-8/16/32, Mixtral-8x7B (4-bit) | Use quantization for large MOE |
| **64GB RAM** | All standard models, DeepSeek-MOE (8-bit), Mixtral-8x7B (8-bit) | Better performance with FP32 |
| **96GB RAM** | All models except Arctic, Mixtral-8x22B (4-bit) | Can run most models natively |

##### NVIDIA GPUs

| GPU | VRAM | Recommended Models | Notes |
|-----|------|-------------------|-------|
| **RTX 3060** | 12GB | GPT-2, Qwen2-0.5B/1.5B, T5-base, Switch-Base models | Limited to smaller models |
| **RTX 3090/4090** | 24GB | All standard models, Mixtral-8x7B (4-bit), DeepSeek-MOE (8-bit) | Good for most use cases |
| **A100 40GB** | 40GB | All models except Arctic, Mixtral-8x7B (FP16) | Professional training |
| **A100 80GB** | 80GB | All models except Arctic (single GPU), Mixtral-8x22B (8-bit) | Large model training |

#### 🚀 Model Usage Examples

##### Standard Models
```bash
# Small models (fits on most devices)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft
python train.py --model t5-small --dataset glue --dataset_config cola --mode sft
python train.py --model Qwen/Qwen2-0.5B --dataset openai/gsm8k --dataset_config main --mode sft
# Qwen3 models under 5B (both Base and post-trained versions available)
python train.py --model Qwen/Qwen3-0.6B-Base --dataset stanfordnlp/imdb --mode sft  # 600M pretrained only
python train.py --model Qwen/Qwen3-0.6B --dataset stanfordnlp/imdb --mode sft  # 600M post-trained
python train.py --model Qwen/Qwen3-1.7B-Base --dataset glue --dataset_config sst2 --mode sft  # 1.7B pretrained only
python train.py --model Qwen/Qwen3-1.7B --dataset glue --dataset_config sst2 --mode sft  # 1.7B post-trained
python train.py --model Qwen/Qwen3-4B-Base --dataset openai/gsm8k --dataset_config main --mode sft  # 4B pretrained only
python train.py --model Qwen/Qwen3-4B --dataset openai/gsm8k --dataset_config main --mode sft  # 4B post-trained
python train.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset virattt/financial-qa-10K --mode sft  # 4B Instruct variant

# Qwen2.5 models (alternative option)
python train.py --model Qwen/Qwen2.5-0.5B --dataset stanfordnlp/imdb --mode sft  # Qwen2.5 smallest
python train.py --model Qwen/Qwen2.5-1.5B --dataset glue --dataset_config sst2 --mode sft  # Qwen2.5 1.5B

# Medium models
python train.py --model gpt2-medium --dataset stanfordnlp/imdb --mode sft --batch_size 8
python train.py --model facebook/bart-base --dataset glue --dataset_config mrpc --mode sft

# Large models
python train.py --model gpt2-large --dataset yale-nlp/FinanceMath --mode sft --batch_size 4
python train.py --model Qwen/Qwen2-7B --dataset cais/mmlu --mode sft --batch_size 2
python train.py --model Qwen/Qwen3-1.7B-Base --dataset glue --dataset_config mrpc --mode sft --batch_size 8  # Qwen3 1.7B pretrained
python train.py --model Qwen/Qwen3-1.7B --dataset glue --dataset_config mrpc --mode sft --batch_size 8  # Qwen3 1.7B post-trained
python train.py --model Qwen/Qwen3-4B-Base --dataset virattt/financial-qa-10K --mode sft --batch_size 4  # Qwen3 4B pretrained
python train.py --model Qwen/Qwen3-4B --dataset virattt/financial-qa-10K --mode sft --batch_size 4  # Qwen3 4B post-trained
python train.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset openai/gsm8k --dataset_config main --mode sft  # 4B Instruct
```

##### MOE Models
```bash
# Small MOE models (direct loading)
python train.py --model google/switch-base-8 --dataset stanfordnlp/imdb --mode sft --device mps
python train.py --model google/switch-base-16 --dataset glue --dataset_config sst2 --mode sft

# Large MOE models with quantization
python train.py --model mistralai/Mixtral-8x7B-v0.1 --dataset stanfordnlp/imdb --mode sft --moe_load_in_4bit --batch_size 1
python train.py --model deepseek-ai/deepseek-moe-16b-base --dataset openai/gsm8k --dataset_config main --mode sft --moe_load_in_8bit

# Qwen3 MOE models
python train.py --model Qwen/Qwen3-30B-A3B --dataset stanfordnlp/imdb --mode sft --moe_load_in_4bit --batch_size 1  # Small MOE
python train.py --model Qwen/Qwen3-235B-A22B --dataset openai/gsm8k --dataset_config main --mode sft --moe_load_in_4bit  # Large MOE (multi-GPU)

# MOE with Flash Attention (CUDA only)
python train.py --model mistralai/Mixtral-8x7B-v0.1 --dataset glue --dataset_config cola --mode sft --use_flash_attention --device cuda
```

#### ⚙️ MOE-Specific Features

The pipeline includes automatic MOE detection and optimization:

- **Automatic Detection**: MOE models are identified by name patterns
- **Device Optimization**: Automatic dtype selection (FP32 for MPS, FP16 for CUDA)
- **Expert Configuration**: Automatic configuration of expert routing
- **Memory Estimation**: Parameter counting and memory usage estimation
- **Quantization Support**: 4-bit and 8-bit loading options for large models

#### 🔧 Model Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model name or path | `gpt2`, `mistralai/Mixtral-8x7B-v0.1` |
| `--use_flash_attention` | Enable Flash Attention 2 (CUDA only) | `--use_flash_attention` |
| `--moe_load_in_4bit` | Load MOE in 4-bit quantization | `--moe_load_in_4bit` |
| `--moe_load_in_8bit` | Load MOE in 8-bit quantization | `--moe_load_in_8bit` |
| `--device` | Device selection | `auto`, `cuda`, `mps`, `cpu` |

#### 📈 Performance Recommendations

##### For Training Speed
1. **Small datasets**: Use GPT-2 or T5-small
2. **Large datasets**: Use Switch-Base models (efficient sparse computation)
3. **Multi-task**: Use Mixtral models with appropriate quantization

##### For Quality
1. **Best overall**: Mixtral-8x7B (with quantization if needed)
2. **Mathematical tasks**: Qwen2 models or DeepSeek-MOE
3. **Code generation**: GPT-2 Large or CodeLlama variants
4. **Financial tasks**: Fine-tuned BART or T5 models

##### For Memory Efficiency
1. **Limited RAM (<16GB)**: GPT-2 small, T5-small, Switch-Base-8
2. **Moderate RAM (16-32GB)**: GPT-2 medium/large, Qwen2-0.5B/1.5B, quantized MOE
3. **High RAM (>32GB)**: All standard models, MOE with appropriate quantization

### Classification Prompts

The pipeline automatically generates appropriate prompts for different classification tasks:

**IMDB Sentiment:**
```
Classify the sentiment of this movie review as either 'positive' or 'negative'.

Review: This movie is absolutely fantastic!
Sentiment: positive
```

**GLUE CoLA (Grammar):**
```
Determine if this sentence is grammatically acceptable or unacceptable.

Sentence: The book was on the table.
Grammar: acceptable
```

**GLUE MRPC (Paraphrase):**
```
Determine if these two sentences are equivalent or not_equivalent in meaning.

Sentence 1: The cat sat on the mat.
Sentence 2: A cat was sitting on the mat.
Relation: equivalent
```

## 📊 Comprehensive Dataset Support

The pipeline supports **22 datasets** across four main categories, all using a **generative approach** where classification tasks are treated as text generation problems.

### 🏷️ Classification Datasets (6 datasets)

#### 1. IMDB Movie Reviews
- **HuggingFace ID**: `stanfordnlp/imdb`
- **Task Type**: Binary sentiment classification
- **Training Modes**: ✅ SFT, ✅ Pretrain
- **Dataset Statistics**:
  - Train: 25,000 examples
  - Test: 25,000 examples
  - Unsupervised: 50,000 examples
  - Total size: ~133 MB
- **Labels**: {0: "negative", 1: "positive"}
- **Usage**: `--dataset stanfordnlp/imdb`

#### 2. GLUE Benchmark
- **HuggingFace ID**: `glue`
- **Task Type**: Multi-task NLU benchmark (9 tasks)
- **Training Modes**: ✅ SFT
- **Total Examples**: ~1.48M across all tasks

| Config | Task | Train | Val | Test | Labels | Description |
|--------|------|-------|-----|------|--------|-------------|
| `cola` | Grammar | 8,551 | 1,043 | 1,063 | acceptable, unacceptable | Grammar acceptability |
| `sst2` | Sentiment | 67,349 | 872 | 1,821 | positive, negative | Sentiment analysis |
| `mrpc` | Paraphrase | 3,668 | 408 | 1,725 | equivalent, not_equivalent | Paraphrase detection |
| `qqp` | Questions | 363,846 | 40,430 | 390,965 | duplicate, not_duplicate | Question paraphrase |
| `mnli` | NLI | 392,702 | 9,815/9,832 | 9,796/9,847 | entailment, neutral, contradiction | Natural language inference |
| `qnli` | QA-NLI | 104,743 | 5,463 | 5,463 | entailment, not_entailment | Question answering NLI |
| `rte` | Entailment | 2,490 | 277 | 3,000 | entailment, not_entailment | Textual entailment |
| `wnli` | Winograd | 635 | 71 | 146 | entailment, not_entailment | Winograd NLI |

**Usage**: `--dataset glue --dataset_config [cola|sst2|mrpc|qqp|mnli|qnli|rte|wnli]`

#### 3. Financial PhraseBank
- **HuggingFace ID**: `takala/financial_phrasebank`
- **Task Type**: Financial sentiment classification (3-class)
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - sentences_allagree: 2,264 examples
  - sentences_75agree: 3,453 examples
  - sentences_66agree: 4,217 examples
  - sentences_50agree: 4,846 examples
- **Labels**: {0: "negative", 1: "neutral", 2: "positive"}
- **Domain**: Financial news sentences
- **Usage**: `--dataset takala/financial_phrasebank`

#### 4. Twitter Financial News
- **HuggingFace ID**: `zeroshot/twitter-financial-news-sentiment`
- **Task Type**: Financial sentiment classification (3-class)
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Train: 9,938 examples
  - Validation: 2,486 examples
  - Size: ~1.08 MB
- **Labels**: {0: "bearish", 1: "bullish", 2: "neutral"}
- **Domain**: Financial Twitter data
- **Usage**: `--dataset zeroshot/twitter-financial-news-sentiment`

#### 5. FiQA Sentiment Classification
- **HuggingFace ID**: `TheFinAI/fiqa-sentiment-classification`
- **Task Type**: Financial sentiment classification
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Train: 822 examples
  - Validation: 117 examples
  - Test: 234 examples
- **Labels**: {0: "negative", 1: "neutral", 2: "positive"} (score-based conversion)
- **Domain**: Financial microblogs and news
- **Usage**: `--dataset TheFinAI/fiqa-sentiment-classification`

#### 6. MMLU (Massive Multitask Language Understanding)
- **HuggingFace ID**: `cais/mmlu`
- **Task Type**: Multiple choice knowledge evaluation (57 subjects)
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Total examples: ~15,908 across all subjects
  - Subjects: 57 academic subjects from STEM to humanities
  - Test examples: 14,042 / Dev examples: 1,531 / Val examples: 285
- **Labels**: Multiple choice (A, B, C, D options)
- **Domain**: Academic knowledge across diverse subjects
- **Usage**: `--dataset cais/mmlu`

### 🤖 Generative/Q&A Datasets (10 datasets)

#### 7. FiQA Question Answering
- **HuggingFace ID**: `LLukas22/fiqa`
- **Task Type**: Financial question answering
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Train: ~14.5k examples
  - Test: ~2.56k examples
- **Schema**: question, answer columns
- **Domain**: Financial Q&A
- **Usage**: `--dataset LLukas22/fiqa`

#### 8. Finance Alpaca
- **HuggingFace ID**: `gbharti/finance-alpaca`
- **Task Type**: Financial instruction following
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Total: 68,900 examples
  - Size: 42.9 MB
- **Schema**: instruction, input, output (Alpaca format)
- **Domain**: Financial instruction following
- **Usage**: `--dataset gbharti/finance-alpaca`

#### 9. SEC Financial Reports
- **HuggingFace ID**: `JanosAudran/financial-reports-sec`
- **Task Type**: Financial document analysis/generation
- **Training Modes**: ✅ SFT, ✅ Pretrain
- **Dataset Statistics**:
  - large_lite: 20.5M rows
  - small_lite: 240k rows
- **Domain**: SEC 10-K filings (US public firms to 2020)
- **Usage**: `--dataset JanosAudran/financial-reports-sec`

#### 10. Financial News Articles
- **HuggingFace ID**: `ashraq/financial-news-articles`
- **Task Type**: Financial document pretraining/generation
- **Training Modes**: ✅ SFT, ✅ Pretrain
- **Dataset Statistics**:
  - Total: 306,437 articles
  - Size: ~280M tokens
  - Time span: 2009-2020
- **Schema**: title, text, ticker columns
- **Domain**: Financial news articles
- **Usage**: `--dataset ashraq/financial-news-articles`

#### 11. Sujet Finance QA Vision
- **HuggingFace ID**: `sujet-ai/Sujet-Finance-QA-Vision-100k`
- **Task Type**: Visual financial question answering
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Total images: 9,801 financial documents
  - Total QA pairs: 107,050
  - Train: 9,210 examples / Test: 589 examples
- **Special**: Supports vision-language models
- **Usage**: `--dataset sujet-ai/Sujet-Finance-QA-Vision-100k`

#### 12. GSM8K (Grade School Math)
- **HuggingFace ID**: `openai/gsm8k`
- **Task Type**: Mathematical word problem solving
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Train: 7,473 examples
  - Test: 1,319 examples
  - Configs: main, socratic
- **Schema**: question, answer columns with step-by-step solutions
- **Domain**: Grade school level math problems
- **Usage**: `--dataset openai/gsm8k --dataset_config main`

#### 12. FinanceMath
- **HuggingFace ID**: `yale-nlp/FinanceMath`
- **Task Type**: Knowledge-intensive financial mathematics
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Validation: 200 examples
  - Test: 1,000 examples
- **Schema**: Complex financial math problems with detailed solutions
- **Domain**: Financial mathematics requiring domain expertise
- **Usage**: `--dataset yale-nlp/FinanceMath`

#### 13. DeepMind Math Dataset
- **HuggingFace ID**: `deepmind/math_dataset`
- **Task Type**: Mathematical reasoning across multiple domains
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - Multiple mathematics modules with varying difficulty
  - Algebra, arithmetic, calculus, probability, etc.
- **Schema**: question, answer pairs with mathematical notation
- **Domain**: School-level mathematics across various topics
- **Usage**: `--dataset deepmind/math_dataset`

#### 14. BigCodeBench
- **HuggingFace ID**: `bigcode/bigcodebench`
- **Task Type**: Programming with complex function calls
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - 1,140 function-level programming tasks
  - Uses 139 different libraries
- **Schema**: instruction/prompt, canonical_solution
- **Domain**: Software engineering with diverse function calls
- **Usage**: `--dataset bigcode/bigcodebench`

#### 15. MMLU-Pro (Enhanced Reasoning)
- **HuggingFace ID**: `TIGER-Lab/MMLU-Pro`
- **Task Type**: Advanced multitask reasoning (10 options vs 4)
- **Training Modes**: ✅ SFT
- **Dataset Statistics**:
  - 12,000+ questions across 14 domains
  - Increased difficulty and reasoning focus
- **Schema**: Enhanced multiple choice with detailed reasoning
- **Domain**: Advanced knowledge requiring chain-of-thought reasoning
- **Usage**: `--dataset TIGER-Lab/MMLU-Pro`

### 📚 Pretraining Corpora (7 datasets)

#### 16. Common Corpus
- **HuggingFace ID**: `PleIAs/common_corpus`
- **Task Type**: Large-scale multilingual pretraining
- **Training Modes**: ✅ Pretrain
- **Dataset Statistics**:
  - Total tokens: **2 trillion** (1,998,647,168,282)
  - English tokens: 867,033,096,123
  - Languages: 33 languages with >1B tokens
- **Domain**: Multilingual web content
- **Usage**: `--dataset PleIAs/common_corpus`

#### 17. BookCorpus
- **HuggingFace ID**: `bookcorpus/bookcorpus`
- **Task Type**: Literary text pretraining
- **Training Modes**: ✅ Pretrain
- **Dataset Statistics**:
  - Total examples: 74,004,228 sentences
  - Dataset size: 4.85 GB
  - Books: ~7,185 unique books
  - Words: ~1 billion words
- **Domain**: Literature from various genres
- **Usage**: `--dataset bookcorpus/bookcorpus`

#### 19. AdaptLLM Finance Tasks
- **HuggingFace ID**: `AdaptLLM/finance-tasks`
- **Task Type**: Financial domain evaluation tasks
- **Training Modes**: ✅ SFT, evaluation
- **Dataset Statistics**:
  - Total subsets: 5 (ConvFinQA, FPB, FiQA_SA, Headline, NER)
  - Total examples: ~23k across all subsets
- **Domain**: Financial NLP evaluation
- **Usage**: `--dataset AdaptLLM/finance-tasks --dataset_config [ConvFinQA|FPB|FiQA_SA|Headline|NER]`

#### 20. OpenWebMath
- **HuggingFace ID**: `open-web-math/open-web-math`
- **Task Type**: Mathematical web content for pretraining
- **Training Modes**: ✅ Pretrain
- **Dataset Statistics**:
  - Total documents: 6.3 million high-quality mathematical documents
  - Filtered from 200B+ HTML files from Common Crawl
  - Size: ~14.7 GB of mathematical text
- **Domain**: Mathematical content from web pages
- **Usage**: `--dataset open-web-math/open-web-math`

#### 21. WikiText
- **HuggingFace ID**: `wikitext`
- **Task Type**: Wikipedia pretraining corpus
- **Training Modes**: ✅ Pretrain
- **Dataset Statistics**:
  - **wikitext-103-raw-v1**: 1,801,350 train examples (546.5 MB)
  - **wikitext-2-raw-v1**: 36,718 train examples
  - Total tokens: >100 million
- **Domain**: Wikipedia articles
- **Usage**: `--dataset wikitext --dataset_config [wikitext-2-raw-v1|wikitext-103-raw-v1]`

### 📊 Dataset Summary by Size

| Size Category | Count | Examples |
|---------------|-------|----------|
| **Small (< 10K)** | 5 | FiQA Sentiment, Financial PhraseBank, Twitter Financial, GSM8K, FinanceMath |
| **Medium (10K - 100K)** | 6 | FiQA Q&A, Finance Alpaca, WikiText-2, AdaptLLM, MMLU, Sujet Finance QA |
| **Large (100K - 1M)** | 4 | IMDB, SEC Reports, News Articles, BigCodeBench |
| **Very Large (> 1M)** | 6 | GLUE, BookCorpus, WikiText-103, Common Corpus, OpenWebMath, DeepMind Math |

### 🎯 Training Mode Support

| Training Mode | Datasets | Use Cases |
|---------------|----------|-----------|
| **SFT Only** | 14 datasets | Classification, Q&A, reasoning, coding, mathematical problem solving |
| **Pretrain Only** | 5 datasets | Large-scale language modeling, mathematical pretraining |
| **Both SFT & Pretrain** | 3 datasets | Domain adaptation (IMDB, SEC Reports, News Articles) |

### 🧠 Task Complexity Levels

| Complexity Level | Datasets | Description |
|------------------|----------|-------------|
| **Basic Classification** | 5 datasets | Sentiment analysis, basic categorization |
| **Advanced Reasoning** | 8 datasets | Mathematical reasoning, multitask understanding |
| **Domain Expertise** | 6 datasets | Financial knowledge, specialized domains |
| **Code Generation** | 2 datasets | Programming and software engineering |

### 🔧 Dataset Mixture Examples

```bash
# Mix classification datasets (4-way mixture)
python train.py --datasets stanfordnlp/imdb glue glue glue --dataset_configs None sst2 cola mrpc --mixture_rates 0.4 0.3 0.2 0.1 --mode sft

# Mix finance datasets
python train.py --datasets takala/financial_phrasebank zeroshot/twitter-financial-news-sentiment --dataset_configs None None --mixture_rates 0.6 0.4 --mode sft

# Mix generative and classification
python train.py --datasets LLukas22/fiqa zeroshot/twitter-financial-news-sentiment --dataset_configs None None --mixture_rates 0.7 0.3 --mode sft

# Mix mathematical reasoning datasets
python train.py --datasets openai/gsm8k yale-nlp/FinanceMath --dataset_configs main None --mixture_rates 0.6 0.4 --mode sft

# Mix reasoning and coding datasets
python train.py --datasets cais/mmlu bigcode/bigcodebench --dataset_configs None None --mixture_rates 0.5 0.5 --mode sft
```

## 🔧 Configuration Options

### Model Selection
```bash
# Decoder-only models (recommended)
python train.py --model gpt2
python train.py --model Qwen/Qwen2-0.5B
python train.py --model microsoft/DialoGPT-small

# Encoder-decoder models
python train.py --model t5-small
python train.py --model facebook/bart-base
```

### Training Modes
```bash
# Supervised fine-tuning with generative classification
python train.py --mode sft --dataset stanfordnlp/imdb

# Pretraining (language modeling)
python train.py --mode pretrain --dataset wikitext --dataset_config wikitext-2-raw-v1

# Reinforcement learning with GRPO
python train.py --mode rl --dataset openai/gsm8k --dataset_config main --max_steps 100
```

### Hyperparameters
```bash
python train.py --batch_size 16 --max_length 256 --save_steps 1000

# Control logging frequency (default: matches eval_steps)
python train.py --logging_steps 50  # Log training metrics every 50 steps
```

### Device Selection
```bash
python train.py --device auto    # Automatic detection
python train.py --device cuda    # Force CUDA
python train.py --device mps     # Force Apple Silicon
python train.py --device cpu     # Force CPU
```

### Multi-GPU Training
```bash
# Enable multi-GPU training (single-node)
python train.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft

# Multi-GPU with gradient accumulation for larger effective batch sizes
python train.py --multi_gpu --gradient_accumulation_steps 4 --batch_size 8 --model gpt2 --dataset stanfordnlp/imdb --mode sft

# Multi-GPU with specific DDP backend
python train.py --multi_gpu --ddp_backend nccl --model gpt2 --dataset stanfordnlp/imdb --mode sft

# Optimal multi-GPU training with torchrun (recommended)
torchrun --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft

# Alternative with torch.distributed.launch (older PyTorch versions)
python -m torch.distributed.launch --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft
```

## 📁 Project Structure

```
lancy_thesis/
├── train.py               # Main entry point for training
├── src/                   # Source code modules
│   ├── main.py           # Training pipeline orchestration
│   ├── config/
│   │   └── args.py       # Configuration and argument parsing
│   ├── models/
│   │   └── utils.py      # Model loading and prompt creation
│   ├── data/
│   │   └── utils.py      # Dataset loading and tokenization
│   └── training/
│       ├── utils.py          # Training setup and utilities
│       ├── metadata_mixin.py # TensorBoard metadata logging mixin
│       └── rl_utils.py       # Reinforcement learning utilities
├── tests/                 # Test suite
│   ├── test_modular_structure.py  # Core functionality tests
│   ├── test_moe_m1.py             # MOE model tests
│   ├── test_multi_gpu.py         # Multi-GPU tests
│   └── test_rl.py                # RL/GRPO tests
├── experimental_plan/     # Thesis experimental documentation
├── proposal/             # Thesis proposal documents
└── runs/                 # Training outputs and checkpoints
```

## 🧪 Testing

Run the test suites to verify functionality:

```bash
# Test core modular structure and generative approach
python tests/test_modular_structure.py

# Test MOE model support on Apple Silicon
python tests/test_moe_m1.py

# Test multi-GPU functionality
python tests/test_multi_gpu.py

# Test reinforcement learning with GRPO
python tests/test_rl.py
```

**Core tests** (`test_modular_structure.py`):
- ✅ Import functionality
- ✅ Tokenizer setup
- ✅ Task determination
- ✅ Classification task detection
- ✅ Label mapping
- ✅ Prompt creation
- ✅ Model loading

**MOE tests** (`test_moe_m1.py`):
- ✅ MOE model detection
- ✅ MPS device compatibility
- ✅ Memory usage estimation
- ✅ Model configuration
- ✅ Quantization recommendations

## 💾 Checkpointing and Resumption

**Automatic checkpoint management:**
```bash
python train.py --save_steps 500 --save_total_limit 3
```

**Resume from checkpoint:**
```bash
# Resume from latest checkpoint
python train.py --resume_from_checkpoint latest

# Resume from specific checkpoint
python train.py --resume_from_checkpoint ./runs/model_gpt2_dataset_glue_config_sst2_mode_sft_maxlen_128_batch_8_save_steps_500_task_causal-lm_20250703_153515/checkpoints/checkpoint-1000

# Continue training with different parameters
python train.py --resume_from_checkpoint latest --eval_steps 200 --save_steps 500

# Resume and evaluate best model
python train.py --resume_from_checkpoint latest --load_best_model_at_end --metric_for_best_model eval_loss
```

**Custom output directory:**
```bash
python train.py --output_dir ./custom_experiment
```

## 🎯 Examples

### Sentiment Analysis (IMDB)
```bash
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --max_length 512
```

### Grammar Checking (CoLA)
```bash
python train.py --model gpt2 --dataset glue --dataset_config cola --mode sft
```

### Paraphrase Detection (MRPC)
```bash
python train.py --model gpt2 --dataset glue --dataset_config mrpc --mode sft --batch_size 16
```

### Natural Language Inference (MNLI)
```bash
python train.py --model gpt2 --dataset glue --dataset_config mnli --mode sft --max_length 256
```

### Mathematical Reasoning (GSM8K)
```bash
python train.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode sft --max_length 512
```

### Financial Mathematics (FinanceMath)
```bash
python train.py --model gpt2 --dataset yale-nlp/FinanceMath --mode sft --max_length 1024
```

### Multitask Reasoning (MMLU)
```bash
python train.py --model gpt2 --dataset cais/mmlu --mode sft --max_length 512
```

### Code Generation (BigCodeBench)
```bash
python train.py --model gpt2 --dataset bigcode/bigcodebench --mode sft --max_length 1024
```

### Mathematical Pretraining (OpenWebMath)
```bash
python train.py --model gpt2 --dataset open-web-math/open-web-math --mode pretrain --max_length 512
```

### Training Metric Logging
```bash
# Log training metrics (loss, grad_norm, learning_rate, epoch) every 50 steps
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --logging_steps 50

# Logging frequency automatically matches evaluation frequency (default behavior)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_steps 100
# This will log training metrics every 100 steps to match evaluation

# Different frequencies for logging and evaluation
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --logging_steps 10 --eval_steps 100
# Logs training metrics every 10 steps, evaluates every 100 steps
```

## 🔍 How It Works

### 1. **Dataset Processing**
- Detects classification tasks automatically
- Converts integer labels to text labels (0 → "negative", 1 → "positive")
- Creates task-specific prompts

### 2. **Model Architecture**
- Uses `AutoModelForCausalLM` for all tasks (generative approach)
- No classification heads - pure text generation
- Language modeling loss function

### 3. **Training Process**
- Formats data as prompt + target pairs
- Uses `DataCollatorForLanguageModeling`
- Trains to generate correct label text

### 4. **Inference** (Future Work)
- Generate text and parse output
- Extract predicted labels from generated text
- Handle various output formats

## 🔄 Dataset Mixture Training

**NEW FEATURE**: Mix multiple datasets with configurable mixture rates for multi-task learning.

### Basic Mixture Usage
```bash
# Mix IMDB and GLUE SST-2 with 70%/30% split
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.7 0.3 --mode sft

# Mix multiple datasets for pretraining
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.6 0.4 --mode pretrain
```

### Key Mixture Features
- **Schema Harmonization**: Automatically handles different dataset schemas (IMDB vs GLUE formats)
- **Proportional Sampling**: Samples from each dataset according to specified mixture rates  
- **Unified Tokenization**: Handles mixed datasets with appropriate prompts for classification tasks
- **Hierarchical Naming**: Generates clear experiment names like `mix_imdb-sentiment-70_glue-sentiment-30`

### Mixture Arguments
- `--datasets`: Space-separated list of dataset names
- `--dataset_configs`: Corresponding configs (use "None" for datasets without configs)
- `--mixture_rates`: Floating-point rates that must sum to 1.0

### Example Mixtures
```bash
# Binary sentiment tasks (IMDB + SST-2)
python train.py --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.8 0.2 --mode sft

# Multi-task classification (sentiment + grammar)
python train.py --datasets stanfordnlp/imdb glue --dataset_configs None cola --mixture_rates 0.5 0.5 --mode sft

# Three-way mixture
python train.py --datasets stanfordnlp/imdb glue glue --dataset_configs None sst2 mrpc --mixture_rates 0.5 0.3 0.2 --mode sft
```

## 🚀 Advanced Usage

### Custom Prompts
The prompt templates can be customized in `model_utils.py`:

```python
def create_classification_prompt(text, sentence2=None, dataset_name="", dataset_config=None):
    # Customize prompts here
    if 'imdb' in dataset_name.lower():
        return f"Classify the sentiment: {text}\nSentiment:"
```

### Adding New Datasets
1. Add label mapping in `get_label_mapping()`
2. Add prompt template in `create_classification_prompt()`
3. Update `is_classification_task()` if needed

### Custom Models
```bash
# Use any HuggingFace model
python train.py --model your-username/your-model-name
python train.py --model ./path/to/local/model
```

## 📈 Monitoring

### Training Outputs

Training outputs include:
- **Automatic logging**: Text logs in `runs/*/logs/`
- **TensorBoard**: Comprehensive metrics and metadata in `runs/*/tensorboard/`
- **Checkpoints**: Model states in `runs/*/checkpoints/`
- **Configuration**: Saved arguments in JSON and readable text format

### TensorBoard Features

The pipeline provides rich TensorBoard logging with comprehensive training metadata:

#### Metrics Logged
- **Training Loss**: Real-time training loss curves
- **Evaluation Metrics**: Validation loss and performance metrics
- **Learning Rate**: LR scheduler progression
- **Gradient Norms**: Monitor gradient stability

#### Metadata Dashboard (NEW)

At training start (step 0), the following metadata is automatically logged:

**Scalars:**
- **Token Budget**: Total tokens processed (batch_size × max_length × steps)
- **Total Parameters**: Model parameter count
- **Trainable Parameters**: Parameters being updated (important for LoRA)
- **Trainable Percentage**: Ratio of trainable to total parameters

**Text Summaries:**
- **Configuration Summary**: Complete training configuration
- **Attention Implementation**: Actual attention mechanism used (not just the command-line setting)
- **Device Information**: Hardware details and mixed precision settings
- **Dataset Information**: Dataset names, configs, and mixture rates (if applicable)

#### Viewing TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir runs/

# View specific experiment
tensorboard --logdir runs/your_experiment_name/tensorboard/

# Compare multiple runs
tensorboard --logdir_spec exp1:runs/exp1,exp2:runs/exp2
```

### Attention Implementation Tracking

The pipeline automatically resolves and logs the actual attention implementation used:

- **Command-line**: `--attn_implementation auto`
- **Actual (on MPS)**: `eager` (automatically resolved for stability)
- **TensorBoard shows**: `Attention: eager` (the actual implementation)

This is particularly important on Apple Silicon (MPS) devices where SDPA attention can cause NaN values, so the system automatically falls back to eager attention and logs this resolution.

## ⚠️ Important Notes

1. **Model Compatibility**: Only works with generative models (GPT-style, T5-style)
2. **Memory Usage**: Generative approach may use more memory due to longer sequences
3. **Training Time**: May take longer than classification heads due to text generation
4. **Evaluation**: Currently focuses on training; evaluation metrics would need custom implementation

## ⚡ Performance Optimizations

### FlashAttention 2

FlashAttention 2 provides 2-3x speedup on attention computation with automatic fallback:

```bash
# Enable FlashAttention 2 (CUDA only)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --use_flash_attention

# Automatic fallback on MPS/CPU
# On macOS: Warns and uses standard attention
# Missing package: Provides installation instructions
```

**Benefits:**
- 2-3x faster attention computation
- 50-70% memory reduction
- Supports sequences up to 40K tokens
- Perfect for long-context training

### Sequence Packing

Pack multiple short sequences into single training examples for 3-5x speedup:

```bash
# Enable packing for pretraining
python train.py --model gpt2 --dataset wikitext --mode pretrain --use_packing --max_length 2048

# Combine with FlashAttention for maximum performance
python train.py --model Qwen/Qwen3-1.7B-Base --dataset bookcorpus/bookcorpus \
    --mode pretrain --use_packing --use_flash_attention --max_length 4096
```

**Automatic dataset preprocessing:**
- Q&A datasets (question/answer) → Combined format
- Input/Output datasets → Concatenated with separators
- Instruction/Response → Formatted for training

**Expected speedups:**
- Short text (50-100 tokens): 3-5x faster
- Medium text (200-500 tokens): 2-3x faster
- Long text (>1000 tokens): 1-1.5x faster

### LoRA (Low-Rank Adaptation)

Memory-efficient fine-tuning with 90-99% parameter reduction:

```bash
# Basic LoRA fine-tuning
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --use_lora

# Custom LoRA parameters
python train.py --model Qwen/Qwen3-1.7B-Base --dataset virattt/financial-qa-10K \
    --mode sft --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05

# Combine all optimizations
python train.py --model Qwen/Qwen3-1.7B-Base --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 --mode pretrain \
    --use_lora --use_packing --use_flash_attention \
    --lora_r 16 --batch_size 32 --max_length 4096
```

**LoRA Parameters:**
- `--lora_r`: Rank (default: 16, common: 8, 16, 32, 64)
- `--lora_alpha`: Scaling (default: 32, typically 2x rank)
- `--lora_dropout`: Dropout (default: 0.1)

## 🖥️ Multi-GPU Training

The pipeline supports **single-node multi-GPU training** using PyTorch's DistributedDataParallel (DDP) for optimal performance.

### Quick Start

```bash
# Basic multi-GPU training
python train.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft

# Recommended: Use torchrun for optimal setup
torchrun --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft
```

### Multi-GPU Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--multi_gpu` | False | Enable multi-GPU training with DDP |
| `--ddp_backend` | nccl | Distributed backend (`nccl` for CUDA, `gloo` for debugging) |
| `--gradient_accumulation_steps` | 1 | Accumulate gradients to simulate larger batch sizes |

### Effective Batch Size Calculation

With multi-GPU training, the **total effective batch size** is:
```
Total Batch Size = per_device_batch_size × num_gpus × gradient_accumulation_steps
```

**Example:**
```bash
# 4 GPUs × batch_size=8 × accumulation=4 = 128 total batch size
torchrun --nproc_per_node=4 main.py --multi_gpu --batch_size 8 --gradient_accumulation_steps 4
```

### Launch Methods

#### Method 1: torchrun (Recommended)
```bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft

# With custom settings
torchrun --nproc_per_node=8 main.py --multi_gpu --batch_size 4 --gradient_accumulation_steps 8 --ddp_backend nccl
```

#### Method 2: torch.distributed.launch (Legacy)
```bash
# Older PyTorch versions
python -m torch.distributed.launch --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft
```

#### Method 3: Direct Python (Basic)
```bash
# Will work but shows optimization suggestions
python train.py --multi_gpu --model gpt2 --dataset stanfordnlv/imdb --mode sft
```

### Performance Optimizations

- **Mixed Precision**: Automatically enabled with CUDA GPUs (`fp16=True`)
- **DataLoader Workers**: Automatically configured (4 workers for multi-GPU, 2 for single GPU)
- **Pin Memory**: Enabled for GPU training to improve data transfer speed
- **DDP Backend**: Uses NCCL for optimal CUDA communication

### Monitoring Multi-GPU Training

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check training logs
tail -f runs/*/logs/training.log

# View TensorBoard metrics
tensorboard --logdir runs/*/tensorboard/
```

### Experiment Naming

Multi-GPU experiments are automatically labeled in directory names:
```
2025-07-30_14h32m_gpt2_imdb-sentiment_sft_bs8_len256_acc4_multigpu
```

### Troubleshooting

#### Issue: "Multi-GPU requested but only 1 GPU available"
**Solution:** System has only one GPU. Remove `--multi_gpu` flag or add more GPUs.

#### Issue: "Multi-GPU not supported with MPS"
**Solution:** MPS (Apple Silicon) doesn't support multi-GPU. Use CUDA GPUs instead.

#### Issue: "CUDA requested but not available"
**Solution:** Install CUDA-compatible PyTorch or use `--device cpu` for testing.

#### Issue: Slow training with multi-GPU
**Solutions:**
- Use `torchrun` instead of direct Python execution
- Increase `--gradient_accumulation_steps` to reduce communication overhead
- Ensure all GPUs are of similar performance
- Check that data loading isn't the bottleneck

### Memory Management

For large models or datasets:
```bash
# Reduce per-device batch size, increase accumulation
torchrun --nproc_per_node=4 main.py --multi_gpu --batch_size 2 --gradient_accumulation_steps 16

# Use gradient checkpointing (if supported by model)
torchrun --nproc_per_node=4 main.py --multi_gpu --fp16 --batch_size 4
```

### Supported Configurations

- ✅ **Single-node multi-GPU**: Multiple GPUs on one machine
- ✅ **Mixed precision training**: Automatic FP16 with CUDA
- ✅ **Gradient accumulation**: Simulate larger batch sizes
- ✅ **All dataset types**: Classification, generative, pretraining
- ❌ **Multi-node training**: Not currently supported

## 🤖 Reinforcement Learning with GRPO

The pipeline now supports **Group Relative Policy Optimization (GRPO)** for reinforcement learning fine-tuning, completing the three-stage LLM training pipeline: pretraining → SFT → RL.

### Features

- **GRPO Implementation**: Memory-efficient alternative to PPO using group-based advantage estimation
- **Simple Reward Functions**: Built-in task-specific reward functions for different dataset types
- **Custom Reward Models**: Support for loading external reward models
- **Automatic Dataset Preparation**: Converts datasets to prompt-completion pairs for RL training
- **Compatible with All Datasets**: Works with classification, Q&A, mathematical reasoning, and code generation tasks

### Quick Start

```bash
# Basic RL training with GSM8K (mathematical reasoning)
python train.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode rl --max_steps 100

# RL training with custom hyperparameters
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode rl \
  --grpo_beta 0.1 --rl_learning_rate 1e-6 --max_prompt_length 256 --max_completion_length 256

# RL training with custom reward model
python train.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode rl \
  --reward_model path/to/reward/model --max_steps 200
```

### RL-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--grpo_beta` | 0.1 | Beta parameter for KL regularization strength |
| `--grpo_group_size` | 2 | Group size for advantage computation |
| `--max_prompt_length` | 512 | Maximum length for prompts |
| `--max_completion_length` | 512 | Maximum length for completions |
| `--rl_learning_rate` | 1e-6 | Learning rate for RL (typically lower than SFT) |
| `--rl_warmup_steps` | 100 | Number of warmup steps |
| `--reward_model` | None | Path to custom reward model (optional) |

### Built-in Reward Functions

The pipeline includes task-specific reward functions:

- **Mathematical Tasks** (GSM8K, FinanceMath): Rewards step-by-step reasoning, calculations, and numerical answers
- **Sentiment Classification** (IMDB, Financial sentiment): Rewards correct sentiment predictions
- **Code Generation** (BigCodeBench): Rewards proper code patterns and structure
- **Financial Tasks**: Rewards domain-specific terminology and accuracy
- **GLUE Tasks**: Rewards correct task-specific outputs (grammar, paraphrase, entailment)

### RL Training Pipeline

1. **Dataset Preparation**: Converts any dataset to prompt-completion pairs
2. **Reward Function**: Uses built-in heuristics or custom reward models
3. **GRPO Training**: Optimizes policy using group-based advantage estimation
4. **Generation**: Model learns to generate better responses based on rewards

### Examples

```bash
# Mathematical reasoning with RL
python train.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode rl \
  --batch_size 4 --max_steps 500 --max_prompt_length 256 --max_completion_length 256

# Financial Q&A with RL
python train.py --model gpt2 --dataset LLukas22/fiqa --mode rl \
  --grpo_beta 0.05 --rl_learning_rate 5e-7 --max_steps 300

# Code generation with RL
python train.py --model gpt2 --dataset bigcode/bigcodebench --mode rl \
  --max_prompt_length 512 --max_completion_length 512 --batch_size 2

# Sentiment analysis with RL
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode rl \
  --grpo_beta 0.2 --max_steps 200
```

### Performance Tips

- Use lower learning rates (1e-6 to 1e-7) for RL compared to SFT
- Start with smaller batch sizes due to memory requirements for generation
- Adjust `grpo_beta` to control exploration vs exploitation
- Consider using a smaller model first to validate the setup

### Current Status

✅ **Fully Implemented and Tested**: 
- GRPO trainer integration with HuggingFace TRL
- Task-specific reward function framework  
- Automatic dataset preparation for RL training
- RL-specific hyperparameters and configuration
- Model compatibility with generation methods
- Support for all dataset types (classification, Q&A, mathematical reasoning, financial tasks)
- Integration with main training pipeline

✅ **Verified Working**:
- Mathematical reasoning (GSM8K): Rewards step-by-step solutions
- Sentiment classification (IMDB): Rewards correct sentiment predictions  
- Financial Q&A (FiQA): Rewards domain-specific responses
- All built-in reward functions operational
- Full end-to-end RL training pipeline functional

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 