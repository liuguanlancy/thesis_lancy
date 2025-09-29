# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular HuggingFace training pipeline that uses a **generative approach** for all classification tasks. Instead of using classification heads, all tasks (including sentiment analysis, grammar checking, paraphrase detection) are treated as text generation problems.

**Key Architecture Principle**: The pipeline uses `AutoModelForCausalLM` for all tasks, training models to generate text labels (e.g., "positive", "negative") rather than predicting class probabilities.

## Thesis Context

This codebase is part of a Master's thesis project titled **"Information-Secure Language Models for Finance Applications"** by Guanlan Liu at the University of Zurich. The thesis focuses on developing a lightweight, privacy-preserving chatbot designed specifically for financial applications that can run efficiently on local devices.

### Thesis Goals
- **Privacy-Preserving**: Eliminate the need to upload sensitive financial data to external services
- **Lightweight**: Target models that can run on personal devices (laptops/phones)
- **Finance-Focused**: Optimize for financial document analysis and tasks
- **Model Size Target**: 2-3B parameters (e.g., Gemma2 2B, LLaMA3 1.5B/3B)
- **Techniques**: Leverage LoRA for efficient finetuning and quantization for deployment

### Related Documents
- **Thesis Proposal**: Located in `proposal/main.tex` with bibliography in `proposal/ref.bib`
- **Experimental Plan**: Comprehensive plan in `experimental_plan/` directory with 8 detailed documents covering methodology, datasets, metrics, hardware, and timeline

## Core Commands

### Running Training

The training pipeline can be run from the project root using `train.py` or directly via `python src/main.py`:

```bash
# Basic training with IMDB sentiment analysis (from project root)
python train.py --model gpt2 --dataset stanfordnlp/imdb --task auto --mode sft

# Or directly via src/main.py
python src/main.py --model gpt2 --dataset stanfordnlp/imdb --task auto --mode sft

# GLUE benchmark tasks (e.g., SST-2, CoLA, MRPC)
python train.py --model gpt2 --dataset glue --dataset_config sst2 --task auto --mode sft

# Dataset mixture training (NEW FEATURE)
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.7 0.3 --mode sft
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.6 0.4 --mode pretrain

# Multi-GPU training (NEW FEATURE)
python train.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft --batch_size 8 --gradient_accumulation_steps 4
torchrun --nproc_per_node=4 main.py --multi_gpu --model gpt2 --dataset stanfordnlp/imdb --mode sft

# New financial classification datasets
python train.py --model gpt2 --dataset zeroshot/twitter-financial-news-topic --mode sft --max_length 256
python train.py --model gpt2 --dataset FinGPT/fingpt-sentiment-train --mode sft --max_length 512

# New financial Q&A datasets
python train.py --model gpt2 --dataset virattt/financial-qa-10K --mode sft --max_length 1024
python train.py --model gpt2 --dataset AdaptLLM/finance-tasks --dataset_config ConvFinQA --mode sft
python train.py --model gpt2 --dataset AdaptLLM/finance-tasks --dataset_config FPB --mode sft

# Mathematical and reasoning datasets
python train.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode sft --max_length 512
python train.py --model gpt2 --dataset TIGER-Lab/MMLU-Pro --mode sft --max_length 512

# Pretraining on web corpus (OpenWebText)
python train.py --model gpt2 --dataset Skylion007/openwebtext --mode pretrain --max_length 512

# Custom parameters
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --batch_size 16 --max_length 512 --save_steps 1000

# Control training duration
python train.py --model gpt2 --dataset glue --dataset_config sst2 --mode sft --max_steps 100  # Train for specific number of steps
python train.py --model gpt2 --dataset glue --dataset_config sst2 --mode sft --num_train_epochs 3  # Train for multiple epochs

# Control logging frequency (NEW FEATURE)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --logging_steps 50  # Log training metrics every 50 steps
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_steps 100  # Logging defaults to match eval (100)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --logging_steps 10 --eval_steps 100  # Different frequencies

# Control evaluation frequency and scope (NEW FEATURES)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_steps 50  # Evaluate every 50 steps
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_steps 100 --save_steps 500  # Eval every 100 steps, save every 500
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_max_batches 10  # Evaluate using only 10 batches
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --eval_steps 50 --eval_max_batches 5  # Frequent but quick evals

# MOE model training (NEW FEATURE)
python train.py --model google/switch-base-8 --dataset stanfordnlp/imdb --mode sft --device mps --batch_size 4
python train.py --model mistralai/Mixtral-8x7B-v0.1 --dataset glue --dataset_config sst2 --mode sft --moe_load_in_4bit --batch_size 1

# LoRA fine-tuning (NEW FEATURE)
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --use_lora --lora_r 16 --lora_alpha 32
python train.py --model gpt2 --dataset virattt/financial-qa-10K --mode sft --use_lora --lora_r 8 --batch_size 8
python train.py --model Qwen/Qwen2.5-1.5B --dataset FinGPT/fingpt-sentiment-train --mode sft --use_lora --lora_r 32

# Qwen3 model training (NEW SUPPORT) - Models under 5B
# Both Base (pretrained only) and post-trained versions are available

# Base models (pretrained only - best for custom fine-tuning)
python train.py --model Qwen/Qwen3-0.6B-Base --dataset stanfordnlp/imdb --mode sft --use_lora --lora_r 4  # 600M pretrained
python train.py --model Qwen/Qwen3-1.7B-Base --dataset openai/gsm8k --dataset_config main --mode sft  # 1.7B pretrained
python train.py --model Qwen/Qwen3-4B-Base --dataset virattt/financial-qa-10K --mode sft --max_length 512  # 4B pretrained

# Post-trained models (already fine-tuned on diverse data)
python train.py --model Qwen/Qwen3-0.6B --dataset stanfordnlp/imdb --mode sft --use_lora --lora_r 4  # 600M post-trained
python train.py --model Qwen/Qwen3-1.7B --dataset openai/gsm8k --dataset_config main --mode sft  # 1.7B post-trained
python train.py --model Qwen/Qwen3-4B --dataset virattt/financial-qa-10K --mode sft --max_length 512  # 4B post-trained

# Instruct variant (instruction-tuned)
python train.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset TIGER-Lab/MMLU-Pro --mode sft  # 4B Instruct for advanced reasoning

# Qwen3 with LoRA for memory efficiency
python train.py --model Qwen/Qwen3-0.6B-Base --dataset glue --dataset_config cola --mode sft --use_lora --lora_r 8
python train.py --model Qwen/Qwen3-1.7B --dataset stanfordnlp/imdb --mode sft --use_lora --lora_r 16 --lora_alpha 32
```

### Testing
```bash
# Run all tests (unit, integration, functional)
bash tests/scripts/run_all_tests.sh

# Run specific test suites
bash tests/scripts/run_unit_tests.sh       # Fast unit tests
bash tests/scripts/run_integration_tests.sh # Integration tests
bash tests/scripts/run_functional_tests.sh  # Functional tests

# Run individual test files (legacy)
python tests/unit/test_modular_structure.py  # Core functionality
python tests/unit/check_device.py           # Device detection

# Test outputs are saved in: tests/outputs/[unit|integration|functional]/
```

### Linting and Code Quality
```bash
# Run flake8 (installed in requirements.txt)
flake8 src/ --max-line-length=120 --ignore=E203,W503

# Format code with black (if installed)
black src/ tests/ --line-length=120

# Type checking (if mypy installed)
mypy src/ --ignore-missing-imports
```

### Environment Setup
```bash
# Use the existing conda environment
conda activate /Users/mengzhao/miniconda3/envs/lancy

# Or create from lancy.yml if needed
conda env create -f lancy.yml

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Key packages: torch, transformers, datasets, accelerate, peft, trl, flash_attn
```

## Project Structure

The codebase is organized into a clean package structure:

```
lancy_thesis/
├── src/                    # Main source code
│   ├── main.py            # Entry point (orchestrates the pipeline)
│   ├── config/
│   │   └── args.py        # Argument parsing and experiment naming
│   ├── models/
│   │   └── utils.py       # Model loading, tokenizer setup, task determination
│   ├── data/
│   │   └── utils.py       # Dataset loading and tokenization
│   └── training/
│       ├── utils.py       # Training arguments and checkpoint management
│       └── rl_utils.py    # Reinforcement learning utilities
├── tests/                  # Test suite
│   ├── test_modular_structure.py
│   ├── test_moe_m1.py
│   ├── test_multi_gpu.py
│   └── test_rl.py
├── experimental_plan/      # Thesis experimental documentation
├── proposal/              # Thesis proposal documents
└── train.py              # Root entry point (redirects to src/main.py)
```

### Core Modules

- **`src/main.py`**: Orchestrates the training pipeline
- **`src/config/args.py`**: Argument parsing and experiment directory generation
- **`src/models/utils.py`**: Model loading, tokenizer setup, task determination, and prompt creation
- **`src/data/utils.py`**: Dataset loading and generative tokenization for classification
- **`src/training/utils.py`**: Training arguments, data collation, and checkpoint management
- **`src/training/rl_utils.py`**: GRPO and reinforcement learning support

### Key Functions by Module

**src/models/utils.py**:
- `setup_tokenizer()`: Configures tokenizer with proper padding tokens
- `determine_task()`: Auto-detects task type (always returns 'causal-lm' for generative approach)
- `get_label_mapping()`: Maps integer labels to text labels (0→"negative", 1→"positive")
- `create_classification_prompt()`: Generates task-specific prompts for different datasets
- `load_model()`: Loads model with MOE detection, quantization, and device optimization
- `is_moe_model()`: Detects Mixture of Experts models
- `get_moe_config()`: Returns MOE-specific configuration

**src/data/utils.py**:
- `load_and_prepare_dataset()`: Loads datasets from HuggingFace Hub
- `load_and_prepare_dataset_mixture()`: Loads and mixes multiple datasets with configurable rates
- `harmonize_dataset_schema()`: Standardizes different dataset schemas for concatenation
- `is_classification_task()`: Detects if a dataset is a classification task
- `create_tokenize_function()`: Creates tokenization function for generative training
- `prepare_dataset_for_packing()`: Preprocesses datasets for sequence packing

**src/training/utils.py**:
- `check_device_availability()`: Auto-detects CUDA/MPS/CPU
- `create_training_arguments()`: Sets up HuggingFace TrainingArguments
- `setup_text_logging()`: Configures logging to files
- `get_data_collator()`: Returns appropriate data collator (packing-aware)
- `save_training_config()`: Saves configuration in JSON and text formats

**src/training/metadata_mixin.py**:
- `MetadataLoggerMixin`: Adds TensorBoard metadata logging to trainer
- Logs token budget, parameter counts, attention implementation, device info

## Generative Classification Approach

This pipeline's unique feature is treating classification as text generation:

### Prompt Templates
- **IMDB**: `"Classify the sentiment of this movie review as either 'positive' or 'negative'.\n\nReview: {text}\nSentiment:"`
- **GLUE CoLA**: `"Determine if this sentence is grammatically acceptable or unacceptable.\n\nSentence: {text}\nGrammar:"`
- **GLUE MRPC**: `"Determine if these two sentences are equivalent or not_equivalent in meaning.\n\nSentence 1: {text1}\nSentence 2: {text2}\nRelation:"`

### Model Compatibility
- ✅ **Compatible**: GPT-2, Qwen2, Qwen3 (0.6B/1.7B/4B in both Base and post-trained versions), Llama 3.2 (1B/3B), T5, BART (decoder-only and encoder-decoder models)
- ✅ **MOE Models**: Mixtral, Switch Transformers, DeepSeek-MOE, Arctic, Qwen3-MOE (with MPS/CUDA support)
- ❌ **Incompatible**: BERT, RoBERTa, DeBERTa (encoder-only models)
- 🚀 **Qwen3 Special Features**: Thinking mode for complex reasoning, 36T token training, 119 languages
- 🚀 **Llama 3.2 Special Features**: Edge-optimized (1B/3B), 128K context window, multimodal-ready architecture

## Training Modes

- **`sft`**: Supervised fine-tuning using generative approach for classification
- **`pretrain`**: Standard causal language modeling for pretraining

## Dataset Support

### Supported Datasets
- **IMDB**: `stanfordnlp/imdb` (binary sentiment)
- **GLUE**: `glue` with configs: `cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`

### Label Mappings (in `get_label_mapping()`)
```python
# IMDB: {0: "negative", 1: "positive"}
# GLUE SST-2: {0: "negative", 1: "positive"}
# GLUE CoLA: {0: "unacceptable", 1: "acceptable"}
# GLUE MRPC: {0: "not_equivalent", 1: "equivalent"}
```

## Checkpoint Management

### Experiment Naming
The pipeline automatically generates clean, readable experiment names in the format:
`{date}_{time}_{model}_{dataset-task}_{mode}_{params}`

**Examples:**
- `2025-07-30_14h32m_gpt2_glue-sentiment_sft_bs16_len128`
- `2025-07-30_14h35m_qwen2-0.5b_imdb-sentiment_sft_bs8_len256_ep3`
- `2025-07-30_14h40m_bert-base_glue-grammar_sft_bs4_len512_steps1000_save100`
- `2025-07-30_01h59m_gpt2_mix_imdb-sentiment-70_glue-sentiment-30_sft_bs2_len64_steps2` (dataset mixture)

**Name Components:**
- **Date/Time**: `2025-07-30_14h32m` (readable timestamp)
- **Model**: `gpt2`, `qwen2-0.5b`, `bert-base` (shortened model names)
- **Dataset-Task**: `imdb-sentiment`, `glue-grammar`, `glue-paraphrase` (semantic task names)
- **Mode**: `sft`, `pretrain` (training mode)
- **Parameters**: `bs16_len128_ep3` (batch size, max length, epochs/steps)

### Directory Structure
- **Resume Training**: `--resume_from_checkpoint latest` or specific checkpoint path
- **Checkpoints**: Saved in `runs/{experiment_name}/checkpoints/`
- **Logs**: Text logs in `runs/{experiment_name}/logs/training.log`
- **TensorBoard**: Metrics in `runs/{experiment_name}/tensorboard/`

## Device Management

Automatic device detection in `check_device_availability()`:
- CUDA → CUDA
- Apple Silicon → MPS
- Fallback → CPU

## Important Implementation Notes

1. **Tokenizer Configuration**: Left padding for generation, automatic padding token assignment
2. **Task Auto-Detection**: Always uses `causal-lm` task for consistency with generative approach
3. **Data Collation**: Uses `DataCollatorForLanguageModeling` for all tasks
4. **Model Loading**: Always uses `AutoModelForCausalLM` regardless of original task type

## Testing Structure

The test suite has been reorganized into three categories:

### Unit Tests (`tests/unit/`)
- **test_modular_structure.py**: Core functionality (imports, tokenizer, tasks, prompts)
- **check_device.py**: Device detection and availability

### Integration Tests (`tests/integration/`)
- Tests for model loading with different configurations
- Dataset processing and mixing
- Training pipeline integration

### Functional Tests (`tests/functional/`)
- End-to-end training tests
- MOE model tests on Apple Silicon
- Multi-GPU functionality
- Reinforcement learning with GRPO

Test outputs are automatically saved to `tests/outputs/[category]/` for debugging.

This modular design allows easy extension for new datasets by adding label mappings and prompt templates to the respective utility functions.

## Supported Datasets (26 Total)

The pipeline supports a comprehensive range of datasets across different complexity levels:

### Classification Datasets (10)
- **IMDB**: `stanfordnlp/imdb` - Movie review sentiment (binary)
- **GLUE**: `glue` with configs (cola, sst2, mrpc, etc.) - Multi-task NLU benchmark
- **Financial PhraseBank**: `takala/financial_phrasebank` - Financial sentiment (3-class)
- **Twitter Financial Sentiment**: `zeroshot/twitter-financial-news-sentiment` - Financial tweets (bearish/bullish/neutral)
- **Twitter Financial Topic**: `zeroshot/twitter-financial-news-topic` - 20 financial topic classes
- **FiQA Sentiment**: `TheFinAI/fiqa-sentiment-classification` - Financial microblogs
- **FinGPT Sentiment**: `FinGPT/fingpt-sentiment-train` - Instruction-based sentiment (76K examples)
- **MMLU-Pro**: `TIGER-Lab/MMLU-Pro` - Advanced multi-choice reasoning (12K test)
- **AdaptLLM FPB**: `AdaptLLM/finance-tasks` (config: FPB) - Financial PhraseBank variant
- **AdaptLLM FiQA_SA**: `AdaptLLM/finance-tasks` (config: FiQA_SA) - FiQA sentiment variant
- **MMLU**: `cais/mmlu` - Requires trust_remote_code (not fully supported)

### Generative/Q&A Datasets (11)
- **FiQA Q&A**: `LLukas22/fiqa` - Financial question answering
- **Finance Alpaca**: `gbharti/finance-alpaca` - Financial instruction following (68K)
- **Financial QA 10K**: `virattt/financial-qa-10K` - QA on 10-K filings (7K examples)
- **AdaptLLM ConvFinQA**: `AdaptLLM/finance-tasks` (config: ConvFinQA) - Conversational financial QA
- **AdaptLLM NER**: `AdaptLLM/finance-tasks` (config: NER) - Named entity recognition
- **Sujet Finance QA Vision**: `sujet-ai/Sujet-Finance-QA-Vision-100k` - Multimodal financial QA
- **GSM8K**: `openai/gsm8k` (config: main) - Grade school math word problems
- **FinanceMath**: `yale-nlp/FinanceMath` - Financial math (gated, not accessible)
- **DeepMind Math**: `deepmind/math_dataset` - Mathematical reasoning
- **BigCodeBench**: `bigcode/bigcodebench` - Code generation with function calls
- **SEC Reports**: `JanosAudran/financial-reports-sec` - Requires trust_remote_code

### Pretraining Corpora (4)
- **OpenWebText**: `Skylion007/openwebtext` - Web content (39.8 GB)
- **BookCorpus**: `bookcorpus/bookcorpus` - Literary texts (~1B words)
- **WikiText**: `wikitext` - Wikipedia articles
- **Common Corpus**: `PleIAs/common_corpus` - 2T tokens (very large)

## LoRA (Low-Rank Adaptation) Support

**NEW**: The pipeline now supports LoRA for efficient fine-tuning with significantly reduced memory requirements.

### LoRA Features
- **Automatic Module Detection**: Automatically detects target modules based on model architecture
- **Configurable Parameters**: Adjustable rank, alpha, and dropout
- **Memory Efficient**: Reduces trainable parameters by 90-99%
- **Compatible Models**: Works with GPT-2, LLaMA, Qwen, Gemma, Phi, and more

### LoRA Usage
```bash
# Basic LoRA fine-tuning
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft --use_lora

# Custom LoRA parameters
python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \
    --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05

# LoRA with larger models
python train.py --model meta-llama/Llama-2-7b-hf --dataset virattt/financial-qa-10K \
    --mode sft --use_lora --lora_r 16 --batch_size 4
```

### LoRA Parameters
- `--use_lora`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: 16, common values: 8, 16, 32, 64)
- `--lora_alpha`: LoRA alpha scaling parameter (default: 32, typically 2x rank)
- `--lora_dropout`: Dropout for LoRA layers (default: 0.1)
- `--lora_target_modules`: Target modules (auto-detected by default)

## MOE (Mixture of Experts) Support

The pipeline also supports MOE models with automatic detection, MPS optimization, and quantization options.

### Supported MOE Models
- **Mixtral**: `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x22B`
- **Switch Transformers**: `google/switch-base-8`, `google/switch-base-16`, `google/switch-base-32`
- **DeepSeek-MOE**: `deepseek-ai/deepseek-moe-16b-base`
- **Arctic**: `snowflake/arctic-base`, `snowflake/arctic-instruct`

### MOE Features
- **Automatic Detection**: MOE models are automatically detected and configured
- **MPS Optimization**: Optimized for Apple Silicon (M1/M2) with float32 precision
- **CUDA Support**: Full float16 support on NVIDIA GPUs
- **Memory Management**: Smart device mapping and dtype selection
- **Quantization**: Support for 4-bit and 8-bit loading (via --moe_load_in_4bit/8bit)

### MOE on Apple M1 Max (32GB)
For M1 Max users, here are the recommended approaches:

#### Small MOE Models (Direct Loading)
```bash
# Google Switch Transformers (3-6GB)
python train.py --model google/switch-base-8 --dataset stanfordnlp/imdb --mode sft --device mps

# With custom settings
python train.py --model google/switch-base-16 --dataset glue --dataset_config sst2 --mode sft --batch_size 4 --max_length 256
```

#### Large MOE Models (Requires Quantization)
```bash
# Mixtral with 4-bit quantization (~13GB)
python train.py --model mistralai/Mixtral-8x7B-v0.1 --dataset stanfordnlp/imdb --mode sft --moe_load_in_4bit --batch_size 1

# DeepSeek with 8-bit quantization (~16GB)
python train.py --model deepseek-ai/deepseek-moe-16b-base --dataset glue --dataset_config cola --mode sft --moe_load_in_8bit --batch_size 1
```

### MOE Implementation Details
- **Detection**: `is_moe_model()` in `model_utils.py` identifies MOE architectures
- **Configuration**: `get_moe_config()` returns model-specific expert settings
- **Loading**: `load_model()` handles device mapping and precision settings
- **Memory**: Automatic parameter counting and memory estimation

### Testing MOE Support
Run the dedicated MOE test script:
```bash
python test_moe_m1.py
```

This will:
- Test MOE model detection
- Verify MPS compatibility
- Show memory requirements
- Provide optimization recommendations

## Dataset Mixture Feature

**NEW**: The pipeline now supports mixing multiple datasets with configurable mixture rates for both SFT and pretrain modes.

### Usage
```bash
# Mix IMDB and GLUE SST-2 with 70%/30% split
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.7 0.3 --mode sft

# Mix multiple datasets for pretraining
python train.py --model gpt2 --datasets stanfordnlp/imdb glue --dataset_configs None sst2 --mixture_rates 0.6 0.4 --mode pretrain
```

### Key Features
- **Schema Harmonization**: Automatically handles different dataset schemas (IMDB vs GLUE column formats)
- **Proportional Sampling**: Samples from each dataset according to specified mixture rates
- **Unified Tokenization**: Handles mixed datasets with appropriate prompts for classification tasks
- **Hierarchical Naming**: Generates clear experiment names like `mix_imdb-sentiment-70_glue-sentiment-30`

### Arguments
- `--datasets`: Space-separated list of dataset names
- `--dataset_configs`: Corresponding configs (use "None" for datasets without configs)
- `--mixture_rates`: Floating-point rates that must sum to 1.0

### Implementation Details
- **Dataset Loading**: `load_and_prepare_dataset_mixture()` in `dataset_utils.py`
- **Schema Standardization**: `harmonize_dataset_schema()` converts all datasets to common `{text, label}` format
- **Mixed Tokenization**: Intelligent tokenization that detects dataset types and applies appropriate processing

## Important Technical Details

### Flash Attention and MPS Compatibility
- FlashAttention 2 is CUDA-only; automatically falls back to standard attention on MPS
- On Apple Silicon (MPS), the pipeline automatically uses `eager` attention to prevent NaN values
- TensorBoard logs the actual attention implementation used, not just the command-line setting

### Memory Management for Large Models
- MOE models on M1 Max (32GB): Use 4-bit quantization for Mixtral, 8-bit for DeepSeek
- LoRA reduces trainable parameters by 90-99%, essential for large models on limited hardware
- Sequence packing provides 3-5x speedup on short sequences during pretraining

### Dataset Processing Optimizations
- Classification tasks are automatically converted to generative format with appropriate prompts
- Dataset mixtures use proportional sampling with schema harmonization
- Q&A datasets are preprocessed to combine question/answer pairs for packing

## Development Guidelines

### Git Commit Messages
- **Never mention Claude or AI assistance in commit messages**
- Focus on technical changes and improvements made
- Use clear, descriptive commit messages that explain what was changed and why
- Follow conventional commit format when possible

### Adding New Features
1. **New Datasets**: Add label mapping in `get_label_mapping()` and prompt template in `create_classification_prompt()`
2. **New Models**: Ensure compatibility with `AutoModelForCausalLM` or add special handling in `load_model()`
3. **New Tasks**: Update `is_classification_task()` and tokenization logic as needed