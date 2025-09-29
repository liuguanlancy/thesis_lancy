import argparse
import os
import sys
from datetime import datetime


def shorten_model_name(model_name):
    """Convert long model names to readable short forms."""
    model_name = model_name.lower().replace('/', '-').replace('\\', '-')
    
    # Common model name mappings including MOE models
    model_mappings = {
        'gpt2': 'gpt2',
        'microsoft-diagpt-small': 'diagpt-sm',
        'microsoft-diagpt-medium': 'diagpt-md',
        'microsoft-diagpt-large': 'diagpt-lg',
        'qwen-qwen2-0.5b': 'qwen2-0.5b',
        'qwen-qwen2-1.5b': 'qwen2-1.5b',
        'qwen-qwen2-7b': 'qwen2-7b',
        'qwen-qwen3-0.6b': 'qwen3-0.6b',
        'bert-base-uncased': 'bert-base',
        'bert-large-uncased': 'bert-large',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        't5-small': 't5-sm',
        't5-base': 't5-base',
        't5-large': 't5-lg',
        'facebook-bart-base': 'bart-base',
        'facebook-bart-large': 'bart-lg',
        # MOE models
        'mistralai-mixtral-8x7b': 'mixtral-8x7b',
        'mistralai-mixtral-8x7b-instruct': 'mixtral-8x7b-inst',
        'mistralai-mixtral-8x22b': 'mixtral-8x22b',
        'google-switch-base-8': 'switch-b8',
        'google-switch-base-16': 'switch-b16',
        'google-switch-base-32': 'switch-b32',
        'deepseek-ai-deepseek-moe-16b': 'deepseek-moe-16b',
        'snowflake-arctic-base': 'arctic-base',
        'snowflake-arctic-instruct': 'arctic-inst',
    }
    
    # Check for exact matches first
    if model_name in model_mappings:
        return model_mappings[model_name]
    
    # Pattern-based simplification
    for pattern, replacement in model_mappings.items():
        if pattern in model_name:
            return replacement
    
    # If no mapping found, create a simple version
    # Take first part before any dash/slash and limit length
    simplified = model_name.split('-')[0].split('/')[0]
    return simplified[:10]  # Limit to 10 characters


def create_dataset_task_name(dataset_name, dataset_config=None):
    """Create a concise dataset-task identifier."""
    dataset_name = dataset_name.lower().replace('/', '-')
    
    if 'imdb' in dataset_name:
        return 'imdb-sentiment'
    elif dataset_name == 'glue' and dataset_config:
        task_descriptions = {
            'cola': 'glue-grammar',
            'sst2': 'glue-sentiment', 
            'mrpc': 'glue-paraphrase',
            'qqp': 'glue-questions',
            'mnli': 'glue-nli',
            'qnli': 'glue-qa-nli',
            'rte': 'glue-entail',
            'wnli': 'glue-winograd',
            'stsb': 'glue-similarity'
        }
        return task_descriptions.get(dataset_config, f'glue-{dataset_config}')
    elif 'wikitext' in dataset_name:
        return 'wikitext-lm'
    elif 'financial_phrasebank' in dataset_name or 'takala-financial_phrasebank' in dataset_name:
        return 'finphrasebank-sentiment'
    elif 'twitter-financial-news-sentiment' in dataset_name or 'zeroshot-twitter-financial-news-sentiment' in dataset_name:
        return 'fintwitter-sentiment'
    elif 'fiqa-sentiment-classification' in dataset_name or 'thefinai-fiqa-sentiment-classification' in dataset_name:
        return 'fiqa-sentiment'
    elif 'fiqa' in dataset_name.lower() and 'llukas22' in dataset_name.lower():
        return 'fiqa-qa'
    elif 'finance-alpaca' in dataset_name or 'gbharti-finance-alpaca' in dataset_name:
        return 'finance-alpaca'
    elif 'financial-reports-sec' in dataset_name or 'janosaudran-financial-reports-sec' in dataset_name:
        return 'sec-reports'
    elif 'sujet-finance-qa' in dataset_name.lower() or 'sujet-ai-sujet-finance-qa' in dataset_name.lower():
        return 'sujet-finqa'
    elif 'common_corpus' in dataset_name.lower() or 'pleias-common_corpus' in dataset_name.lower():
        return 'common-corpus'
    elif 'openwebtext' in dataset_name.lower() or 'skylion007-openwebtext' in dataset_name.lower():
        return 'openwebtext'
    elif 'bookcorpus' in dataset_name.lower():
        return 'bookcorpus'
    elif 'adaptllm-finance-tasks' in dataset_name.lower():
        return 'adaptllm-finance'
    elif 'open-web-math' in dataset_name.lower() or 'open_web_math' in dataset_name.lower():
        return 'openwebmath'
    elif 'gsm8k' in dataset_name.lower() or 'openai-gsm8k' in dataset_name.lower():
        return 'gsm8k-math'
    elif 'financemath' in dataset_name.lower() or 'yale-nlp-financemath' in dataset_name.lower():
        return 'financemath'
    elif 'math_dataset' in dataset_name.lower() or 'deepmind-math_dataset' in dataset_name.lower():
        return 'deepmind-math'
    elif 'bigcodebench' in dataset_name.lower() or 'bigcode-bigcodebench' in dataset_name.lower():
        return 'bigcodebench'
    elif 'mmlu' in dataset_name.lower() and 'pro' in dataset_name.lower():
        return 'mmlu-pro'
    elif 'mmlu' in dataset_name.lower():
        return 'mmlu'
    else:
        # Generic case: take first part and limit length
        simplified = dataset_name.split('-')[0]
        return simplified[:15]


def generate_experiment_dir_name(args):
    """Generate a clean, hierarchical directory name for experiments."""
    # Create readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    
    # Get shortened names
    model_short = shorten_model_name(args.model)
    
    # Handle dataset mixtures or single dataset
    if args.datasets is not None:
        # For mixtures, create a compact representation
        dataset_parts = []
        for i, (dataset, rate) in enumerate(zip(args.datasets, args.mixture_rates)):
            config = args.dataset_configs[i] if args.dataset_configs else None
            task_name = create_dataset_task_name(dataset, config)
            # Use percentage for rates (e.g., 70 for 0.7)
            rate_pct = int(rate * 100)
            dataset_parts.append(f"{task_name}-{rate_pct}")
        dataset_task = "mix_" + "_".join(dataset_parts)
    else:
        dataset_task = create_dataset_task_name(args.dataset, args.dataset_config)
    
    # Build key parameters string
    key_params = []
    
    # Add batch size
    key_params.append(f"bs{args.batch_size}")
    
    # Add sequence length
    key_params.append(f"len{args.max_length}")
    
    # Add training duration info
    if hasattr(args, 'max_steps') and args.max_steps is not None:
        key_params.append(f"steps{args.max_steps}")
    else:
        epochs = getattr(args, 'num_train_epochs', 1)
        if epochs != 1:  # Only show if not default
            key_params.append(f"ep{epochs}")
    
    # Add gradient accumulation if not default
    if getattr(args, 'gradient_accumulation_steps', 1) > 1:
        key_params.append(f"acc{args.gradient_accumulation_steps}")
    
    # Add multi-GPU info if enabled
    if getattr(args, 'multi_gpu', False):
        key_params.append("multigpu")
    
    # Add save frequency if not default
    if args.save_steps != 500:  # 500 is typical default
        key_params.append(f"save{args.save_steps}")
    
    key_params_str = "_".join(key_params)
    
    # Build final directory name
    # Format: {timestamp}_{model}_{dataset-task}_{mode}_{params}
    dir_name = f"{timestamp}_{model_short}_{dataset_task}_{args.mode}_{key_params_str}"
    
    # Create full path under runs directory
    runs_dir = "runs"
    
    # Ensure runs directory exists
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        print(f"Created runs directory: {runs_dir}")
    
    full_path = os.path.join(runs_dir, dir_name)
    
    return full_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run HuggingFace models on different datasets.")
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B', help='Model name or path (e.g., bert-base-uncased, gpt2, etc.)')
    parser.add_argument('--dataset', type=str, default='stanfordnlp/imdb', help='Dataset name from HuggingFace Datasets (e.g., imdb, squad, glue, etc.)')
    parser.add_argument('--dataset_config', type=str, default=None, help='Dataset config/task name (e.g., for GLUE: cola, sst2, mrpc, etc.)')
    
    # Dataset mixture arguments
    parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                       help='Multiple datasets for mixture training (e.g., stanfordnlp/imdb glue)')
    parser.add_argument('--dataset_configs', type=str, nargs='+', default=None,
                       help='Configs for multiple datasets (e.g., None sst2). Use "None" for datasets without config')
    parser.add_argument('--mixture_rates', type=float, nargs='+', default=None,
                       help='Mixture rates for each dataset (e.g., 0.7 0.3). Should sum to 1.0')
    parser.add_argument('--task', type=str, default='auto', help='Task type (auto, sequence-classification, causal-lm, etc.)')
    parser.add_argument('--mode', type=str, default='sft', choices=['pretrain', 'sft', 'rl'], 
                       help='Training mode: pretrain (next token prediction), sft (supervised fine-tuning), or rl (reinforcement learning with GRPO)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device for training')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], 
                       help='Device to use for training: auto (automatic detection), cuda, mps, or cpu')
    
    # Multi-GPU configuration
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Enable multi-GPU training using DistributedDataParallel (DDP)')
    parser.add_argument('--ddp_backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                       help='Distributed backend for multi-GPU training (nccl for CUDA, gloo for CPU/debugging)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps to simulate larger batch sizes')
    
    # Checkpointing arguments
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X steps')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Limit the total amount of checkpoints to save')
    parser.add_argument('--save_strategy', type=str, default='steps', choices=['no', 'steps', 'epoch'], 
                       help='The checkpoint save strategy to use')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints and logs (if not specified, will be auto-generated)')
    parser.add_argument('--eval_steps', type=int, default=None, 
                       help='Run evaluation every X steps. When specified alone, automatically sets eval_strategy to "steps". For epoch-based evaluation, use --eval_strategy epoch instead')
    parser.add_argument('--eval_max_batches', type=int, default=-1,
                       help='Maximum number of batches to use during evaluation. Use -1 for full evaluation dataset (default: -1)')
    parser.add_argument('--logging_steps', type=int, default=None,
                       help='Log training metrics (loss, grad_norm, lr) every X steps. Defaults to eval_steps if not specified, or 10 if eval_steps is also not set')
    parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['no', 'steps', 'epoch'], 
                       help='The evaluation strategy to use ("epoch" for per-epoch eval, "steps" for step-based eval with --eval_steps, "no" to disable)')
    parser.add_argument('--eval_on_start', action='store_true',
                       help='Run evaluation at the beginning of training (step 0)')
    parser.add_argument('--load_best_model_at_end', action='store_true', 
                       help='Whether to load the best model found during training at the end of training')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss', 
                       help='The metric to use to compare two different models')
    parser.add_argument('--greater_is_better', action='store_true', 
                       help='Whether the `metric_for_best_model` should be maximized or not')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                       help='Path to a specific checkpoint to resume from (or "latest" for the most recent)')
    
    # Training control arguments
    parser.add_argument('--num_train_epochs', type=int, default=1, 
                       help='Total number of training epochs to perform')
    parser.add_argument('--max_steps', type=int, default=None, 
                       help='If set, overrides num_train_epochs and training stops after this many steps')
    
    # Learning rate and optimizer arguments
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Initial learning rate (default: 5e-5)')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 
                               'constant', 'constant_with_warmup', 'inverse_sqrt', 
                               'reduce_lr_on_plateau', 'cosine_with_min_lr'],
                       help='Learning rate scheduler type (default: linear)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of warmup steps (default: 0)')
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                       help='Warmup ratio - fraction of total steps (overrides warmup_steps if > 0)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for AdamW optimizer (default: 0.0)')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                       help='Beta1 for AdamW optimizer (default: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                       help='Beta2 for AdamW optimizer (default: 0.999)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                       help='Epsilon for AdamW optimizer (default: 1e-8)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping (default: 1.0)')
    
    # Mixed precision training arguments
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 mixed precision training (automatically enabled for CUDA if not specified)')
    parser.add_argument('--bf16', action='store_true',
                       help='Use BF16 mixed precision training (recommended for Ampere GPUs and newer)')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Explicitly disable mixed precision training (use FP32)')
    
    # RL-specific arguments
    parser.add_argument('--reward_model', type=str, default=None,
                       help='Path or name of reward model for RL training (if None, will use simple reward function)')
    parser.add_argument('--grpo_beta', type=float, default=0.1,
                       help='Beta parameter for GRPO (controls strength of KL regularization)')
    parser.add_argument('--grpo_group_size', type=int, default=2,
                       help='Group size for GRPO advantage computation')
    parser.add_argument('--max_prompt_length', type=int, default=512,
                       help='Maximum prompt length for RL training')
    parser.add_argument('--max_completion_length', type=int, default=512,
                       help='Maximum completion length for RL training')
    parser.add_argument('--rl_learning_rate', type=float, default=1e-6,
                       help='Learning rate for RL training (typically lower than SFT)')
    parser.add_argument('--rl_warmup_steps', type=int, default=100,
                       help='Number of warmup steps for RL training')
    
    # Attention implementation arguments
    parser.add_argument('--attn_implementation', type=str, default='auto',
                       choices=['auto', 'eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention', 'paged_attention'],
                       help='Attention implementation to use. Options: '
                            'auto (let model choose best), '
                            'eager (vanilla PyTorch, most compatible), '
                            'sdpa (PyTorch scaled dot-product attention, good balance), '
                            'flash_attention_2 (memory efficient, CUDA only), '
                            'flash_attention_3 (latest Flash Attention, CUDA only), '
                            'flex_attention (customizable patterns, PyTorch 2.5+), '
                            'paged_attention (optimized for inference with KV-cache)')
    
    # MOE-specific arguments
    parser.add_argument('--use_flash_attention', action='store_true',
                       help='DEPRECATED: Use --attn_implementation flash_attention_2 instead')
    parser.add_argument('--moe_load_in_4bit', action='store_true',
                       help='Load MOE model in 4-bit quantization for memory efficiency')
    parser.add_argument('--moe_load_in_8bit', action='store_true',
                       help='Load MOE model in 8-bit quantization for memory efficiency')
    parser.add_argument('--moe_expert_parallel', action='store_true',
                       help='Enable expert parallelism for MOE models (requires multi-GPU)')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA (Low-Rank Adaptation) for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha parameter (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                       default=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                       help='Target modules for LoRA adaptation')
    
    # Packing arguments for efficient training with short sequences
    parser.add_argument('--use_packing', action='store_true',
                       help='Enable sequence packing for efficient training with short sequences')
    parser.add_argument('--packing_max_length', type=int, default=None,
                       help='Maximum length for packed sequences (default: same as max_length)')
    parser.add_argument('--return_position_ids', action='store_true', default=True,
                       help='Return position IDs for packed sequences (needed for proper positional encoding)')
    
    # MPS-specific arguments
    parser.add_argument('--disable_mps_fix', action='store_true',
                       help='Disable the MPS-safe trainer fix and use the original trainer even on MPS devices (may cause NaN eval_loss)')
    
    # Multi-dataset evaluation arguments
    parser.add_argument('--separate_mixture_eval', action='store_true', default=True,
                       help='Evaluate each dataset in mixture separately (default: True)')
    parser.add_argument('--log_eval_spread', action='store_true', default=True,
                       help='Log spread metrics (min/max/std) for multi-dataset evaluation (default: True)')
    
    args, unknown = parser.parse_known_args()
    
    # Check if eval_strategy was explicitly set by the user
    eval_strategy_explicitly_set = '--eval_strategy' in sys.argv
    
    # If unknown args exist, re-parse to get error
    if unknown:
        args = parser.parse_args()
    
    # Handle deprecated --use_flash_attention flag
    if args.use_flash_attention:
        print("Warning: --use_flash_attention is deprecated. Using --attn_implementation flash_attention_2 instead.")
        if args.attn_implementation == 'auto':
            args.attn_implementation = 'flash_attention_2'
    
    # Validate mixture arguments
    if args.datasets is not None:
        # Check that all required mixture arguments are provided
        if args.mixture_rates is None:
            raise ValueError("--mixture_rates must be provided when using --datasets")
        
        if len(args.datasets) != len(args.mixture_rates):
            raise ValueError(f"Number of datasets ({len(args.datasets)}) must match number of mixture rates ({len(args.mixture_rates)})")
        
        if args.dataset_configs is not None and len(args.dataset_configs) != len(args.datasets):
            raise ValueError(f"Number of dataset configs ({len(args.dataset_configs)}) must match number of datasets ({len(args.datasets)})")
        
        # Check that mixture rates sum to 1.0 (with small tolerance)
        total_rate = sum(args.mixture_rates)
        if abs(total_rate - 1.0) > 1e-6:
            raise ValueError(f"Mixture rates must sum to 1.0, got {total_rate}")
        
        # Convert "None" strings to actual None values in dataset_configs
        if args.dataset_configs is not None:
            args.dataset_configs = [None if config == "None" else config for config in args.dataset_configs]
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = generate_experiment_dir_name(args)
    
    # If eval_steps is specified but eval_strategy was not explicitly set, automatically switch to 'steps'
    if args.eval_steps is not None and args.eval_strategy == 'epoch' and not eval_strategy_explicitly_set:
        print(f"Note: Auto-setting eval_strategy='steps' because --eval_steps was specified. Use --eval_strategy epoch for epoch-based evaluation")
        args.eval_strategy = 'steps'
    
    return args


def print_checkpointing_config(args):
    """Print checkpointing configuration for user reference."""
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    if args.dataset_config:
        print(f"Dataset config: {args.dataset_config}")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    if getattr(args, 'multi_gpu', False):
        print(f"Multi-GPU: Enabled")
        print(f"DDP Backend: {args.ddp_backend}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    else:
        print(f"Multi-GPU: Disabled")
    print("\n" + "="*50)
    print("OUTPUT DIRECTORY STRUCTURE")
    print("="*50)
    print(f"Experiment directory: {args.output_dir}")
    print(f"  ├── checkpoints/     # Model checkpoints")
    print(f"  ├── logs/            # Textual training logs")
    print(f"  └── tensorboard/     # TensorBoard logs")
    print("\n" + "="*50)
    print("CHECKPOINTING CONFIGURATION")
    print("="*50)
    print(f"Save strategy: {args.save_strategy}")
    if args.save_strategy == 'steps':
        print(f"Save every {args.save_steps} steps")
    print(f"Maximum checkpoints to keep: {args.save_total_limit}")
    print(f"Evaluation strategy: {args.eval_strategy}")
    if args.eval_strategy == 'steps' and args.eval_steps:
        print(f"Evaluate every {args.eval_steps} steps")
    if args.eval_max_batches != -1:
        print(f"Evaluation limited to {args.eval_max_batches} batches")
    else:
        print(f"Evaluation uses full validation set")
    if getattr(args, 'eval_on_start', False):
        print(f"Evaluation at step 0: ENABLED")
    if args.load_best_model_at_end:
        print(f"Will load best model based on: {args.metric_for_best_model}")
        print(f"Greater is better: {args.greater_is_better}")
    if args.resume_from_checkpoint:
        print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    print("="*50 + "\n") 