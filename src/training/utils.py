import torch
import os
import logging
import json
import numpy as np
from datetime import datetime
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import default_data_collator, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint


def check_device_availability(device_choice='auto', multi_gpu=False):
    """Check and report available devices, optionally use specified device."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA device count: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            # Print memory info
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"    Memory: {memory_total:.1f} GB")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Determine device based on user choice or auto-detection
    if device_choice == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
            if multi_gpu and torch.cuda.device_count() > 1:
                print(f"Multi-GPU training enabled with {torch.cuda.device_count()} GPUs")
            elif multi_gpu and torch.cuda.device_count() == 1:
                print("Warning: Multi-GPU requested but only 1 GPU available, using single GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            if multi_gpu:
                print("Warning: Multi-GPU not supported with MPS, using single MPS device")
        else:
            device = "cpu"
            if multi_gpu:
                print("Warning: Multi-GPU requested but no GPUs available, using CPU")
        print(f"Auto-detected device: {device}")
    else:
        device = device_choice
        # Validate the specified device is available
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print(f"Warning: MPS requested but not available, falling back to CPU")
            device = 'cpu'
        print(f"Using specified device: {device}")

    # Return device info dictionary for better handling
    device_info = {
        'device': device,
        'multi_gpu': multi_gpu and device == 'cuda' and torch.cuda.device_count() > 1,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'effective_gpu_count': torch.cuda.device_count() if (multi_gpu and device == 'cuda') else (1 if device in ['cuda', 'mps'] else 0)
    }
    
    if device_info['multi_gpu']:
        print(f"Multi-GPU setup: Using {device_info['gpu_count']} GPUs")
    
    return device_info


def create_training_arguments(mode, batch_size=8, save_steps=500, save_total_limit=3, save_strategy='steps', 
                            load_best_model_at_end=False, metric_for_best_model='eval_loss', 
                            greater_is_better=False, output_dir='./results', eval_steps=None, 
                            eval_strategy='epoch', num_train_epochs=1, max_steps=None,
                            device_info=None, gradient_accumulation_steps=1, ddp_backend='nccl',
                            eval_on_start=False, learning_rate=5e-5, lr_scheduler_type='linear',
                            warmup_steps=0, warmup_ratio=0.0, weight_decay=0.0,
                            adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                            max_grad_norm=1.0, fp16=None, bf16=None, no_mixed_precision=False,
                            logging_steps=None):
    """Create training arguments configuration based on mode.
    
    Args:
        mode: Training mode ('pretrain', 'sft', 'rl')
        batch_size: Batch size per device
        save_steps: Save checkpoint every X steps
        save_total_limit: Maximum number of checkpoints to keep
        save_strategy: Save strategy ('steps', 'epoch', 'no')
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric to use for model selection
        greater_is_better: Whether higher metric is better
        output_dir: Output directory for checkpoints and logs
        eval_steps: Evaluate every X steps
        eval_strategy: Evaluation strategy ('steps', 'epoch', 'no')
        num_train_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides epochs)
        device_info: Device information dictionary
        gradient_accumulation_steps: Number of gradient accumulation steps
        ddp_backend: Distributed training backend
        eval_on_start: Whether to evaluate at start
        learning_rate: Initial learning rate
        lr_scheduler_type: Type of learning rate scheduler
        warmup_steps: Number of warmup steps
        warmup_ratio: Warmup ratio (overrides warmup_steps if > 0)
        weight_decay: Weight decay for AdamW
        adam_beta1: Beta1 for AdamW
        adam_beta2: Beta2 for AdamW
        adam_epsilon: Epsilon for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Use FP16 mixed precision (None=auto, True=force, False=disable)
        bf16: Use BF16 mixed precision (None=auto, True=force, False=disable)
        no_mixed_precision: Explicitly disable mixed precision
        logging_steps: Log training metrics every X steps (defaults to eval_steps, or 10 if not set)
    """
    # Set eval_steps to save_steps if not specified
    if eval_steps is None and eval_strategy == 'steps':
        eval_steps = save_steps
    
    # Set logging_steps to eval_steps if not specified, or 10 if eval_steps is also not set
    if logging_steps is None:
        if eval_steps is not None:
            logging_steps = eval_steps  # Match evaluation frequency
        else:
            logging_steps = 10  # Default fallback
    
    # Default device info if not provided (for backward compatibility)
    if device_info is None:
        device_info = {'device': 'auto', 'multi_gpu': False, 'gpu_count': 0, 'effective_gpu_count': 0}
    
    # Calculate effective batch size for multi-GPU
    effective_batch_size = batch_size
    if device_info['multi_gpu']:
        # For multi-GPU, the effective batch size is per_device_batch_size * num_gpus * gradient_accumulation_steps
        total_effective_batch_size = batch_size * device_info['gpu_count'] * gradient_accumulation_steps
        print(f"Multi-GPU setup: per_device_batch_size={batch_size}, num_gpus={device_info['gpu_count']}, "
              f"gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"Total effective batch size: {total_effective_batch_size}")
    
    # Create subdirectories for organized output
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    # Determine mixed precision settings
    use_fp16 = False
    use_bf16 = False
    
    if not no_mixed_precision:
        is_cuda = torch.cuda.is_available() and device_info['device'] == 'cuda'
        
        if is_cuda:
            # Check if BF16 is explicitly requested or should be auto-enabled
            if bf16 is True:
                # User explicitly requested BF16
                if torch.cuda.is_bf16_supported():
                    use_bf16 = True
                    print("Using BF16 mixed precision training")
                else:
                    print("Warning: BF16 requested but not supported on this GPU. Falling back to FP16.")
                    use_fp16 = True
            elif fp16 is True:
                # User explicitly requested FP16
                use_fp16 = True
                print("Using FP16 mixed precision training")
            elif bf16 is None and fp16 is None:
                # Auto mode: prefer BF16 if available, otherwise FP16
                if torch.cuda.is_bf16_supported():
                    use_bf16 = True
                    print("Auto-detected BF16 support. Using BF16 mixed precision training")
                else:
                    use_fp16 = True
                    print("Using FP16 mixed precision training (BF16 not supported)")
        else:
            # Not CUDA device (MPS or CPU)
            if bf16 is True or fp16 is True:
                print(f"Warning: Mixed precision requested but not supported on {device_info['device']} device. Using FP32.")
    else:
        print("Mixed precision explicitly disabled. Using FP32.")
    
    training_args = {
        'output_dir': checkpoint_dir,            # checkpoints directory
        'per_device_train_batch_size': batch_size,  # batch size per device
        'per_device_eval_batch_size': batch_size,   # batch size for evaluation
        'gradient_accumulation_steps': gradient_accumulation_steps,  # gradient accumulation
        'eval_strategy': eval_strategy,      # evaluation strategy
        'eval_steps': eval_steps,            # evaluate every X steps
        'save_strategy': save_strategy,      # save strategy
        'save_steps': save_steps,            # save checkpoint every X steps
        'save_total_limit': save_total_limit, # limit total checkpoints saved
        'logging_dir': tensorboard_dir,      # directory for tensorboard logs
        'logging_steps': logging_steps,      # Log training metrics at specified intervals
        'fp16': use_fp16,                    # use FP16 mixed precision
        'bf16': use_bf16,                    # use BF16 mixed precision
        'dataloader_pin_memory': False,      # disable pin_memory for MPS
        'dataloader_num_workers': 0,         # disable multiprocessing for MPS
        'use_cpu': False,                    # Allow GPU usage
        # Learning rate and optimizer settings
        'learning_rate': learning_rate,      # Initial learning rate
        'lr_scheduler_type': lr_scheduler_type,  # LR scheduler type
        'warmup_steps': warmup_steps,        # Warmup steps
        'warmup_ratio': warmup_ratio,        # Warmup ratio (overrides warmup_steps if > 0)
        'weight_decay': weight_decay,        # Weight decay for AdamW
        'adam_beta1': adam_beta1,            # AdamW beta1
        'adam_beta2': adam_beta2,            # AdamW beta2
        'adam_epsilon': adam_epsilon,        # AdamW epsilon
        'max_grad_norm': max_grad_norm,      # Gradient clipping
        # Additional arguments for pretraining
        'prediction_loss_only': False,  # Always compute proper metrics during evaluation
        # Checkpointing and model selection
        'load_best_model_at_end': load_best_model_at_end,
        'metric_for_best_model': metric_for_best_model,
        'greater_is_better': greater_is_better,
        'eval_on_start': eval_on_start,  # Evaluate at step 0
        # Ensure learning rate is logged to TensorBoard
        'report_to': ['tensorboard'],        # Report to TensorBoard
    }
    
    # Add DDP-specific arguments for multi-GPU training
    if device_info['multi_gpu']:
        training_args.update({
            'ddp_backend': ddp_backend,          # distributed backend (nccl for CUDA)
            'ddp_find_unused_parameters': False,  # usually False for better performance
            'dataloader_num_workers': 4,         # enable multiprocessing for multi-GPU
            'dataloader_pin_memory': True,       # enable pin_memory for better GPU transfer
        })
        print(f"DDP configuration: backend={ddp_backend}, find_unused_parameters=False")
    elif device_info['device'] == 'cuda':
        # Single GPU optimizations
        training_args.update({
            'dataloader_num_workers': 2,         # some multiprocessing for single GPU
            'dataloader_pin_memory': True,       # enable pin_memory for GPU
        })
    
    # Add training duration parameters
    if max_steps is not None:
        training_args['max_steps'] = max_steps
        print(f"Training will run for {max_steps} steps (overriding epochs)")
    else:
        training_args['num_train_epochs'] = num_train_epochs
        print(f"Training will run for {num_train_epochs} epoch(s)")
    
    # Print learning rate configuration
    print(f"Learning rate configuration:")
    print(f"  Initial LR: {learning_rate}")
    print(f"  Scheduler: {lr_scheduler_type}")
    if warmup_ratio > 0:
        print(f"  Warmup: {warmup_ratio*100:.1f}% of total steps")
    elif warmup_steps > 0:
        print(f"  Warmup: {warmup_steps} steps")
    else:
        print(f"  Warmup: None")
    if weight_decay > 0:
        print(f"  Weight decay: {weight_decay}")
    
    # Validate warmup steps
    if max_steps is not None and warmup_steps >= max_steps:
        print(f"\nWARNING: Warmup steps ({warmup_steps}) >= max steps ({max_steps})")
        print(f"         Model will never exit warmup phase and won't learn properly!")
        adjusted_warmup = max(1, max_steps // 10)
        print(f"         Consider setting warmup_steps to {adjusted_warmup} (10% of max steps)")
        print(f"         Continuing with current settings, but training may not work correctly\n")
    
    return TrainingArguments(**training_args)


def create_data_collator(tokenizer, mode):
    """Create appropriate data collator based on mode."""
    # Use DataCollatorForLanguageModeling for all modes since we're using generative approach
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Use causal language modeling, not masked language modeling
    )


def load_checkpoint_if_exists(model, output_dir, resume_from_checkpoint=None):
    """Load the latest checkpoint if it exists."""
    # Create checkpoints directory path
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    if resume_from_checkpoint is not None:
        if resume_from_checkpoint == "latest":
            # Check if checkpoint directory exists before looking for checkpoints
            if os.path.exists(checkpoint_dir):
                last_checkpoint = get_last_checkpoint(checkpoint_dir)
                if last_checkpoint is not None:
                    print(f"Resuming from latest checkpoint: {last_checkpoint}")
                    return last_checkpoint
                else:
                    print("No checkpoint found, starting training from scratch")
                    return None
            else:
                print(f"Checkpoint directory {checkpoint_dir} does not exist, starting training from scratch")
                return None
        else:
            # Check if the specified checkpoint exists
            if os.path.exists(resume_from_checkpoint):
                print(f"Resuming from specified checkpoint: {resume_from_checkpoint}")
                return resume_from_checkpoint
            else:
                print(f"Specified checkpoint not found: {resume_from_checkpoint}")
                return None
    else:
        # Auto-resume from latest checkpoint if available
        if os.path.exists(checkpoint_dir):
            last_checkpoint = get_last_checkpoint(checkpoint_dir)
            if last_checkpoint is not None:
                print(f"Auto-resuming from checkpoint: {last_checkpoint}")
                return last_checkpoint
            else:
                print("No checkpoint found, starting training from scratch")
                return None
        else:
            print(f"Checkpoint directory {checkpoint_dir} does not exist, starting training from scratch")
            return None


def compute_metrics_for_pretraining(eval_preds):
    """Compute perplexity for language model pretraining."""
    # Note: This function is kept for potential future use but currently
    # the Trainer computes loss internally for language modeling.
    # Perplexity should be calculated from eval_loss in the training loop.
    return {}


def create_trainer(model, training_args, tokenized_dataset, tokenized_eval_dataset, data_collator, eval_max_batches=-1, mode='sft', disable_mps_fix=False, log_eval_spread=True, use_packing=False, callbacks=None, args_dict=None):
    """Create and configure the Trainer.
    
    Args:
        model: The model to train
        training_args: HuggingFace TrainingArguments
        tokenized_dataset: The tokenized training dataset
        tokenized_eval_dataset: The tokenized evaluation dataset (can be dict for multi-eval)
        data_collator: The data collator
        eval_max_batches: Maximum number of batches for evaluation (-1 for full eval)
        mode: Training mode ('pretrain' or 'sft')
        disable_mps_fix: If True, use original trainer even on MPS (may cause NaN eval_loss)
        log_eval_spread: Whether to log spread metrics for multi-dataset evaluation
        use_packing: Whether sequence packing is enabled
        callbacks: List of callbacks to add to the trainer
        args_dict: Original training arguments dictionary for metadata logging
    """
    # Don't use compute_metrics for pretraining - let Trainer handle loss internally
    compute_metrics = None
    
    # Check if we're on MPS device
    try:
        device = next(model.parameters()).device
        is_mps = device.type == 'mps'
    except:
        is_mps = False
    
    # Check if we have multiple eval datasets
    has_multi_eval = isinstance(tokenized_eval_dataset, dict)
    
    # Choose appropriate trainer based on settings
    if has_multi_eval:
        # Use MultiEvalTrainer for multiple evaluation datasets
        try:
            from training.multi_eval_trainer import MultiEvalTrainer
        except ImportError:
            from src.training.multi_eval_trainer import MultiEvalTrainer
        
        trainer = MultiEvalTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_datasets_dict=tokenized_eval_dataset,
            data_collator=data_collator,
            eval_max_batches=eval_max_batches,
            log_eval_spread=log_eval_spread,
            use_packing=use_packing,
            training_args_dict=args_dict,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        print(f"Using MultiEvalTrainer for {len(tokenized_eval_dataset)} separate eval datasets")
        if eval_max_batches > 0:
            print(f"  with max {eval_max_batches} evaluation batches per dataset")
    elif is_mps and not disable_mps_fix:
        # Use MPS-safe trainer to prevent NaN eval loss
        try:
            from training.mps_safe_trainer import MPSSafeTrainer
        except ImportError:
            from src.training.mps_safe_trainer import MPSSafeTrainer
        trainer = MPSSafeTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            eval_max_batches=eval_max_batches,
            training_args_dict=args_dict,
            use_packing=use_packing,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        print(f"Using MPSSafeTrainer for MPS device (prevents NaN eval loss)")
        if eval_max_batches > 0:
            print(f"  with max {eval_max_batches} evaluation batches")
    elif is_mps and disable_mps_fix:
        # User explicitly disabled MPS fix - warn them
        print("⚠️  WARNING: MPS device detected but MPS fix disabled via --disable_mps_fix")
        print("    You may experience NaN eval_loss during evaluation.")
        print("    To enable the fix, remove the --disable_mps_fix flag.")
        # Fall through to use regular trainer selection below
        if eval_max_batches > 0:
            # Use custom trainer with evaluation batch limiting
            try:
                from training.custom_trainer import LimitedEvalTrainer
            except ImportError:
                from src.training.custom_trainer import LimitedEvalTrainer
            trainer = LimitedEvalTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
                eval_max_batches=eval_max_batches,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            print(f"Using LimitedEvalTrainer with max {eval_max_batches} evaluation batches")
        else:
            # Use standard trainer with metadata logging
            try:
                from training.metadata_trainer import MetadataTrainer
            except ImportError:
                from src.training.metadata_trainer import MetadataTrainer
            trainer = MetadataTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=data_collator,
                training_args_dict=args_dict,
                use_packing=use_packing,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            print("Using standard HuggingFace Trainer with metadata logging")
    elif eval_max_batches > 0:
        # Use custom trainer with evaluation batch limiting
        try:
            from training.custom_trainer import LimitedEvalTrainer
        except ImportError:
            from src.training.custom_trainer import LimitedEvalTrainer
        trainer = LimitedEvalTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            eval_max_batches=eval_max_batches,
            training_args_dict=args_dict,
            use_packing=use_packing,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        print(f"Using LimitedEvalTrainer with max {eval_max_batches} evaluation batches")
    else:
        # Use standard trainer with metadata logging
        try:
            from training.metadata_trainer import MetadataTrainer
        except ImportError:
            from src.training.metadata_trainer import MetadataTrainer
        trainer = MetadataTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            training_args_dict=args_dict,
            use_packing=use_packing,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
    
    # Verify the device the model is on
    print(f"Model device after Trainer initialization: {next(model.parameters()).device}")
    
    return trainer


def setup_text_logging(output_dir, args=None):
    """Set up text logging in the logs subdirectory and save arguments to JSON."""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging configuration
    log_file = os.path.join(logs_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logging.info(f"Text logging set up in: {log_file}")
    
    # Save arguments to JSON file if provided
    if args is not None:
        args_file = os.path.join(logs_dir, 'args.json')
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        # Add metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": output_dir,
            "args": args_dict
        }
        
        with open(args_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"Arguments saved to: {args_file}")
        
        # Also save a human-readable version
        args_txt_file = os.path.join(logs_dir, 'args.txt')
        with open(args_txt_file, 'w') as f:
            f.write(f"Experiment Configuration\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"\nArguments:\n")
            f.write(f"{'-'*60}\n")
            for key, value in sorted(args_dict.items()):
                f.write(f"{key:30s}: {value}\n")
        
        logging.info(f"Arguments (readable) saved to: {args_txt_file}")
    
    return logs_dir


def format_tokens(num_tokens):
    """Format token count into human-readable format."""
    if num_tokens < 1000:
        return str(num_tokens)
    elif num_tokens < 1_000_000:
        return f"{num_tokens/1000:.1f}K"
    elif num_tokens < 1_000_000_000:
        return f"{num_tokens/1_000_000:.1f}M"
    else:
        return f"{num_tokens/1_000_000_000:.2f}B"


def collect_training_metadata(model, args, training_args, tokenizer=None, train_dataset=None):
    """Collect comprehensive training metadata.
    
    Args:
        model: The model being trained
        args: Command-line arguments (can be dict or object)
        training_args: HuggingFace TrainingArguments
        tokenizer: Optional tokenizer
        train_dataset: Optional training dataset for size info
    
    Returns:
        Dictionary containing comprehensive training metadata
    """
    # Helper function to get attribute from dict or object
    def get_arg(name, default=None):
        if isinstance(args, dict):
            return args.get(name, default)
        else:
            return getattr(args, name, default)
    
    # Calculate max steps if not explicitly set
    if training_args.max_steps > 0:
        max_steps = training_args.max_steps
    elif train_dataset is not None:
        steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        max_steps = steps_per_epoch * training_args.num_train_epochs
    else:
        max_steps = training_args.max_steps if training_args.max_steps > 0 else 1000  # Default fallback
    
    # Calculate token budget
    # Get max_length from args, with better fallback
    max_length = get_arg('max_length', 1024)  # Default to 1024 instead of 128
    token_budget = (
        training_args.per_device_train_batch_size *
        training_args.gradient_accumulation_steps *
        max_length *
        max_steps
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model config details
    model_config = {}
    if hasattr(model, 'config'):
        config = model.config
        model_config = {
            'hidden_size': getattr(config, 'hidden_size', getattr(config, 'd_model', None)),
            'num_layers': getattr(config, 'num_hidden_layers', 
                                getattr(config, 'n_layer', 
                                      getattr(config, 'num_layers', None))),
            'num_heads': getattr(config, 'num_attention_heads',
                               getattr(config, 'n_head', 
                                     getattr(config, 'num_heads', None))),
            'vocab_size': getattr(config, 'vocab_size', None),
            'model_type': getattr(config, 'model_type', None),
        }
    
    # Get device info
    try:
        device = str(next(model.parameters()).device)
    except:
        device = 'unknown'
    
    # Determine device count and type
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_type = 'cuda'
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = None
    elif torch.backends.mps.is_available():
        device_count = 1
        device_type = 'mps'
        device_name = 'Apple Silicon'
    else:
        device_count = 0
        device_type = 'cpu'
        device_name = 'CPU'
    
    metadata = {
        # Computed values
        'token_budget': token_budget,
        'token_budget_human': format_tokens(token_budget),
        
        # Model info
        'model_name': get_arg('model', 'unknown'),
        'total_parameters': total_params,
        'total_parameters_human': f"{total_params/1e9:.2f}B" if total_params > 1e9 else f"{total_params/1e6:.1f}M",
        'trainable_parameters': trainable_params,
        'trainable_percentage': f"{trainable_params/total_params*100:.1f}%" if total_params > 0 else "0%",
        **model_config,
        
        # Training config - use actual attention implementation if available
        'attention_implementation': get_arg('actual_attn_implementation', get_arg('attn_implementation', 'auto')),
        'mixed_precision': 'bf16' if get_arg('bf16', False) else
                          'fp16' if get_arg('fp16', False) else 'fp32',
        'sequence_packing': get_arg('use_packing', False),
        'max_sequence_length': max_length,  # Use the max_length we calculated above
        'per_device_batch_size': training_args.per_device_train_batch_size,
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'effective_batch_size': training_args.per_device_train_batch_size * 
                                training_args.gradient_accumulation_steps,
        'learning_rate': training_args.learning_rate,
        'lr_scheduler': training_args.lr_scheduler_type,
        'warmup_steps': training_args.warmup_steps,
        'warmup_ratio': training_args.warmup_ratio,
        'optimizer': 'AdamW',  # Default in HF Trainer
        'weight_decay': training_args.weight_decay,
        'adam_beta1': getattr(training_args, 'adam_beta1', 0.9),
        'adam_beta2': getattr(training_args, 'adam_beta2', 0.999),
        'adam_epsilon': getattr(training_args, 'adam_epsilon', 1e-8),
        
        # Hardware
        'device': device,
        'device_type': device_type,
        'device_name': device_name,
        'device_count': device_count,
        
        # Dataset
        'datasets': get_arg('datasets', None) or [get_arg('dataset', 'unknown')],
        'dataset_configs': get_arg('dataset_configs', None),
        'mixture_rates': get_arg('mixture_rates', None),
        'train_dataset_size': len(train_dataset) if train_dataset else None,
        
        # LoRA (if enabled)
        'lora_enabled': get_arg('use_lora', False),
        'lora_r': get_arg('lora_r', None) if get_arg('use_lora', False) else None,
        'lora_alpha': get_arg('lora_alpha', None) if get_arg('use_lora', False) else None,
        'lora_dropout': get_arg('lora_dropout', None) if get_arg('use_lora', False) else None,
        'lora_target_modules': get_arg('lora_target_modules', None) if get_arg('use_lora', False) else None,
        
        # Training plan
        'max_steps': max_steps,
        'num_train_epochs': training_args.num_train_epochs,
        'save_steps': training_args.save_steps,
        'save_strategy': training_args.save_strategy,
        'eval_steps': training_args.eval_steps if training_args.eval_strategy == 'steps' else None,
        'eval_strategy': training_args.eval_strategy,
        'eval_on_start': get_arg('eval_on_start', False),
        'checkpoint_dir': training_args.output_dir,
        'resume_from_checkpoint': get_arg('resume_from_checkpoint', None),
        
        # Additional flags
        'multi_gpu': get_arg('multi_gpu', False),
        'disable_mps_fix': get_arg('disable_mps_fix', False),
        'mode': get_arg('mode', 'sft'),
    }
    
    return metadata


def format_metadata_as_markdown(metadata):
    """Format metadata dictionary as markdown for TensorBoard text display.
    
    Args:
        metadata: Dictionary containing training metadata
    
    Returns:
        Formatted markdown string
    """
    lines = ["# Training Configuration\n"]
    
    # Model Section
    lines.append("\n## Model")
    lines.append(f"- **Name**: {metadata.get('model_name', 'Unknown')}")
    lines.append(f"- **Total Parameters**: {metadata.get('total_parameters_human', 'N/A')}")
    lines.append(f"- **Trainable Parameters**: {metadata.get('trainable_parameters', 'N/A')} ({metadata.get('trainable_percentage', 'N/A')})")
    if metadata.get('model_type'):
        lines.append(f"- **Model Type**: {metadata['model_type']}")
    if metadata.get('hidden_size'):
        lines.append(f"- **Hidden Size**: {metadata['hidden_size']}")
    if metadata.get('num_layers'):
        lines.append(f"- **Layers**: {metadata['num_layers']}")
    if metadata.get('num_heads'):
        lines.append(f"- **Attention Heads**: {metadata['num_heads']}")
    
    # Training Section
    lines.append("\n## Training")
    lines.append(f"- **Token Budget**: {metadata.get('token_budget_human', 'N/A')}")
    lines.append(f"- **Max Steps**: {metadata.get('max_steps', 'N/A')}")
    lines.append(f"- **Effective Batch Size**: {metadata.get('effective_batch_size', 'N/A')}")
    lines.append(f"- **Max Sequence Length**: {metadata.get('max_sequence_length', 'N/A')}")
    lines.append(f"- **Attention Implementation**: {metadata.get('attention_implementation', 'auto')}")
    lines.append(f"- **Mixed Precision**: {metadata.get('mixed_precision', 'fp32')}")
    lines.append(f"- **Sequence Packing**: {'Enabled' if metadata.get('sequence_packing') else 'Disabled'}")
    
    # Optimizer Section
    lines.append("\n## Optimizer")
    lines.append(f"- **Learning Rate**: {metadata.get('learning_rate', 'N/A')}")
    lines.append(f"- **LR Scheduler**: {metadata.get('lr_scheduler', 'N/A')}")
    lines.append(f"- **Warmup Steps**: {metadata.get('warmup_steps', 0)}")
    lines.append(f"- **Weight Decay**: {metadata.get('weight_decay', 0.0)}")
    
    # Hardware Section
    lines.append("\n## Hardware")
    lines.append(f"- **Device**: {metadata.get('device', 'Unknown')}")
    lines.append(f"- **Device Type**: {metadata.get('device_type', 'Unknown')}")
    if metadata.get('device_name'):
        lines.append(f"- **Device Name**: {metadata['device_name']}")
    if metadata.get('device_count', 0) > 0:
        lines.append(f"- **Device Count**: {metadata['device_count']}")
    
    # Dataset Section
    if metadata.get('datasets'):
        lines.append("\n## Dataset")
        lines.append(f"- **Datasets**: {', '.join(metadata['datasets'])}")
        if metadata.get('dataset_configs'):
            lines.append(f"- **Configs**: {', '.join(str(c) for c in metadata['dataset_configs'])}")
        if metadata.get('mixture_rates'):
            lines.append(f"- **Mixture Rates**: {metadata['mixture_rates']}")
        if metadata.get('train_dataset_size'):
            lines.append(f"- **Training Samples**: {metadata['train_dataset_size']:,}")
    
    # LoRA Section (if enabled)
    if metadata.get('lora_enabled'):
        lines.append("\n## LoRA Configuration")
        lines.append(f"- **Rank**: {metadata.get('lora_r', 'N/A')}")
        lines.append(f"- **Alpha**: {metadata.get('lora_alpha', 'N/A')}")
        lines.append(f"- **Dropout**: {metadata.get('lora_dropout', 'N/A')}")
        if metadata.get('lora_target_modules'):
            lines.append(f"- **Target Modules**: {', '.join(metadata['lora_target_modules'])}")
    
    return "\n".join(lines) 