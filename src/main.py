#!/usr/bin/env python3
"""
Main training script for HuggingFace models.
This script orchestrates the training pipeline using modular components.
"""

import os
import json
import torch
from config.args import parse_arguments, print_checkpointing_config
from data.utils import load_and_prepare_dataset, load_and_prepare_dataset_mixture, load_mixture_with_separate_evals, create_tokenize_function, prepare_dataset_for_training, is_mixture_classification_task
from models.utils import setup_tokenizer, determine_task, get_model_class, load_model, configure_model_padding, apply_lora
from training.utils import (
    check_device_availability, create_training_arguments, create_data_collator, 
    load_checkpoint_if_exists, create_trainer, setup_text_logging
)
from training.enhanced_logging import setup_enhanced_logging
from training.rl_utils import (
    create_simple_reward_function, create_grpo_config, prepare_rl_dataset,
    create_grpo_trainer, load_reward_model, create_reward_function_from_model
)


def initialize_distributed_training(device_info):
    """Initialize distributed training if multi-GPU is enabled."""
    if not device_info['multi_gpu']:
        return
    
    # Check if we're already in a distributed environment
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        print(f"Distributed training detected: LOCAL_RANK={local_rank}")
    else:
        print("Multi-GPU enabled but not in distributed environment.")
        print("For optimal multi-GPU performance, consider using:")
        print("  torchrun --nproc_per_node=<num_gpus> main.py --multi_gpu [your_args]")
        print("  or")
        print("  python -m torch.distributed.launch --nproc_per_node=<num_gpus> main.py --multi_gpu [your_args]")
        print("Falling back to DataParallel (less efficient but works)...")


def setup_training_components(args):
    """Setup all training components: dataset, tokenizer, model, and task."""
    print(f"Training mode: {args.mode}")
    
    # Load and prepare dataset(s)
    if args.datasets is not None:
        # Multiple datasets - check if we want separate evaluations
        if getattr(args, 'separate_mixture_eval', True):
            # Load with separate eval sets for each dataset
            train_dataset, eval_datasets_dict, dataset_names = load_mixture_with_separate_evals(
                args.datasets, args.dataset_configs, args.mixture_rates
            )
            eval_dataset = eval_datasets_dict  # Dictionary of eval datasets
        else:
            # Original behavior - use mixed dataset for eval too
            train_dataset = load_and_prepare_dataset_mixture(
                args.datasets, args.dataset_configs, args.mixture_rates
            )
            eval_dataset = train_dataset  # For mixed datasets, use same for eval
        # For task determination, use the first dataset as reference
        primary_dataset = args.datasets[0]
        primary_config = args.dataset_configs[0] if args.dataset_configs else None
    else:
        # Single dataset - load both train and eval
        train_dataset, eval_dataset = load_and_prepare_dataset(
            args.dataset, args.dataset_config, return_eval=True
        )
        primary_dataset = args.dataset
        primary_config = args.dataset_config
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(args.model)
    
    # Determine task based on primary dataset
    task = determine_task(primary_dataset, args.task, args.mode)
    
    # Get model class and load model with MOE support
    model_class = get_model_class(task)
    model, actual_attn_implementation = load_model(
        args.model, 
        task, 
        model_class, 
        device=args.device,
        attn_implementation=getattr(args, 'attn_implementation', 'auto')
    )
    
    # Store actual attention implementation for metadata logging
    args.actual_attn_implementation = actual_attn_implementation
    
    # Configure model padding
    configure_model_padding(model, tokenizer)
    
    # Apply LoRA if requested
    if args.use_lora:
        model = apply_lora(model, args)
    
    return train_dataset, eval_dataset, tokenizer, model, task


def setup_training_pipeline(args, train_dataset, eval_dataset, tokenizer, model, task):
    """Setup the training pipeline with all necessary components."""
    # Setup textual logging and save arguments
    logs_dir = setup_text_logging(args.output_dir, args)
    
    # Setup enhanced logging callback
    logger, enhanced_callback = setup_enhanced_logging(args.output_dir, args)
    
    # Check device availability with multi-GPU support
    device_info = check_device_availability(args.device, getattr(args, 'multi_gpu', False))
    print(f"Model loaded: {model.__class__.__name__}")
    
    # Initialize distributed training if needed
    initialize_distributed_training(device_info)
    
    # Handle RL mode differently
    if args.mode == 'rl':
        return setup_rl_training_pipeline(args, train_dataset, tokenizer, model, device_info)
    
    # metric_for_best_model doesn't need adjustment - MultiEvalTrainer provides eval_loss
    metric_for_best_model = args.metric_for_best_model
    
    # Standard training pipeline for pretrain/sft modes
    # Create training arguments with device info and multi-GPU support
    training_args = create_training_arguments(
        args.mode, 
        batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=args.greater_is_better,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        device_info=device_info,
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
        ddp_backend=getattr(args, 'ddp_backend', 'nccl'),
        eval_on_start=getattr(args, 'eval_on_start', False),
        # Learning rate and optimizer settings
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        # Mixed precision settings
        fp16=getattr(args, 'fp16', None),
        bf16=getattr(args, 'bf16', None),
        no_mixed_precision=getattr(args, 'no_mixed_precision', False),
        # Logging settings
        logging_steps=getattr(args, 'logging_steps', None)
    )
    
    # Check if packing should be used
    use_packing = getattr(args, 'use_packing', False)
    packing_max_length = getattr(args, 'packing_max_length', None) or args.max_length
    
    if use_packing and args.mode == 'pretrain':
        print("\n" + "="*60)
        print("SEQUENCE PACKING ENABLED")
        print("="*60)
        print(f"Packing sequences up to {packing_max_length} tokens")
        
        # Import packing functions
        from data.utils import prepare_dataset_with_packing, create_data_collator_with_packing
        
        # Pack the dataset
        tokenized_dataset = prepare_dataset_with_packing(
            train_dataset,
            tokenizer,
            max_length=packing_max_length,
            num_proc=4
        )
        
        # For packing, use eval dataset without packing for now (TODO: pack eval dataset too)
        tokenize_function = create_tokenize_function(
            tokenizer, task, args.mode, args.max_length, args.dataset, args.dataset_config
        )
        
        # Handle eval dataset tokenization - could be dict (for mixtures) or single dataset
        if isinstance(eval_dataset, dict):
            # For mixed datasets with separate evaluations, tokenize each separately
            tokenized_eval_dataset = {}
            for name, dataset in eval_dataset.items():
                # Create dataset-specific tokenize function for proper prompts
                dataset_tokenize_fn = create_tokenize_function(
                    tokenizer, task, args.mode, args.max_length, 
                    dataset_name=name,  # Use the dataset name for proper prompt creation
                    dataset_config=None
                )
                tokenized_eval_dataset[name] = prepare_dataset_for_training(dataset, dataset_tokenize_fn)
        else:
            # Single dataset evaluation
            tokenized_eval_dataset = prepare_dataset_for_training(eval_dataset, tokenize_function)
        
        # Create packing-aware data collator
        data_collator = create_data_collator_with_packing(
            tokenizer,
            task,
            use_packing=True,
            max_length=packing_max_length,
            return_position_ids=getattr(args, 'return_position_ids', True)
        )
        print("="*60 + "\n")
    else:
        # Standard tokenization without packing
        # Create tokenization function and prepare dataset
        if args.datasets is not None:
            # For mixed datasets, pass all dataset information
            tokenize_function = create_tokenize_function(
                tokenizer, task, args.mode, args.max_length, 
                datasets=args.datasets, dataset_configs=args.dataset_configs
            )
        else:
            # For single dataset, use original approach
            tokenize_function = create_tokenize_function(
                tokenizer, task, args.mode, args.max_length, args.dataset, args.dataset_config
            )
        
        tokenized_dataset = prepare_dataset_for_training(train_dataset, tokenize_function)
        
        # Handle eval dataset tokenization - could be dict or single dataset
        if isinstance(eval_dataset, dict):
            # Tokenize each eval dataset separately
            tokenized_eval_dataset = {}
            for name, dataset in eval_dataset.items():
                print(f"Tokenizing eval dataset: {name}")
                tokenized_eval_dataset[name] = prepare_dataset_for_training(dataset, tokenize_function)
        else:
            tokenized_eval_dataset = prepare_dataset_for_training(eval_dataset, tokenize_function)
        
        # Create standard data collator
        data_collator = create_data_collator(tokenizer, args.mode)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint_if_exists(model, args.output_dir, args.resume_from_checkpoint)
    
    # Create trainer with eval batch limiting if specified
    trainer = create_trainer(
        model, 
        training_args, 
        tokenized_dataset,
        tokenized_eval_dataset,
        data_collator,
        eval_max_batches=args.eval_max_batches,
        mode=args.mode,
        disable_mps_fix=getattr(args, 'disable_mps_fix', False),
        log_eval_spread=getattr(args, 'log_eval_spread', True),
        use_packing=use_packing,
        callbacks=[enhanced_callback],
        args_dict=vars(args)  # Pass the original args as dict for metadata
    )
    
    return trainer, checkpoint


def setup_rl_training_pipeline(args, train_dataset, tokenizer, model, device_info):
    """Setup the RL training pipeline with GRPO."""
    print("Setting up RL training pipeline with GRPO...")
    
    # Determine dataset info for reward function
    if args.datasets is not None:
        primary_dataset = args.datasets[0]
        primary_config = args.dataset_configs[0] if args.dataset_configs else None
    else:
        primary_dataset = args.dataset
        primary_config = args.dataset_config
    
    # Prepare dataset for RL training (create prompt-completion pairs)
    rl_dataset = prepare_rl_dataset(train_dataset, tokenizer, primary_dataset, primary_config)
    print(f"Prepared RL dataset with {len(rl_dataset)} examples")
    
    # Create reward function
    if args.reward_model is not None:
        # Try to load custom reward model
        reward_model, reward_tokenizer = load_reward_model(args.reward_model, device_info['device'])
        if reward_model is not None:
            reward_function = create_reward_function_from_model(
                reward_model, reward_tokenizer, device_info['device']
            )
            print(f"Using custom reward model: {args.reward_model}")
        else:
            # Fallback to simple reward function
            reward_function = create_simple_reward_function(primary_dataset, primary_config)
            print("Using simple reward function (custom model failed to load)")
    else:
        # Use simple reward function
        reward_function = create_simple_reward_function(primary_dataset, primary_config)
        print(f"Using simple reward function for {primary_dataset}")
    
    # Create GRPO configuration
    grpo_config = create_grpo_config(args)
    print(f"GRPO Config: beta={args.grpo_beta}, group_size={args.grpo_group_size}")
    
    # Create GRPO trainer
    trainer = create_grpo_trainer(
        model, tokenizer, rl_dataset, grpo_config, reward_function, device_info
    )
    
    # For RL, checkpoint loading is handled differently
    checkpoint = None  # GRPO trainer handles checkpoint loading internally
    
    return trainer, checkpoint


def main():
    """Main function orchestrating the entire training pipeline."""
    # Parse arguments and show configuration
    args = parse_arguments()
    print_checkpointing_config(args)
    
    # Setup training components
    train_dataset, eval_dataset, tokenizer, model, task = setup_training_components(args)
    
    # Setup training pipeline
    trainer, checkpoint = setup_training_pipeline(args, train_dataset, eval_dataset, tokenizer, model, task)
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Always save final checkpoint after training completes
    print("\nSaving final checkpoint...")
    final_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-final")
    trainer.save_model(final_checkpoint_dir)
    
    # Also save the trainer state (optimizer, scheduler, etc.)
    trainer.save_state()
    print(f"Final checkpoint saved to: {final_checkpoint_dir}")
    
    # Always run final evaluation if we have eval data
    if eval_dataset is not None:
        print("\nRunning final evaluation...")
        final_metrics = trainer.evaluate()
        
        # Log final metrics
        print("\nFinal evaluation metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save metrics to file for later analysis
        import json
        metrics_file = os.path.join(args.output_dir, "final_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Final metrics saved to: {metrics_file}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

