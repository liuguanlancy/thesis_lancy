#!/usr/bin/env python3
"""
Example script for pretraining with sequence packing and FlashAttention 2.
Optimized for BookCorpus and other datasets with short sequences.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithFlattening,
)
from datasets import load_dataset
from src.data.packing_utils import (
    DataCollatorForPackedLanguageModeling,
    create_packed_dataset,
    check_flash_attention_availability,
    get_packing_recommendations,
)
import argparse


def main():
    parser = argparse.ArgumentParser(description="Pretrain with sequence packing")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="bookcorpus/bookcorpus")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--use_packing", action="store_true", help="Enable sequence packing")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use FlashAttention 2")
    parser.add_argument("--output_dir", type=str, default="./runs/packed_pretrain")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PRETRAINING WITH SEQUENCE PACKING")
    print("=" * 60)
    
    # Check FlashAttention availability
    print("\nChecking FlashAttention 2 availability...")
    fa_status = check_flash_attention_availability()
    for key, value in fa_status.items():
        print(f"  {key}: {value}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with FlashAttention if requested
    print(f"Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    
    if args.use_flash_attention and fa_status['cuda_available']:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  ✓ Using FlashAttention 2")
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "bookcorpus/bookcorpus":
        dataset = load_dataset(args.dataset, split="train[:10000]")  # Use subset for demo
    else:
        dataset = load_dataset(args.dataset, split="train")
    
    print(f"  Dataset size: {len(dataset)} examples")
    
    # Analyze dataset for packing recommendations
    if 'text' in dataset.column_names:
        sample_texts = dataset.select(range(min(100, len(dataset))))['text']
        avg_length = sum(len(tokenizer.tokenize(t)) for t in sample_texts) / len(sample_texts)
        print(f"  Average sequence length: {avg_length:.1f} tokens")
        
        # Get packing recommendations
        recs = get_packing_recommendations(args.model, args.dataset, avg_length)
        print(f"\nPacking Recommendations:")
        print(f"  Use packing: {recs['use_packing']}")
        print(f"  Recommended max_length: {recs['recommended_max_length']}")
        print(f"  Estimated speedup: {recs['estimated_speedup']:.1f}x")
        print(f"  Reason: {recs['reason']}")
    
    # Prepare dataset
    if args.use_packing:
        print(f"\nPacking sequences into chunks of {args.max_length} tokens...")
        
        # Method 1: Use custom packing
        dataset = create_packed_dataset(
            dataset,
            tokenizer,
            max_length=args.max_length,
            num_proc=4
        )
        
        # Use packing-aware data collator
        data_collator = DataCollatorForPackedLanguageModeling(
            tokenizer=tokenizer,
            max_length=args.max_length,
            return_position_ids=True,
        )
        print(f"  ✓ Packed dataset size: {len(dataset)} sequences")
        
    else:
        print(f"\nTokenizing dataset without packing...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
        )
        
        # Use standard data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=500,
        fp16=False,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4 if args.use_packing else 2,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Sequence packing: {'Enabled' if args.use_packing else 'Disabled'}")
    print(f"  FlashAttention 2: {'Enabled' if args.use_flash_attention else 'Disabled'}")
    print("=" * 60)
    
    trainer.train()
    
    print("\n✓ Training complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()