#!/usr/bin/env python3
"""
Test script for Qwen3 model support.
Tests loading and training of Qwen3 models < 7B parameters.
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.utils import setup_tokenizer, load_model, get_model_class, apply_lora
from src.data.utils import load_and_prepare_dataset, create_tokenize_function, prepare_dataset_for_training
from src.training.utils import check_device_availability, create_training_arguments, create_data_collator, create_trainer
from src.config.args import parse_arguments

def test_qwen3_models():
    """Test Qwen3 models < 7B with 5 training steps."""
    
    # Models to test (all under 5B)
    # Including both Base (pretrained only) and post-trained versions
    test_models = [
        "Qwen/Qwen3-0.6B-Base",    # Pretrained only - 600M parameters
        "Qwen/Qwen3-0.6B",         # Post-trained - 600M parameters
        "Qwen/Qwen3-1.7B-Base",    # Pretrained only - 1.7B parameters
        "Qwen/Qwen3-1.7B",         # Post-trained - 1.7B parameters
        "Qwen/Qwen3-4B-Base",      # Pretrained only - 4B parameters
        "Qwen/Qwen3-4B",           # Post-trained - 4B parameters
        "Qwen/Qwen3-4B-Instruct-2507",  # 4B instruct variant (latest version)
    ]
    
    print("=" * 60)
    print("Testing Qwen3 Model Support (<5B models)")
    print("=" * 60)
    
    results = []
    
    for model_name in test_models:
        print(f"\nTesting: {model_name}")
        print("-" * 40)
        
        try:
            # Step 1: Setup tokenizer
            print("1. Loading tokenizer...")
            tokenizer = setup_tokenizer(model_name)
            print(f"   ✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
            
            # Step 2: Determine device
            print("2. Checking device...")
            device_info = check_device_availability('auto', multi_gpu=False)
            device = device_info['device']
            print(f"   ✓ Device: {device}")
            
            # Step 3: Load model
            print("3. Loading model...")
            model_class = get_model_class('causal-lm')
            model, _ = load_model(model_name, 'causal-lm', model_class, device=device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   ✓ Model loaded: {total_params/1e9:.2f}B parameters")
            
            # Step 4: Apply LoRA for efficiency
            print("4. Applying LoRA (rank=4)...")
            class Args:
                lora_r = 4
                lora_alpha = 8
                lora_dropout = 0.1
                lora_target_modules = None
                moe_load_in_4bit = False
                moe_load_in_8bit = False
            
            args = Args()
            model = apply_lora(model, args)
            
            # Step 5: Load a small dataset (IMDB for testing)
            print("5. Loading test dataset (IMDB)...")
            dataset = load_and_prepare_dataset('stanfordnlp/imdb')
            
            # Take only 100 samples for quick testing
            dataset = dataset.select(range(min(100, len(dataset))))
            
            # Step 6: Tokenize dataset
            print("6. Tokenizing dataset...")
            tokenize_function = create_tokenize_function(
                tokenizer=tokenizer,
                max_length=128,  # Short sequences for testing
                task='causal-lm',
                dataset_name='stanfordnlp/imdb',
                dataset_config=None,
                mode='sft'
            )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )
            
            # Step 7: Create training components
            print("7. Setting up training...")
            training_args = create_training_arguments(
                mode='sft',  # Required first parameter
                batch_size=2,
                save_steps=10,
                save_total_limit=1,
                save_strategy='steps',
                load_best_model_at_end=False,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                output_dir=f"./tests/outputs/functional/qwen3_test_{model_name.split('/')[-1]}",
                eval_steps=10,
                eval_strategy='no',  # Skip evaluation for quick test
                num_train_epochs=1,
                max_steps=5,  # Only 5 steps for testing
                device_info=device_info,
                gradient_accumulation_steps=1,
                ddp_backend='nccl'
            )
            
            data_collator = create_data_collator(tokenizer, 'causal-lm')
            
            # Step 8: Create trainer and run 5 steps
            print("8. Running 5 training steps...")
            trainer = create_trainer(
                model=model,
                training_args=training_args,
                tokenized_dataset=tokenized_dataset,
                data_collator=data_collator
            )
            
            # Train for 5 steps
            train_result = trainer.train()
            
            # Check if training completed
            if train_result.global_step == 5:
                print(f"   ✓ Training completed: {train_result.global_step} steps")
                print(f"   ✓ Final loss: {train_result.training_loss:.4f}")
                results.append({
                    'model': model_name,
                    'status': 'SUCCESS',
                    'params': f"{total_params/1e9:.2f}B",
                    'loss': train_result.training_loss
                })
            else:
                print(f"   ⚠ Training incomplete: {train_result.global_step}/5 steps")
                results.append({
                    'model': model_name,
                    'status': 'PARTIAL',
                    'params': f"{total_params/1e9:.2f}B",
                    'loss': train_result.training_loss
                })
            
            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results.append({
                'model': model_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for result in results:
        if result['status'] == 'SUCCESS':
            print(f"✓ {result['model']}: SUCCESS ({result['params']}, loss={result['loss']:.4f})")
        elif result['status'] == 'PARTIAL':
            print(f"⚠ {result['model']}: PARTIAL ({result['params']}, loss={result['loss']:.4f})")
        else:
            print(f"✗ {result['model']}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Overall result
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nOverall: {success_count}/{len(test_models)} models tested successfully")
    
    return results


def test_qwen3_features():
    """Test Qwen3-specific features."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Specific Features")
    print("=" * 60)
    
    # Test smallest model for features
    model_name = "Qwen/Qwen3-0.6B-Base"
    
    try:
        print(f"\nTesting features with {model_name}")
        
        # Test tokenizer with trust_remote_code
        print("1. Testing tokenizer with trust_remote_code...")
        tokenizer = setup_tokenizer(model_name)
        
        # Test special tokens
        test_text = "Explain quantum computing in simple terms."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   ✓ Tokenization works: {len(tokens)} tokens")
        
        # Test model configuration detection
        print("2. Testing model type detection...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"   ✓ Model type: {config.model_type}")
        print(f"   ✓ Hidden size: {config.hidden_size}")
        print(f"   ✓ Num layers: {config.num_hidden_layers}")
        print(f"   ✓ Num attention heads: {config.num_attention_heads}")
        
        # Test context length
        if hasattr(config, 'max_position_embeddings'):
            print(f"   ✓ Max context length: {config.max_position_embeddings}")
        elif hasattr(config, 'max_seq_len'):
            print(f"   ✓ Max context length: {config.max_seq_len}")
        
        print("\n✓ Qwen3 features test completed")
        
    except Exception as e:
        print(f"\n✗ Qwen3 features test failed: {str(e)}")


if __name__ == "__main__":
    print("Starting Qwen3 Model Tests")
    print("This will test Qwen3 models < 7B with 5 training steps each")
    print("Expected time: ~5-10 minutes per model\n")
    
    # Run main model tests
    results = test_qwen3_models()
    
    # Run feature tests
    test_qwen3_features()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)