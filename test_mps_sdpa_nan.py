#!/usr/bin/env python3
"""
Test script to demonstrate SDPA NaN bug on MPS devices.
This test proves that SDPA attention causes NaN during evaluation on MPS,
while eager attention works correctly.

Run: python test_mps_sdpa_nan.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_device():
    """Check if MPS is available and being used."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS device detected and available")
        print(f"   PyTorch version: {torch.__version__}")
        return device
    else:
        print("❌ MPS device not available. This test requires Apple Silicon Mac.")
        print("   Detected device: CPU" if not torch.cuda.is_available() else "CUDA")
        sys.exit(1)

def create_test_data():
    """Create test data with varying lengths to ensure padding."""
    # Create texts of different lengths to force padding
    texts = [
        "Short text",
        "This is a medium length text that needs some padding",
        "This is a much longer text that will require less padding compared to the short one",
        "Another short one",
        "Medium sized text for testing purposes",
    ]
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset

def test_attention_implementation(attn_implementation, model_name="gpt2", force_sdpa=False):
    """Test a specific attention implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {attn_implementation} attention")
    if force_sdpa:
        print("         (Forcing SDPA even on MPS)")
    print(f"{'='*60}")
    
    device = check_device()
    
    # Load tokenizer with left padding (critical for triggering the bug)
    print("\n1. Loading tokenizer with LEFT padding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Critical for triggering the bug!
    print(f"   ✓ Tokenizer loaded with padding_side = '{tokenizer.padding_side}'")
    
    # Load model with specific attention implementation
    print(f"\n2. Loading model with {attn_implementation} attention...")
    
    try:
        if attn_implementation == 'auto' and not force_sdpa:
            # Let the model loader decide (should use eager on MPS)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use FP32 for MPS stability
                device_map='mps'
            )
        elif attn_implementation == 'sdpa' or force_sdpa:
            # Force SDPA (should cause NaN)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map='mps',
                attn_implementation='sdpa'  # Force SDPA
            )
        else:  # eager
            # Force eager attention (should work)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map='mps',
                attn_implementation='eager'  # Force eager
            )
        
        # Check what attention was actually used
        actual_attn = getattr(model.config, '_attn_implementation', 'unknown')
        print(f"   ✓ Model loaded")
        print(f"   ✓ Actual attention implementation: {actual_attn}")
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return None
    
    # Prepare data
    print("\n3. Preparing test data...")
    dataset = create_test_data()
    
    # Tokenize with padding
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=64,  # Short for quick testing
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    print(f"   ✓ Dataset tokenized: {len(tokenized_dataset)} examples")
    
    # Create data collator for causal LM (critical for the bug)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=None
    )
    
    # Set up training arguments for EVALUATION
    print("\n4. Setting up evaluation...")
    training_args = TrainingArguments(
        output_dir="./test_output",
        per_device_eval_batch_size=2,
        do_eval=True,
        do_train=False,
        no_cuda=True,  # Force MPS/CPU
        fp16=False,  # Use FP32 for MPS
        bf16=False,
        remove_unused_columns=False,
        use_cpu=False if torch.backends.mps.is_available() else True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Run evaluation
    print("\n5. Running evaluation...")
    print("   This is where NaN should appear with SDPA...")
    
    try:
        # Set to eval mode (critical for the bug)
        model.eval()
        
        # Run evaluation
        with torch.no_grad():
            eval_results = trainer.evaluate()
        
        eval_loss = eval_results.get('eval_loss', None)
        
        print("\n" + "="*40)
        if eval_loss is not None:
            if torch.isnan(torch.tensor(eval_loss)):
                print(f"❌ RESULT: NaN loss detected!")
                print(f"   eval_loss = {eval_loss}")
                print(f"   This confirms SDPA causes NaN on MPS!")
            else:
                print(f"✅ RESULT: Valid loss computed!")
                print(f"   eval_loss = {eval_loss:.4f}")
                print(f"   {attn_implementation} attention works correctly!")
        else:
            print("❓ RESULT: No loss computed")
        print("="*40)
        
        return eval_loss
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests to demonstrate the SDPA NaN bug."""
    print("\n" + "="*80)
    print("MPS SDPA NaN BUG DEMONSTRATION")
    print("="*80)
    print("\nThis test demonstrates that:")
    print("1. SDPA attention causes NaN on MPS during evaluation")
    print("2. Eager attention works correctly")
    print("3. The auto mode (with fix) switches to eager and works")
    
    results = {}
    
    # Test 1: Force SDPA (should produce NaN)
    print("\n\n🔬 TEST 1: FORCE SDPA (Expected: NaN)")
    try:
        # We need to bypass the automatic fix
        loss = test_attention_implementation('sdpa', force_sdpa=True)
        results['sdpa_forced'] = loss
    except Exception as e:
        print(f"Test 1 failed: {e}")
        results['sdpa_forced'] = 'error'
    
    # Test 2: Use eager explicitly (should work)
    print("\n\n🔬 TEST 2: EAGER ATTENTION (Expected: Valid loss)")
    try:
        loss = test_attention_implementation('eager')
        results['eager'] = loss
    except Exception as e:
        print(f"Test 2 failed: {e}")
        results['eager'] = 'error'
    
    # Test 3: Use auto mode (should auto-switch to eager on MPS)
    print("\n\n🔬 TEST 3: AUTO MODE (Expected: Valid loss via eager)")
    try:
        loss = test_attention_implementation('auto')
        results['auto'] = loss
    except Exception as e:
        print(f"Test 3 failed: {e}")
        results['auto'] = 'error'
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\nResults:")
    for test, loss in results.items():
        if loss == 'error':
            status = "❌ ERROR"
        elif loss is None:
            status = "❓ No result"
        elif torch.isnan(torch.tensor(loss) if not isinstance(loss, torch.Tensor) else loss):
            status = "❌ NaN (Bug confirmed!)"
        else:
            status = f"✅ Valid ({loss:.4f})"
        print(f"  {test:15} : {status}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Check if bug was demonstrated
    sdpa_nan = results.get('sdpa_forced')
    eager_valid = results.get('eager')
    
    if sdpa_nan is not None and eager_valid is not None:
        sdpa_is_nan = torch.isnan(torch.tensor(sdpa_nan) if not isinstance(sdpa_nan, torch.Tensor) else sdpa_nan)
        eager_is_valid = not torch.isnan(torch.tensor(eager_valid) if not isinstance(eager_valid, torch.Tensor) else eager_valid)
        
        if sdpa_is_nan and eager_is_valid:
            print("\n✅ BUG CONFIRMED: SDPA causes NaN on MPS while eager works!")
            print("   This validates the documented issue and the implemented fix.")
        elif not sdpa_is_nan:
            print("\n❓ UNEXPECTED: SDPA did not produce NaN.")
            print("   This could mean:")
            print("   - The bug has been fixed in your PyTorch version")
            print("   - Different conditions are needed to trigger it")
        elif not eager_is_valid:
            print("\n❌ UNEXPECTED: Even eager attention produced NaN!")
            print("   This suggests a different issue.")
    else:
        print("\n❓ Tests incomplete - cannot draw conclusions")
    
    print("\nTest complete!")
    print("="*80)

if __name__ == "__main__":
    main()