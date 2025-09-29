#!/usr/bin/env python3
"""
Simplified test to demonstrate SDPA NaN bug on MPS.
This directly tests the attention mechanisms without full training pipeline.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
import warnings

warnings.filterwarnings('ignore')

def test_sdpa_vs_eager():
    """Direct test of SDPA vs Eager attention on MPS."""
    
    print("="*60)
    print("DIRECT SDPA vs EAGER ATTENTION TEST ON MPS")
    print("="*60)
    
    # Check device
    if not torch.backends.mps.is_available():
        print("❌ MPS not available. This test requires Apple Silicon.")
        return
    
    device = torch.device("mps")
    print(f"✅ Testing on MPS device")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Model to test
    model_name = "gpt2"  # Small model for quick testing
    
    # Test texts with different lengths (forces padding)
    texts = [
        "Short",
        "This is a much longer text that will have different padding"
    ]
    
    print(f"\n📝 Test inputs (different lengths to force padding):")
    for i, text in enumerate(texts):
        print(f"   [{i}] '{text}' (length: {len(text.split())} words)")
    
    # Load tokenizer with LEFT padding (critical!)
    print(f"\n🔧 Setting up tokenizer with LEFT padding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # This triggers the bug!
    
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=20)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Padding tokens: {(input_ids == tokenizer.pad_token_id).sum().item()}")
    
    results = {}
    
    # Test 1: Model with SDPA (forced)
    print(f"\n🧪 Test 1: SDPA Attention (forced)")
    print("-"*40)
    try:
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = 'sdpa'
        
        model_sdpa = AutoModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation='sdpa',
            torch_dtype=torch.float32
        ).to(device)
        model_sdpa.eval()
        
        print("   Model loaded with SDPA attention")
        
        with torch.no_grad():
            outputs_sdpa = model_sdpa(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Check for NaN in outputs
            hidden_states = outputs_sdpa.last_hidden_state
            has_nan = torch.isnan(hidden_states).any().item()
            
            if has_nan:
                print(f"   ❌ NaN detected in SDPA outputs!")
                print(f"      First NaN position: {torch.isnan(hidden_states).nonzero()[0].tolist()}")
            else:
                print(f"   ✅ No NaN in SDPA outputs")
                print(f"      Output mean: {hidden_states.mean().item():.6f}")
            
            results['sdpa'] = has_nan
            
    except Exception as e:
        print(f"   ❌ SDPA test failed: {e}")
        results['sdpa'] = None
    
    # Test 2: Model with Eager attention
    print(f"\n🧪 Test 2: Eager Attention")
    print("-"*40)
    try:
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = 'eager'
        
        model_eager = AutoModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation='eager',
            torch_dtype=torch.float32
        ).to(device)
        model_eager.eval()
        
        print("   Model loaded with Eager attention")
        
        with torch.no_grad():
            outputs_eager = model_eager(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Check for NaN in outputs
            hidden_states = outputs_eager.last_hidden_state
            has_nan = torch.isnan(hidden_states).any().item()
            
            if has_nan:
                print(f"   ❌ NaN detected in Eager outputs!")
            else:
                print(f"   ✅ No NaN in Eager outputs")
                print(f"      Output mean: {hidden_states.mean().item():.6f}")
            
            results['eager'] = has_nan
            
    except Exception as e:
        print(f"   ❌ Eager test failed: {e}")
        results['eager'] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results['sdpa'] is not None and results['eager'] is not None:
        if results['sdpa'] and not results['eager']:
            print("✅ BUG CONFIRMED!")
            print("   - SDPA produces NaN with left padding on MPS")
            print("   - Eager attention works correctly")
            print("   - This validates the documented SDPA bug on MPS")
        elif not results['sdpa'] and not results['eager']:
            print("❓ UNEXPECTED: Neither produced NaN")
            print("   - Bug may be fixed in your PyTorch version")
            print(f"   - Current PyTorch: {torch.__version__}")
        elif results['sdpa'] and results['eager']:
            print("❌ Both produced NaN - different issue")
        else:
            print("❓ Eager produced NaN but SDPA didn't - very unexpected")
    else:
        print("❌ Tests incomplete")
    
    return results

def test_with_model_output():
    """Test using actual model forward pass with loss calculation."""
    
    print("\n" + "="*60)
    print("TESTING WITH LOSS CALCULATION")
    print("="*60)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return
    
    device = torch.device("mps")
    model_name = "gpt2"
    
    # Create tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Create batch with padding
    texts = ["Short text", "This is a longer text for testing"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=20)
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = input_ids.clone()  # For loss calculation
    
    print(f"Testing with batch size: {input_ids.shape[0]}")
    print(f"Sequence length: {input_ids.shape[1]}")
    
    # Test SDPA
    print("\n🧪 SDPA with loss calculation:")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='sdpa',
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"   ❌ Loss is NaN!")
            else:
                print(f"   ✅ Loss is valid: {loss.item():.4f}")
                
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Eager
    print("\n🧪 Eager with loss calculation:")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='eager',
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"   ❌ Loss is NaN!")
            else:
                print(f"   ✅ Loss is valid: {loss.item():.4f}")
                
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    # Run simplified test
    results = test_sdpa_vs_eager()
    
    # Run with loss calculation
    test_with_model_output()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)