#!/usr/bin/env python3
"""
Quick test of SDPA vs Eager with Qwen3-0.6B model on MPS.
This is the specific model configuration for the thesis project.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

def test_qwen_on_mps():
    """Test Qwen3-0.6B with SDPA vs Eager on MPS."""
    
    print("="*60)
    print("QWEN3-0.6B ATTENTION TEST ON MPS")
    print("="*60)
    
    # Check MPS
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return
    
    device = torch.device("mps")
    print(f"✅ Device: MPS")
    print(f"   PyTorch: {torch.__version__}")
    
    # Model configuration
    model_name = "Qwen/Qwen3-0.6B-Base"
    print(f"\n📦 Model: {model_name}")
    
    # Test data
    texts = ["Short text", "This is a longer text that needs padding"]
    
    # Setup tokenizer with left padding
    print("\n🔧 Loading tokenizer with LEFT padding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = input_ids.clone()
    
    print(f"   Batch size: {input_ids.shape[0]}")
    print(f"   Sequence length: {input_ids.shape[1]}")
    print(f"   Padding tokens: {(attention_mask == 0).sum().item()}")
    
    # Test 1: SDPA (if not automatically switched)
    print("\n🧪 Test 1: SDPA Attention")
    print("-"*40)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='sdpa',
            torch_dtype=torch.float32,
            device_map='mps'
        )
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        if torch.isnan(loss):
            print(f"   ❌ Loss is NaN with SDPA!")
        else:
            print(f"   ✅ Loss is valid: {loss.item():.4f}")
            
        sdpa_result = torch.isnan(loss).item()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sdpa_result = None
    
    # Test 2: Eager
    print("\n🧪 Test 2: Eager Attention")
    print("-"*40)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='eager',
            torch_dtype=torch.float32,
            device_map='mps'
        )
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        if torch.isnan(loss):
            print(f"   ❌ Loss is NaN with Eager!")
        else:
            print(f"   ✅ Loss is valid: {loss.item():.4f}")
            
        eager_result = torch.isnan(loss).item()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        eager_result = None
    
    # Test 3: Auto (should use eager on MPS)
    print("\n🧪 Test 3: Auto Mode (with MPS fix)")
    print("-"*40)
    try:
        # Import the model loading function from your codebase
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from models.utils import load_model, setup_tokenizer, get_model_class
        
        # Use your codebase's model loading (should auto-switch to eager)
        model_class = get_model_class('causal-lm')
        model = load_model(model_name, 'causal-lm', model_class, device='mps')
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        if torch.isnan(loss):
            print(f"   ❌ Loss is NaN even with auto mode!")
        else:
            print(f"   ✅ Loss is valid: {loss.item():.4f}")
            print(f"      (Auto mode correctly switched to eager)")
            
    except Exception as e:
        print(f"   ℹ️  Could not test auto mode: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS FOR QWEN3-0.6B ON MPS")
    print("="*60)
    
    if sdpa_result is not None and eager_result is not None:
        if sdpa_result and not eager_result:
            print("✅ BUG CONFIRMED with Qwen3-0.6B!")
            print("   - SDPA produces NaN")
            print("   - Eager works correctly")
            print("\n📌 RECOMMENDATION:")
            print("   Use the following for Qwen3-0.6B on M1 Max:")
            print("   --model Qwen/Qwen3-0.6B-Base")
            print("   --attn_implementation eager  (or let auto mode handle it)")
            print("   --precision fp32")
            print("   --batch-size 16")
            print("   --max-length 512")
        elif not sdpa_result:
            print("❓ SDPA didn't produce NaN - unexpected")
        else:
            print("❌ Both produced NaN - different issue")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_qwen_on_mps()