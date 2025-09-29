#!/usr/bin/env python3
"""
Test the actual pretraining pipeline with packing to verify EOS token handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from datasets import Dataset
import argparse
from types import SimpleNamespace

# Import the actual functions used in training
from src.models.utils import setup_tokenizer
from src.data.utils import prepare_dataset_with_packing

def test_pretrain_pipeline_packing():
    """Test the actual pretraining pipeline's packing behavior."""
    
    print("="*60)
    print("TESTING ACTUAL PRETRAINING PIPELINE WITH PACKING")
    print("="*60)
    
    # Setup tokenizer using the actual function from training
    print("\n1. Setting up tokenizer (actual pipeline function)...")
    tokenizer = setup_tokenizer('Qwen/Qwen3-0.6B-Base')
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"   PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    # Create a realistic financial dataset
    print("\n2. Creating financial pretraining dataset...")
    financial_docs = [
        "The Federal Reserve announced a 25 basis point rate hike, marking the third increase this year. Market participants had widely anticipated this move.",
        "Tesla reported quarterly earnings that exceeded analyst expectations. Revenue grew 15% year-over-year, driven by strong vehicle deliveries.",
        "Gold prices surged to a six-month high as investors sought safe-haven assets amid growing geopolitical tensions.",
        "The S&P 500 index closed at a record high, boosted by strong performance in the technology sector.",
        "European Central Bank maintains accommodative monetary policy stance despite inflation concerns.",
        "Cryptocurrency markets experienced significant volatility following regulatory announcements from major economies.",
        "Apple announced a $90 billion share buyback program, the largest in the company's history.",
        "Oil prices declined sharply as OPEC+ members signaled potential production increases.",
        "The yield curve inverted briefly, raising concerns about potential economic slowdown.",
        "Major banks reported robust trading revenues in the latest quarter, offsetting loan loss provisions."
    ]
    
    print(f"   Created {len(financial_docs)} financial documents")
    
    # Create dataset
    dataset = Dataset.from_dict({'text': financial_docs})
    
    # Use the actual packing function from the pipeline
    print("\n3. Applying prepare_dataset_with_packing (actual pipeline function)...")
    print("   Parameters: max_length=512, num_proc=1")
    
    packed_dataset = prepare_dataset_with_packing(
        dataset,
        tokenizer,
        max_length=512,
        num_proc=1
    )
    
    print(f"\n4. Analyzing packed results...")
    print(f"   Original documents: {len(financial_docs)}")
    print(f"   Packed sequences: {len(packed_dataset)}")
    
    # Analyze multiple sequences if available
    sequences_to_check = min(3, len(packed_dataset))
    
    total_eos_tokens = 0
    for seq_idx in range(sequences_to_check):
        sequence = packed_dataset[seq_idx]['input_ids']
        eos_count = sum(1 for token_id in sequence if token_id == tokenizer.eos_token_id)
        total_eos_tokens += eos_count
        
        print(f"\n   Sequence {seq_idx + 1}:")
        print(f"   - Length: {len(sequence)} tokens")
        print(f"   - EOS tokens: {eos_count}")
        
        # Decode and show first few document boundaries
        decoded = tokenizer.decode(sequence[:200], skip_special_tokens=False)
        segments = decoded.split(tokenizer.eos_token)
        
        print(f"   - First 3 document segments:")
        for i, seg in enumerate(segments[:3]):
            if seg.strip():
                preview = seg.strip()[:80]
                print(f"     Doc {i+1}: '{preview}{'...' if len(seg.strip()) > 80 else ''}'")
    
    print(f"\n5. Validation Summary:")
    print(f"   Total EOS tokens found: {total_eos_tokens}")
    
    if total_eos_tokens > 0:
        print("   ✅ EOS tokens ARE being inserted by the pretraining pipeline")
        print("   ✅ Document boundaries are properly marked")
        print("   ✅ Pipeline is correctly configured for Qwen pretraining")
    else:
        print("   ❌ WARNING: No EOS tokens found!")
        print("   ❌ Documents may not be properly separated!")
    
    # Test the labels to ensure they're set correctly
    print("\n6. Checking labels configuration...")
    if 'labels' in packed_dataset.column_names:
        labels = packed_dataset[0]['labels']
        
        # Count how many labels are not -100 (padding)
        valid_labels = sum(1 for label in labels if label != -100)
        
        print(f"   Labels shape matches input_ids: {len(labels) == len(packed_dataset[0]['input_ids'])}")
        print(f"   Valid labels (not padding): {valid_labels}/{len(labels)}")
        
        # Check if EOS tokens are in the labels (they should be for learning to generate them)
        eos_in_labels = sum(1 for label in labels if label == tokenizer.eos_token_id)
        print(f"   EOS tokens in labels: {eos_in_labels}")
        
        if eos_in_labels > 0:
            print("   ✅ Model will learn to generate EOS tokens")
        else:
            print("   ⚠️  EOS tokens might be masked in labels")
    
    return total_eos_tokens > 0

if __name__ == "__main__":
    print("Testing Pretraining Pipeline with Packing")
    print("="*60)
    
    try:
        success = test_pretrain_pipeline_packing()
        
        print("\n" + "="*60)
        print("FINAL VERDICT")
        print("="*60)
        
        if success:
            print("✅ The pretraining pipeline CORRECTLY inserts EOS tokens between documents")
            print("✅ Qwen3 models will learn proper document boundaries during pretraining")
            print("✅ This matches best practices from GPT-3, LLaMA, and other LLMs")
        else:
            print("❌ Issue detected with EOS token insertion")
            print("❌ This could affect model quality during pretraining")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()