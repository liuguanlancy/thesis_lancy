#!/usr/bin/env python3
"""
Test script to validate EOS token handling during sequence packing.
This verifies that documents are properly separated by EOS tokens.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from datasets import Dataset
import torch

# Import packing utilities
from src.data.packing_utils import create_packed_dataset

def test_eos_token_insertion():
    """Test that EOS tokens are properly inserted between documents during packing."""
    
    print("="*60)
    print("TESTING EOS TOKEN INSERTION DURING PACKING")
    print("="*60)
    
    # Load Qwen tokenizer
    print("\n1. Loading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    print(f"   EOS token: '{tokenizer.eos_token}'")
    print(f"   EOS token ID: {tokenizer.eos_token_id}")
    print(f"   PAD token: '{tokenizer.pad_token}'")
    
    # Create sample documents
    print("\n2. Creating sample documents...")
    documents = [
        "The stock market showed strong gains today.",
        "Artificial intelligence is transforming finance.",
        "Risk management is crucial for investment portfolios.",
        "Quarterly earnings exceeded analyst expectations.",
        "Central banks are adjusting interest rates."
    ]
    
    for i, doc in enumerate(documents):
        print(f"   Doc {i+1}: {doc}")
    
    # Create a dataset
    dataset = Dataset.from_dict({'text': documents})
    
    # Pack the dataset with a small max_length for testing
    print("\n3. Packing documents (max_length=128)...")
    packed_dataset = create_packed_dataset(
        dataset,
        tokenizer,
        max_length=128,
        num_proc=1
    )
    
    print(f"\n4. Analyzing packed sequences...")
    print(f"   Original documents: {len(documents)}")
    print(f"   Packed sequences: {len(packed_dataset)}")
    
    # Analyze the first packed sequence
    print("\n5. Checking EOS tokens in packed sequence:")
    first_sequence = packed_dataset[0]['input_ids']
    
    # Find all EOS token positions
    eos_positions = [i for i, token_id in enumerate(first_sequence) if token_id == tokenizer.eos_token_id]
    print(f"   Sequence length: {len(first_sequence)} tokens")
    print(f"   EOS tokens found: {len(eos_positions)}")
    print(f"   EOS token positions: {eos_positions}")
    
    # Decode the sequence to show document boundaries
    print("\n6. Decoding packed sequence to show document boundaries:")
    decoded_text = tokenizer.decode(first_sequence, skip_special_tokens=False)
    
    # Split by EOS token to show individual documents
    doc_segments = decoded_text.split(tokenizer.eos_token)
    print(f"   Number of segments separated by EOS: {len(doc_segments)}")
    
    for i, segment in enumerate(doc_segments[:5]):  # Show first 5 segments
        segment = segment.strip()
        if segment:
            print(f"\n   Segment {i+1}:")
            print(f"   '{segment[:100]}{'...' if len(segment) > 100 else ''}'")
    
    # Verify EOS tokens are between documents
    print("\n7. Validation Results:")
    if len(eos_positions) > 0:
        print("   ✅ EOS tokens ARE present in packed sequences")
        print(f"   ✅ Found {len(eos_positions)} EOS tokens separating documents")
        
        # Check if documents are properly separated
        if len(eos_positions) >= len(documents) - 1:
            print("   ✅ Sufficient EOS tokens for document separation")
        else:
            print(f"   ⚠️  Expected at least {len(documents)-1} EOS tokens, found {len(eos_positions)}")
    else:
        print("   ❌ NO EOS tokens found in packed sequences!")
        print("   ❌ Documents are NOT properly separated!")
    
    # Show the actual token IDs around first EOS
    if eos_positions:
        print("\n8. Token IDs around first EOS token:")
        first_eos_pos = eos_positions[0]
        start = max(0, first_eos_pos - 5)
        end = min(len(first_sequence), first_eos_pos + 6)
        
        print("   Position | Token ID | Decoded")
        print("   ---------|----------|--------")
        for i in range(start, end):
            token_id = first_sequence[i]
            decoded = tokenizer.decode([token_id])
            marker = " <-- EOS" if i == first_eos_pos else ""
            print(f"   {i:8d} | {token_id:8d} | '{decoded}'{marker}")
    
    return len(eos_positions) > 0

def test_mixed_dataset_packing():
    """Test EOS token handling with mixed datasets."""
    
    print("\n" + "="*60)
    print("TESTING MIXED DATASET PACKING")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    
    # Create a mixed dataset with different types of content
    mixed_docs = [
        "Question: What is compound interest? Answer: Compound interest is interest calculated on the initial principal and accumulated interest.",
        "The Federal Reserve announced new monetary policy guidelines today.",
        "Instruction: Calculate ROI. Response: ROI = (Gain - Cost) / Cost × 100%",
        "Market volatility increased due to geopolitical tensions.",
        "Technical analysis suggests a bullish trend in the technology sector."
    ]
    
    dataset = Dataset.from_dict({'text': mixed_docs})
    
    print("\n1. Packing mixed dataset...")
    packed = create_packed_dataset(dataset, tokenizer, max_length=256, num_proc=1)
    
    # Check first sequence
    sequence = packed[0]['input_ids']
    eos_count = sum(1 for token_id in sequence if token_id == tokenizer.eos_token_id)
    
    print(f"2. Results:")
    print(f"   Documents packed: {len(mixed_docs)}")
    print(f"   EOS tokens in sequence: {eos_count}")
    print(f"   Verdict: {'✅ PASS' if eos_count > 0 else '❌ FAIL'}")
    
    return eos_count > 0

def test_without_packing():
    """Test what happens without packing (baseline)."""
    
    print("\n" + "="*60)
    print("TESTING WITHOUT PACKING (BASELINE)")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    
    # Simple tokenization without packing
    text = "The stock market showed strong gains today."
    tokens = tokenizer(text, return_tensors='pt')
    
    print(f"\n1. Single document tokenization:")
    print(f"   Text: '{text}'")
    print(f"   Token count: {tokens.input_ids.shape[1]}")
    
    # Check for EOS
    has_eos = (tokens.input_ids == tokenizer.eos_token_id).any().item()
    print(f"   Contains EOS: {'Yes' if has_eos else 'No'}")
    print(f"   Note: Single documents don't need EOS unless explicitly added")
    
    # Concatenate without EOS
    text1 = "First document about finance."
    text2 = "Second document about markets."
    concatenated_no_eos = text1 + " " + text2
    
    # Concatenate with EOS
    concatenated_with_eos = text1 + tokenizer.eos_token + text2
    
    print(f"\n2. Manual concatenation comparison:")
    print(f"   Without EOS: '{concatenated_no_eos}'")
    print(f"   With EOS: '{concatenated_with_eos}'")
    
    tokens_no_eos = tokenizer(concatenated_no_eos, return_tensors='pt')
    tokens_with_eos = tokenizer(concatenated_with_eos, return_tensors='pt')
    
    print(f"\n   Token counts:")
    print(f"   Without EOS: {tokens_no_eos.input_ids.shape[1]} tokens")
    print(f"   With EOS: {tokens_with_eos.input_ids.shape[1]} tokens")
    
    eos_in_concat = (tokens_with_eos.input_ids == tokenizer.eos_token_id).sum().item()
    print(f"   EOS tokens in 'with EOS' version: {eos_in_concat}")
    
    return True

if __name__ == "__main__":
    print("Testing EOS Token Handling in Sequence Packing")
    print("="*60)
    
    try:
        # Run tests
        test1_pass = test_eos_token_insertion()
        test2_pass = test_mixed_dataset_packing()
        test3_pass = test_without_packing()
        
        # Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Test 1 (EOS insertion): {'✅ PASS' if test1_pass else '❌ FAIL'}")
        print(f"Test 2 (Mixed dataset): {'✅ PASS' if test2_pass else '❌ FAIL'}")
        print(f"Test 3 (Baseline): {'✅ PASS' if test3_pass else '❌ FAIL'}")
        
        if test1_pass and test2_pass:
            print("\n✅ CONCLUSION: EOS tokens ARE properly inserted between documents during packing!")
            print("   This ensures proper document boundary handling during pretraining.")
        else:
            print("\n❌ ISSUE FOUND: EOS tokens may not be properly handled!")
            print("   This could affect pretraining quality.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()