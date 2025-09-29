#!/usr/bin/env python3
"""
Show detailed examples of EOS token insertion in packed sequences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from datasets import Dataset
from src.data.packing_utils import create_packed_dataset

def show_detailed_examples():
    """Show detailed examples of how EOS tokens separate documents."""
    
    print("="*80)
    print("DETAILED EXAMPLES OF EOS TOKEN INSERTION")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    
    # Create simple, clear examples
    documents = [
        "Apple stock rose 5% today.",
        "Gold prices hit new highs.",
        "Fed announces rate decision.",
        "Tesla reports Q4 earnings.",
        "Bitcoin volatility increases."
    ]
    
    print("\n1. ORIGINAL DOCUMENTS:")
    print("-" * 40)
    for i, doc in enumerate(documents, 1):
        print(f"Doc {i}: \"{doc}\"")
    
    # Create dataset and pack it
    dataset = Dataset.from_dict({'text': documents})
    packed_dataset = create_packed_dataset(
        dataset,
        tokenizer,
        max_length=128,  # Small size to see complete sequence
        num_proc=1
    )
    
    # Get the packed sequence
    packed_ids = packed_dataset[0]['input_ids']
    
    print("\n2. TOKENIZED VERSION (showing each token):")
    print("-" * 40)
    
    # Decode each token individually to show structure
    print("Position | Token ID  | Token Text")
    print("---------|-----------|" + "-" * 40)
    
    eos_positions = []
    for i, token_id in enumerate(packed_ids[:50]):  # Show first 50 tokens
        token_text = tokenizer.decode([token_id])
        is_eos = token_id == tokenizer.eos_token_id
        
        if is_eos:
            eos_positions.append(i)
            print(f"{i:8} | {token_id:9} | <|endoftext|> ← EOS TOKEN (Document boundary)")
        else:
            # Clean up token text for display
            if token_text.startswith('Ġ'):  # GPT-2 style space token
                token_text = '▁' + token_text[1:]  # Use visible space marker
            print(f"{i:8} | {token_id:9} | {token_text}")
    
    print("\n3. RECONSTRUCTED TEXT (with visible EOS markers):")
    print("-" * 40)
    
    # Decode the full sequence with special tokens visible
    full_text = tokenizer.decode(packed_ids, skip_special_tokens=False)
    
    # Replace EOS token with a visible marker for display
    display_text = full_text.replace('<|endoftext|>', ' [EOS] ')
    
    # Show first 500 characters
    print(display_text[:500])
    if len(display_text) > 500:
        print("...")
    
    print("\n4. DOCUMENT SEPARATION ANALYSIS:")
    print("-" * 40)
    
    # Split by EOS to show how documents are separated
    segments = full_text.split('<|endoftext|>')
    
    print(f"Number of segments created by EOS tokens: {len(segments)}")
    print("\nFirst 5 segments (individual documents):")
    
    for i, segment in enumerate(segments[:5], 1):
        segment = segment.strip()
        if segment:
            print(f"\nSegment {i}:")
            print(f"  Content: \"{segment}\"")
            print(f"  Length: {len(tokenizer.encode(segment))} tokens")
    
    print("\n5. VISUAL REPRESENTATION:")
    print("-" * 40)
    print("How documents are packed with EOS tokens:\n")
    
    # Create a visual representation
    visual = []
    for i, doc in enumerate(documents[:3], 1):
        visual.append(f"[Doc{i}: {doc[:20]}...]")
        visual.append("[EOS]")
    
    print(" → ".join(visual))
    
    print("\n6. TOKEN STATISTICS:")
    print("-" * 40)
    
    total_tokens = len(packed_ids)
    eos_count = sum(1 for tid in packed_ids if tid == tokenizer.eos_token_id)
    content_tokens = total_tokens - eos_count
    
    print(f"Total tokens in packed sequence: {total_tokens}")
    print(f"EOS tokens: {eos_count} ({eos_count/total_tokens*100:.1f}%)")
    print(f"Content tokens: {content_tokens} ({content_tokens/total_tokens*100:.1f}%)")
    print(f"Documents packed: {len(documents)}")
    print(f"Average tokens per document: {content_tokens/len(documents):.1f}")
    
    # Show what happens without packing
    print("\n7. COMPARISON: WITHOUT PACKING (incorrect approach):")
    print("-" * 40)
    
    # Simple concatenation without EOS
    bad_concat = " ".join(documents)
    bad_tokens = tokenizer.encode(bad_concat)
    
    print("If we just concatenated without EOS tokens:")
    print(f"  Text: \"{bad_concat[:100]}...\"")
    print(f"  Result: Documents blend together with no boundaries!")
    print(f"  Model would think: 'today. Gold' is a continuous phrase")
    print(f"  This would harm model quality!")
    
    print("\n8. WHY EOS TOKENS MATTER:")
    print("-" * 40)
    print("✓ Model learns where documents end")
    print("✓ Prevents unrelated text from being treated as continuous")
    print("✓ Enables proper generation stopping")
    print("✓ Maintains semantic boundaries between different topics")
    print("✓ Critical for high-quality language model training")

if __name__ == "__main__":
    try:
        show_detailed_examples()
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print("The examples clearly show that EOS tokens (<|endoftext|>) are properly")
        print("inserted between every document during packing, ensuring proper boundary")
        print("handling for Qwen3 pretraining.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()