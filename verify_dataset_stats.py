#!/usr/bin/env python3
"""
Verify dataset statistics for thesis tables 3.3 and 3.4.
Uses HuggingFace API for example counts and samples 1000 examples for token estimation.
"""

from datasets import load_dataset, get_dataset_config_info
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Current values from tables 3.3 and 3.4
CURRENT_VALUES = {
    'virattt/financial-qa-10K': {'examples': '7.1K', 'tokens': '3.5M', 'config': None},
    'FinGPT/fingpt-sentiment-train': {'examples': '76.8K', 'tokens': '19.1M', 'config': None},
    'gbharti/finance-alpaca': {'examples': '68.9K', 'tokens': '17.2M', 'config': None},
    'LLukas22/fiqa': {'examples': '17.4K', 'tokens': '4.3M', 'config': None},
    'zeroshot/twitter-financial-news-sentiment': {'examples': '1.1K', 'tokens': '0.3M', 'config': None},
    'JanosAudran/financial-reports-sec': {'examples': '54.3K', 'tokens': '80M', 'config': 'small_lite'},
    'ashraq/financial-news-articles': {'examples': '300K', 'tokens': '197M', 'config': None},
    'wikitext': {'examples': '103K', 'tokens': '103M', 'config': 'wikitext-103-v1'},
}

SAMPLE_SIZE = 1000  # Sample this many examples for token counting

def format_number(num):
    """Format number as K or M for readability."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def count_tokens_sample(dataset, tokenizer, text_columns, sample_size):
    """Count tokens in a sample and extrapolate to full dataset."""
    total_tokens_sample = 0

    # Get actual sample size (might be smaller than requested)
    actual_sample_size = min(sample_size, len(dataset))

    for i in range(actual_sample_size):
        example = dataset[i]

        # Extract text from all relevant columns
        texts = []
        for col in text_columns:
            if col in example and example[col] is not None:
                texts.append(str(example[col]))

        # Concatenate all text
        full_text = ' '.join(texts)

        # Tokenize and count
        tokens = tokenizer(full_text, truncation=False, add_special_tokens=True)
        total_tokens_sample += len(tokens['input_ids'])

    # Calculate average tokens per example
    avg_tokens_per_example = total_tokens_sample / actual_sample_size

    # Extrapolate to full dataset
    total_examples = len(dataset)
    estimated_total_tokens = int(avg_tokens_per_example * total_examples)

    return estimated_total_tokens, avg_tokens_per_example, actual_sample_size

def get_text_columns(dataset_name, example):
    """Determine which columns contain text to tokenize."""
    # Common text column patterns
    text_cols = []

    # Check common column names
    for col in ['text', 'content', 'question', 'answer', 'sentence', 'headline',
                'input', 'output', 'instruction', 'response', 'context']:
        if col in example:
            text_cols.append(col)

    # Dataset-specific handling (align with scripts/count_dataset_tokens.py)
    if 'financial-qa-10K' in dataset_name:
        # Include context when available
        text_cols = ['context', 'question', 'answer']
    elif 'fingpt-sentiment' in dataset_name:
        # Include instruction + input/output
        text_cols = ['instruction', 'input', 'output']
    elif 'finance-alpaca' in dataset_name:
        # Prefer full text if present, else instruction/input/output
        cols = []
        if 'text' in example:
            cols.append('text')
        cols += ['instruction', 'input', 'output']
        text_cols = cols
    elif 'fiqa' in dataset_name:
        text_cols = ['question', 'answer']
    elif 'financial-news-sentiment' in dataset_name:
        text_cols = ['text']
    elif 'financial-reports-sec' in dataset_name:
        # SEC dataset exposes 'sentence' (and sometimes 'section')
        cols = ['sentence']
        if 'section' in example:
            cols.insert(0, 'section')
        text_cols = cols
    elif 'financial-news-articles' in dataset_name:
        # Prefer title + text
        cols = []
        if 'title' in example:
            cols.append('title')
        cols.append('text')
        text_cols = cols
    elif 'wikitext' in dataset_name:
        text_cols = ['text']

    return text_cols

def verify_dataset(dataset_name, config=None):
    """Verify statistics for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Verifying: {dataset_name}" + (f":{config}" if config else ""))
    print(f"{'='*80}")

    try:
        # Load dataset (streaming for efficiency, then take sample)
        print("Loading dataset sample...")
        if config:
            ds = load_dataset(dataset_name, config, split='train', trust_remote_code=True, streaming=False)
        else:
            ds = load_dataset(dataset_name, split='train', trust_remote_code=True, streaming=False)

        # Count examples
        num_examples = len(ds)
        print(f"Number of examples: {num_examples:,} ({format_number(num_examples)})")

        # Load Qwen3 tokenizer (aligns with token counter script)
        print("Loading Qwen3 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine text columns
        first_example = ds[0]
        text_columns = get_text_columns(dataset_name, first_example)
        print(f"Text columns: {text_columns}")

        # Sample and count tokens
        sample_size = min(SAMPLE_SIZE, num_examples)
        print(f"Sampling {sample_size} examples for token counting...")
        total_tokens, avg_tokens, actual_sample = count_tokens_sample(ds, tokenizer, text_columns, sample_size)
        print(f"Average tokens per example: {avg_tokens:.1f} (from {actual_sample} samples)")
        print(f"Estimated total tokens: {total_tokens:,} ({format_number(total_tokens)})")

        # Get current values
        current = CURRENT_VALUES.get(dataset_name, {})
        current_examples = current.get('examples', 'N/A')
        current_tokens = current.get('tokens', 'N/A')

        print(f"\nComparison:")
        print(f"  Current table: {current_examples} examples, {current_tokens} tokens")
        print(f"  Actual values: {format_number(num_examples)} examples, {format_number(total_tokens)} tokens")

        return {
            'dataset': dataset_name,
            'config': config,
            'actual_examples': num_examples,
            'actual_tokens': total_tokens,
            'avg_tokens_per_example': avg_tokens,
            'table_examples': current_examples,
            'table_tokens': current_tokens,
            'examples_match': format_number(num_examples) == current_examples,
            'tokens_match': format_number(total_tokens) == current_tokens,
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Verify all datasets and generate report."""
    print("Dataset Statistics Verification")
    print("Using Qwen3 tokenizer for token counts")

    # Dataset specifications
    datasets_to_verify = [
        ('virattt/financial-qa-10K', None),
        ('FinGPT/fingpt-sentiment-train', None),
        ('gbharti/finance-alpaca', None),
        ('LLukas22/fiqa', None),
        ('zeroshot/twitter-financial-news-sentiment', None),
        ('JanosAudran/financial-reports-sec', 'small_lite'),
        ('ashraq/financial-news-articles', None),
        ('wikitext', 'wikitext-103-v1'),
    ]

    results = []

    for dataset_name, config in datasets_to_verify:
        result = verify_dataset(dataset_name, config)
        if result:
            results.append(result)

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"\n{'Dataset':<40} {'Table':<20} {'Actual':<20} {'Match'}")
    print("-"*80)

    for r in results:
        name = r['dataset']
        if r['config']:
            name += f":{r['config']}"

        table_val = f"{r['table_examples']}, {r['table_tokens']}"
        actual_val = f"{format_number(r['actual_examples'])}, {format_number(r['actual_tokens'])}"
        match = "✓" if r['examples_match'] and r['tokens_match'] else "✗"

        print(f"{name:<40} {table_val:<20} {actual_val:<20} {match}")

    # Identify discrepancies
    discrepancies = [r for r in results if not (r['examples_match'] and r['tokens_match'])]

    if discrepancies:
        print("\n" + "="*80)
        print("CORRECTIONS NEEDED")
        print("="*80)
        for r in discrepancies:
            name = r['dataset']
            if r['config']:
                name += f":{r['config']}"
            print(f"\n{name}:")
            print(f"  Examples: {r['table_examples']} → {format_number(r['actual_examples'])}")
            print(f"  Tokens: {r['table_tokens']} → {format_number(r['actual_tokens'])}")
    else:
        print("\n✓ All values match!")

if __name__ == '__main__':
    main()
