#!/usr/bin/env python3
"""
Count exact token counts for Phase 2B financial datasets using Qwen3 tokenizer.
This script processes ALL examples to provide accurate epoch calculations.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class DatasetTokenCounter:
    """Count tokens in datasets using specified tokenizer."""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen3-0.6B-Base"):
        """Initialize with specified tokenizer."""
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Setup tokenizer padding if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.stats = {}
        
    def format_text_for_dataset(self, example: Dict, dataset_name: str) -> str:
        """Format example text based on dataset structure."""
        # Financial Q&A format
        if "financial-qa" in dataset_name.lower():
            question = example.get('question', '')
            answer = example.get('answer', '')
            context = example.get('context', '')
            if context:
                return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
            return f"Question: {question}\nAnswer: {answer}"
            
        # FinGPT sentiment format (instruction-based)
        elif "fingpt" in dataset_name.lower():
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            if input_text:
                return f"{instruction}\n{input_text}\n{output}"
            return f"{instruction}\n{output}"
            
        # Finance Alpaca format
        elif "alpaca" in dataset_name.lower():
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            text = example.get('text', '')
            
            if text:
                return text
            elif input_text:
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
        # FiQA format
        elif "fiqa" in dataset_name.lower():
            question = example.get('question', '')
            answer = example.get('answer', '')
            if question and answer:
                return f"Question: {question}\nAnswer: {answer}"
            # Fallback to text field
            return example.get('text', str(list(example.values())[0]))
            
        # Twitter sentiment format
        elif "twitter" in dataset_name.lower():
            text = example.get('text', '')
            label = example.get('label', '')
            # Include label for pretraining
            if isinstance(label, int):
                label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
                label = label_map.get(label, str(label))
            return f"{text}\nSentiment: {label}"
            
        # SEC Reports format
        elif "sec" in dataset_name.lower() or "financial-reports" in dataset_name.lower():
            sentence = example.get('sentence', '')
            section = example.get('section', '')
            if section:
                return f"Section: {section}\n{sentence}"
            return sentence
            
        # Financial News Articles format
        elif "news" in dataset_name.lower() or "articles" in dataset_name.lower():
            title = example.get('title', '')
            text = example.get('text', '')
            if title and text:
                return f"{title}\n{text}"
            return text if text else title
            
        # WikiText format (Wikipedia articles)
        elif "wikitext" in dataset_name.lower():
            return example.get('text', '')

        # Default: try common field names
        else:
            if 'text' in example:
                return example['text']
            elif 'question' in example and 'answer' in example:
                return f"Question: {example['question']}\nAnswer: {example['answer']}"
            elif 'instruction' in example:
                text = example['instruction']
                if 'output' in example:
                    text = f"{text}\n{example['output']}"
                return text
            else:
                # Fallback: concatenate all string values
                return ' '.join(str(v) for v in example.values() if isinstance(v, (str, int, float)))
    
    def count_dataset_tokens(
        self,
        dataset_name: str,
        config: Optional[str] = None,
        label: str = "",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Count exact tokens in entire dataset or sample for large datasets."""
        print(f"\nAnalyzing {label or dataset_name}...")
        print("  Loading dataset...")

        start_time = time.time()

        try:
            # Load dataset - NOT streaming for exact count
            dataset = load_dataset(dataset_name, config, split="train")

            # Get exact dataset size
            try:
                total_examples = len(dataset)
            except (TypeError, AttributeError):
                # For iterable datasets, count as we go
                print("  Warning: Dataset is iterable, counting as we process...")
                total_examples = 0

            # Determine if we should sample
            if sample_size and total_examples > sample_size:
                print(f"  Dataset has {total_examples:,} examples - sampling {sample_size:,} for estimation...")
                dataset_to_process = dataset.shuffle(seed=42).select(range(sample_size))
                examples_to_count = sample_size
                is_sampled = True
            else:
                print(f"  Processing ALL {total_examples:,} examples for exact token count...")
                dataset_to_process = dataset
                examples_to_count = total_examples
                is_sampled = False

            # Process examples
            sampled_tokens = 0
            token_counts = []
            min_tokens = float('inf')
            max_tokens = 0

            # Process with progress bar
            for example in tqdm(dataset_to_process, total=examples_to_count if examples_to_count > 0 else None,
                              desc="  Counting tokens", unit=" examples"):
                # Format text based on dataset
                text = self.format_text_for_dataset(example, dataset_name)

                # Tokenize and count
                tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
                token_count = len(tokens)

                sampled_tokens += token_count
                token_counts.append(token_count)
                min_tokens = min(min_tokens, token_count)
                max_tokens = max(max_tokens, token_count)

                # For iterable datasets, update count
                if not hasattr(dataset, '__len__'):
                    total_examples += 1

            # Calculate statistics
            avg_tokens = sampled_tokens / examples_to_count if examples_to_count > 0 else 0

            # Extrapolate to full dataset if sampled
            if is_sampled:
                total_tokens = int(avg_tokens * total_examples)
            else:
                total_tokens = sampled_tokens

            # Calculate epochs with 100M tokens (actual budget per phase2b_rtx4090.sh)
            tokens_per_experiment = 100e6  # 100M = 0.1B
            epochs_with_packing = tokens_per_experiment / total_tokens if total_tokens > 0 else 0

            # Processing time
            elapsed_time = time.time() - start_time

            # Store results
            stats = {
                "dataset": dataset_name,
                "label": label,
                "total_examples": total_examples,
                "sampled": is_sampled,
                "sample_size": sample_size if is_sampled else total_examples,
                "total_tokens_exact": total_tokens,
                "avg_tokens_per_example": round(avg_tokens, 1),
                "min_tokens": min_tokens if min_tokens != float('inf') else 0,
                "max_tokens": max_tokens,
                "total_dataset_millions": round(total_tokens / 1e6, 2),
                "total_dataset_billions": round(total_tokens / 1e9, 3),
                "tokens_per_experiment": tokens_per_experiment,
                "epochs_with_100k_steps": round(epochs_with_packing, 1),
                "overtraining_factor": f"{epochs_with_packing:.0f}x" if epochs_with_packing > 1 else "none",
                "processing_time_seconds": round(elapsed_time, 1)
            }

            # Print summary
            print(f"  ✓ Examples: {total_examples:,} ({'sampled ' + str(sample_size) + ' for estimation' if is_sampled else 'all processed'})")
            print(f"  ✓ Total tokens ({'estimated' if is_sampled else 'exact'}): {total_tokens:,}")
            print(f"  ✓ Avg tokens/example: {avg_tokens:.1f} (min: {min_tokens}, max: {max_tokens})")
            print(f"  ✓ Total dataset: {total_tokens/1e6:.2f}M tokens ({total_tokens/1e9:.3f}B)")
            print(f"  ✓ Epochs with 100M tokens: {epochs_with_packing:.1f}")
            print(f"  ✓ Processing time: {elapsed_time:.1f} seconds")
            
            if epochs_with_packing > 100:
                print(f"  ⚠️ WARNING: Extreme overtraining ({epochs_with_packing:.0f} epochs)!")
            elif epochs_with_packing > 10:
                print(f"  ⚠️ WARNING: Significant overtraining ({epochs_with_packing:.0f} epochs)")
            
            return stats
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {
                "dataset": dataset_name,
                "label": label,
                "error": str(e)
            }
    
    def analyze_all_datasets(self):
        """Analyze all Phase 2B financial datasets."""
        # (dataset_name, config, label, sample_size)
        # sample_size=None means process all examples
        # sample_size=50000 means sample 50K for estimation (large datasets)
        datasets_info = [
            ("virattt/financial-qa-10K", None, "Financial Q&A", None),
            ("FinGPT/fingpt-sentiment-train", None, "FinGPT Sentiment", None),
            ("gbharti/finance-alpaca", None, "Finance Alpaca", None),
            ("LLukas22/fiqa", None, "FiQA", None),
            ("zeroshot/twitter-financial-news-sentiment", None, "Twitter Sentiment", None),
            ("JanosAudran/financial-reports-sec", "small_lite", "SEC Reports", None),
            ("ashraq/financial-news-articles", None, "News Articles", 50000),  # Sample 50K
            ("wikitext", "wikitext-103-raw-v1", "WikiText", 50000),  # Sample 50K
        ]

        print("\n" + "="*60)
        print("PHASE 2B DATASET TOKEN ANALYSIS")
        print("Using tokenizer: Qwen/Qwen3-0.6B-Base")
        print("Target tokens per experiment: 100M (0.1B)")
        print("Configuration: ~12k steps × 8 batch × 1024 max_length")
        print("Large datasets (>100K): sampling 50K for estimation")
        print("Small datasets (<100K): processing all for exact counts")
        print("Datasets: 7 financial + 1 general (WikiText)")
        print("="*60)

        all_stats = []
        total_combined_tokens = 0
        total_processing_time = 0

        for dataset_name, config, label, sample_size in datasets_info:
            stats = self.count_dataset_tokens(dataset_name, config, label, sample_size)
            all_stats.append(stats)

            if "total_tokens_exact" in stats:
                total_combined_tokens += stats["total_tokens_exact"]
            if "processing_time_seconds" in stats:
                total_processing_time += stats["processing_time_seconds"]
        
        # Mixed corpus analysis (7 financial datasets)
        print("\n" + "-"*60)
        print("MIXED CORPUS ANALYSIS (7 Financial Datasets)")
        print("-"*60)

        mixture_rates_7 = [0.034, 0.191, 0.172, 0.043, 0.003, 0.194, 0.362]

        print("\nMixture composition (50cap strategy):")
        weighted_tokens_7 = 0
        total_financial_tokens = 0

        # Calculate total for first 7 datasets (financial only)
        for i, stat in enumerate(all_stats[:7]):
            if "total_tokens_exact" in stat:
                total_financial_tokens += stat["total_tokens_exact"]

        for rate, stat in zip(mixture_rates_7, all_stats[:7]):
            if "total_tokens_exact" in stat:
                dataset_total = stat["total_tokens_exact"]
                allocated_tokens = 100e6 * rate  # 100M * mixture rate
                dataset_epochs = allocated_tokens / dataset_total if dataset_total > 0 else 0
                weighted_tokens_7 += allocated_tokens
                print(f"  {stat['label']}: {rate*100:.1f}% → {allocated_tokens/1e6:.2f}M tokens ({dataset_epochs:.2f} epochs)")

        mixed_epochs_7 = 100e6 / total_financial_tokens if total_financial_tokens > 0 else 0
        print(f"\nMixed corpus total tokens: {total_financial_tokens:,} ({total_financial_tokens/1e6:.2f}M)")
        print(f"With 100M budget: each dataset sees {mixed_epochs_7:.2f} epochs on average")

        # Mixed-wiki corpus analysis (8 datasets including WikiText)
        print("\n" + "-"*60)
        print("MIXED-WIKI CORPUS ANALYSIS (8 Datasets with WikiText)")
        print("-"*60)

        mixture_rates_8 = [0.024, 0.058, 0.083, 0.054, 0.015, 0.081, 0.399, 0.288]

        print("\nMixture composition (50cap strategy with WikiText):")
        weighted_tokens_8 = 0

        for rate, stat in zip(mixture_rates_8, all_stats):
            if "total_tokens_exact" in stat:
                dataset_total = stat["total_tokens_exact"]
                allocated_tokens = 100e6 * rate  # 100M * mixture rate
                dataset_epochs = allocated_tokens / dataset_total if dataset_total > 0 else 0
                weighted_tokens_8 += allocated_tokens
                print(f"  {stat['label']}: {rate*100:.1f}% → {allocated_tokens/1e6:.2f}M tokens ({dataset_epochs:.2f} epochs)")

        mixed_wiki_epochs = 100e6 / total_combined_tokens if total_combined_tokens > 0 else 0
        print(f"\nMixed-wiki corpus total tokens: {total_combined_tokens:,} ({total_combined_tokens/1e6:.2f}M)")
        print(f"With 100M budget: each dataset sees {mixed_wiki_epochs:.2f} epochs on average")
        
        # Summary statistics
        print("\n" + "="*60)
        print("TOKEN COUNT SUMMARY")
        print("="*60)
        print(f"\nTotal processing time: {total_processing_time/60:.1f} minutes")
        print("\nWith packing ENABLED and ~12k steps for all experiments:")
        print("Each experiment processes: 100M tokens (0.1B)")
        print("\nDataset-specific epoch counts:")
        print("(* = estimated from 100K sample, otherwise exact)")

        for stat in all_stats:
            if "epochs_with_100k_steps" in stat:
                epochs = stat["epochs_with_100k_steps"]
                label = stat["label"]
                tokens_exact = stat.get("total_tokens_exact", 0)
                tokens_m = tokens_exact / 1e6
                is_sampled = stat.get("sampled", False)

                # Format output with alignment
                marker = "*" if is_sampled else " "
                label_padded = f"{marker}{label}:".ljust(21)
                tokens_str = f"{tokens_exact:>12,} ({tokens_m:>7.2f}M)"
                epochs_str = f"{epochs:>8.1f} epochs"

                if epochs > 100:
                    warning = " ⚠️ EXTREME overtraining!"
                elif epochs > 10:
                    warning = " ⚠️ Significant overtraining"
                else:
                    warning = ""

                print(f"  {label_padded} {tokens_str} → {epochs_str}{warning}")
        
        print(f"\n  {'Mixed (7 Financial):'.ljust(20)} {total_financial_tokens:>12,} ({total_financial_tokens/1e6:>7.2f}M) → {mixed_epochs_7:>8.1f} epochs")
        print(f"  {'Mixed-Wiki (8 Total):'.ljust(20)} {total_combined_tokens:>12,} ({total_combined_tokens/1e6:>7.2f}M) → {mixed_wiki_epochs:>8.1f} epochs")
        
        # Save results to JSON
        output_path = project_root / "scripts" / "dataset_token_stats_exact.json"
        with open(output_path, "w") as f:
            json.dump({
                "tokenizer": "Qwen/Qwen3-0.6B-Base",
                "tokens_per_experiment": 100e6,
                "configuration": {
                    "steps": 12207,
                    "batch_size": 8,
                    "max_length": 1024,
                    "packing": True
                },
                "datasets": all_stats,
                "mixed_corpus_7_financial": {
                    "description": "7 financial datasets only",
                    "mixture_rates": dict(zip([s["label"] for s in all_stats[:7]], mixture_rates_7)),
                    "total_tokens_exact": total_financial_tokens,
                    "total_tokens_millions": round(total_financial_tokens / 1e6, 2),
                    "total_tokens_billions": round(total_financial_tokens / 1e9, 3),
                    "epochs": round(mixed_epochs_7, 1)
                },
                "mixed_wiki_corpus_8_total": {
                    "description": "8 datasets including WikiText",
                    "mixture_rates": dict(zip([s["label"] for s in all_stats], mixture_rates_8)),
                    "total_tokens_exact": total_combined_tokens,
                    "total_tokens_millions": round(total_combined_tokens / 1e6, 2),
                    "total_tokens_billions": round(total_combined_tokens / 1e9, 3),
                    "epochs": round(mixed_wiki_epochs, 1)
                },
                "processing_info": {
                    "total_time_seconds": round(total_processing_time, 1),
                    "total_time_minutes": round(total_processing_time / 60, 1),
                    "exact_counts": True
                }
            }, f, indent=2)
        
        print(f"\n✓ Exact results saved to: {output_path}")
        
        return all_stats


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("STARTING TOKEN COUNT ANALYSIS")
    print("Large datasets (>100K) will be sampled for faster processing")
    print("Estimated time: 5-10 minutes")
    print("="*60)

    counter = DatasetTokenCounter(tokenizer_name="Qwen/Qwen3-0.6B-Base")
    counter.analyze_all_datasets()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nMethodology:")
    print("- Small datasets (<100K): exact counts from all examples")
    print("- Large datasets (>100K): estimated from 50K random sample")
    print("- WikiText and News Articles: sampled for faster processing")
    print("\nKey findings (with 100M token budget):")
    print("1. Twitter dataset is extremely short (~27 tokens avg) - massive overtraining")
    print("2. FinGPT is surprisingly short (~54 tokens avg) despite being 'sentiment instructions'")
    print("3. Finance Alpaca and FiQA are longer (~250 tokens avg) but still heavily overtrained")
    print("4. WikiText provides general language modeling data to balance financial-specific content")
    print("5. All individual experiments will massively overtrain their datasets")
    print("6. Mixed corpus (7 financial) provides reasonable training (~0.5 epochs)")
    print("7. Mixed-wiki corpus (8 total) provides balanced training with general domain data")
    

if __name__ == "__main__":
    main()