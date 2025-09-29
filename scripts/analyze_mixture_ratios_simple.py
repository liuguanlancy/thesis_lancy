#!/usr/bin/env python3
"""
Simple analysis of dataset mixture ratios without visualization dependencies.
This script provides detailed calculations for thesis documentation.
"""

import math
from typing import Dict, List

# Dataset information (tokens in millions)
DATASETS_7 = {
    'Financial Q&A': 0.70,
    'FinGPT': 4.14,
    'Finance Alpaca': 8.46,
    'FiQA': 3.60,
    'Twitter': 0.28,
    'SEC Reports': 8.12,
    'News Articles': 197.38
}

DATASETS_8 = {
    'Financial Q&A': 0.70,
    'FinGPT': 4.14,
    'Finance Alpaca': 8.46,
    'FiQA': 3.60,
    'Twitter': 0.28,
    'SEC Reports': 8.12,
    'News Articles': 197.38,
    'WikiText': 103.00
}


def calculate_sqrt_weights(datasets: Dict[str, float]) -> Dict[str, float]:
    """Calculate square root scaled weights."""
    sqrt_values = {k: math.sqrt(v) for k, v in datasets.items()}
    total_sqrt = sum(sqrt_values.values())
    return {k: v / total_sqrt for k, v in sqrt_values.items()}


def apply_50_cap(weights: Dict[str, float], cap: float = 0.5) -> Dict[str, float]:
    """Apply capping rule and redistribute excess weight."""
    weights = weights.copy()

    # Find the maximum weight
    max_dataset = max(weights, key=weights.get)
    max_weight = weights[max_dataset]

    if max_weight <= cap:
        return weights

    # Cap the maximum weight
    excess = max_weight - cap
    weights[max_dataset] = cap

    # Redistribute excess proportionally to other datasets
    remaining_datasets = {k: v for k, v in weights.items() if k != max_dataset}
    total_remaining = sum(remaining_datasets.values())

    for k in remaining_datasets:
        weights[k] += excess * (remaining_datasets[k] / total_remaining)

    return weights


def calculate_proportional_weights(datasets: Dict[str, float]) -> Dict[str, float]:
    """Calculate proportional weights based on token counts."""
    total = sum(datasets.values())
    return {k: v / total for k, v in datasets.items()}


def calculate_uniform_weights(datasets: Dict[str, float]) -> Dict[str, float]:
    """Calculate uniform weights."""
    n = len(datasets)
    return {k: 1.0 / n for k in datasets}


def calculate_entropy(weights: Dict[str, float]) -> float:
    """Calculate entropy of the distribution."""
    entropy = 0
    for w in weights.values():
        if w > 0:
            entropy -= w * math.log2(w)
    return entropy


def calculate_balance_score(weights: Dict[str, float]) -> float:
    """Calculate balance score (0 = imbalanced, 1 = uniform)."""
    n = len(weights)
    uniform = 1.0 / n
    deviations = sum(abs(w - uniform) for w in weights.values())
    max_deviation = 2 * (1 - uniform)
    return 1 - (deviations / max_deviation) if max_deviation > 0 else 1.0


def print_separator(char: str = "=", length: int = 80):
    """Print a separator line."""
    print(char * length)


def print_weights_table(weights: Dict[str, float], title: str):
    """Print weights in a formatted table."""
    print(f"\n{title}")
    print("-" * 50)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for dataset, weight in sorted_weights:
        percentage = weight * 100
        bar = "█" * int(percentage / 2)  # Visual bar (max 50 chars)
        print(f"{dataset:20s}: {weight:.4f} ({percentage:5.1f}%) {bar}")
    print(f"{'Sum':20s}: {sum(weights.values()):.6f}")


def analyze_dataset_configuration(name: str, datasets: Dict[str, float]):
    """Perform complete analysis for a dataset configuration."""
    print_separator()
    print(f"ANALYSIS: {name}")
    print_separator()

    # Basic statistics
    total_tokens = sum(datasets.values())
    print(f"\nDataset Statistics:")
    print(f"  Number of datasets: {len(datasets)}")
    print(f"  Total tokens: {total_tokens:.2f}M")
    print(f"  Average tokens per dataset: {total_tokens/len(datasets):.2f}M")
    print(f"  Largest dataset: {max(datasets, key=datasets.get)} ({max(datasets.values()):.2f}M)")
    print(f"  Smallest dataset: {min(datasets, key=datasets.get)} ({min(datasets.values()):.2f}M)")

    # Token distribution
    print(f"\nToken Distribution:")
    for dataset, tokens in sorted(datasets.items(), key=lambda x: x[1], reverse=True):
        percentage = (tokens / total_tokens) * 100
        print(f"  {dataset:20s}: {tokens:7.2f}M ({percentage:5.1f}%)")

    # Calculate different weighting strategies
    strategies = {
        '50% Cap + Square Root': apply_50_cap(calculate_sqrt_weights(datasets)),
        'Pure Square Root': calculate_sqrt_weights(datasets),
        'Proportional to Size': calculate_proportional_weights(datasets),
        'Uniform (Equal)': calculate_uniform_weights(datasets),
    }

    # Analyze each strategy
    for strategy_name, weights in strategies.items():
        print_weights_table(weights, f"{strategy_name} Scaling")

        # Calculate metrics
        entropy = calculate_entropy(weights)
        max_entropy = math.log2(len(datasets))
        balance = calculate_balance_score(weights)

        print(f"\nMetrics:")
        print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
        print(f"  Balance Score: {balance:.3f} (0=imbalanced, 1=uniform)")
        print(f"  Max weight: {max(weights.values()):.3f}")
        print(f"  Min weight: {min(weights.values()):.3f}")
        print(f"  Ratio (max/min): {max(weights.values())/min(weights.values()):.1f}x")


def show_redistribution_process():
    """Show step-by-step redistribution process for 50% cap."""
    print_separator()
    print("50% CAP REDISTRIBUTION PROCESS (7 DATASETS)")
    print_separator()

    # Step 1: Calculate initial square root weights
    sqrt_weights = calculate_sqrt_weights(DATASETS_7)
    print("\nStep 1: Square Root Weights")
    for dataset, weight in sorted(sqrt_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dataset:20s}: {weight:.4f} ({weight*100:.1f}%)")

    # Step 2: Identify dataset exceeding 50%
    max_dataset = max(sqrt_weights, key=sqrt_weights.get)
    max_weight = sqrt_weights[max_dataset]
    print(f"\nStep 2: Check for Exceeding 50%")
    print(f"  {max_dataset} has {max_weight*100:.1f}% (exceeds 50% cap)")

    # Step 3: Calculate excess
    excess = max_weight - 0.5
    print(f"\nStep 3: Calculate Excess")
    print(f"  Excess to redistribute: {excess:.4f} ({excess*100:.1f}%)")

    # Step 4: Redistribute
    capped_weights = sqrt_weights.copy()
    capped_weights[max_dataset] = 0.5

    remaining = {k: v for k, v in sqrt_weights.items() if k != max_dataset}
    total_remaining = sum(remaining.values())

    print(f"\nStep 4: Redistribution")
    print(f"  Remaining datasets total: {total_remaining:.4f}")
    print(f"  Redistribution per unit: {excess/total_remaining:.4f}")

    for dataset in remaining:
        redistribution = excess * (remaining[dataset] / total_remaining)
        capped_weights[dataset] = sqrt_weights[dataset] + redistribution
        print(f"  {dataset:20s}: {sqrt_weights[dataset]:.4f} + {redistribution:.4f} = {capped_weights[dataset]:.4f}")

    print(f"\nStep 5: Final Weights")
    for dataset, weight in sorted(capped_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dataset:20s}: {weight:.4f} ({weight*100:.1f}%)")
    print(f"  Sum check: {sum(capped_weights.values()):.6f}")


def generate_implementation_code():
    """Generate implementation code for both configurations."""
    print_separator()
    print("IMPLEMENTATION CODE")
    print_separator()

    # 7 datasets
    weights_7 = apply_50_cap(calculate_sqrt_weights(DATASETS_7))
    rates_7 = " ".join([f"{w:.3f}" for w in weights_7.values()])

    print("\n# Bash implementation for 7 financial datasets")
    print('MIXTURE_RATES_7="' + rates_7 + '"')
    print('DATASETS_7=("Financial_QA" "FinGPT" "Finance_Alpaca" "FiQA" "Twitter" "SEC_Reports" "News_Articles")')

    # 8 datasets
    weights_8 = apply_50_cap(calculate_sqrt_weights(DATASETS_8))
    rates_8 = " ".join([f"{w:.3f}" for w in weights_8.values()])

    print("\n# Bash implementation for 8 datasets (with WikiText)")
    print('MIXTURE_RATES_8="' + rates_8 + '"')
    print('DATASETS_8=("Financial_QA" "FinGPT" "Finance_Alpaca" "FiQA" "Twitter" "SEC_Reports" "News_Articles" "WikiText")')

    print("\n# Python implementation")
    print("mixture_rates_7 = {")
    for dataset, weight in weights_7.items():
        print(f'    "{dataset}": {weight:.4f},')
    print("}")

    print("\nmixture_rates_8 = {")
    for dataset, weight in weights_8.items():
        print(f'    "{dataset}": {weight:.4f},')
    print("}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("DATASET MIXTURE RATIO ANALYSIS FOR THESIS")
    print("=" * 80)

    # Analyze both configurations
    analyze_dataset_configuration("7 Financial Datasets", DATASETS_7)
    analyze_dataset_configuration("8 Datasets (Financial + WikiText)", DATASETS_8)

    # Show redistribution process
    show_redistribution_process()

    # Generate implementation code
    generate_implementation_code()

    # Summary comparison
    print_separator()
    print("SUMMARY COMPARISON")
    print_separator()

    weights_7_capped = apply_50_cap(calculate_sqrt_weights(DATASETS_7))
    weights_8_capped = apply_50_cap(calculate_sqrt_weights(DATASETS_8))

    print("\nKey Changes from 7 to 8 datasets:")
    print(f"  News Articles: {weights_7_capped['News Articles']*100:.1f}% → {weights_8_capped['News Articles']*100:.1f}%")
    print(f"  WikiText added: {weights_8_capped['WikiText']*100:.1f}%")
    print(f"  Entropy change: {calculate_entropy(weights_7_capped):.3f} → {calculate_entropy(weights_8_capped):.3f}")
    print(f"  Balance change: {calculate_balance_score(weights_7_capped):.3f} → {calculate_balance_score(weights_8_capped):.3f}")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()