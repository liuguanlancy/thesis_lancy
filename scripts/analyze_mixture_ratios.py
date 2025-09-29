#!/usr/bin/env python3
"""
Analyze and visualize dataset mixture ratios for thesis documentation.
This script provides detailed calculations and visualizations of the mixing strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Dataset information
DATASETS_7 = {
    'Financial Q&A': {'tokens': 0.70, 'examples': 7000, 'color': '#FF6B6B'},
    'FinGPT': {'tokens': 4.14, 'examples': 76000, 'color': '#4ECDC4'},
    'Finance Alpaca': {'tokens': 8.46, 'examples': 68000, 'color': '#45B7D1'},
    'FiQA': {'tokens': 3.60, 'examples': 15000, 'color': '#96CEB4'},
    'Twitter': {'tokens': 0.28, 'examples': 10000, 'color': '#FECA57'},
    'SEC Reports': {'tokens': 8.12, 'examples': 200000, 'color': '#48C9B0'},
    'News Articles': {'tokens': 197.38, 'examples': 306000, 'color': '#BB8FCE'}
}

DATASETS_8 = DATASETS_7.copy()
DATASETS_8['WikiText'] = {'tokens': 103.00, 'examples': 1801350, 'color': '#85929E'}


def calculate_sqrt_weights(datasets: Dict) -> Dict[str, float]:
    """Calculate square root scaled weights."""
    tokens = {k: v['tokens'] for k, v in datasets.items()}
    sqrt_values = {k: np.sqrt(v) for k, v in tokens.items()}
    total_sqrt = sum(sqrt_values.values())
    return {k: v / total_sqrt for k, v in sqrt_values.items()}


def apply_50_cap(weights: Dict[str, float], cap: float = 0.5) -> Dict[str, float]:
    """Apply capping rule and redistribute excess weight."""
    weights = weights.copy()

    # Find datasets exceeding cap
    capped_datasets = {k: v for k, v in weights.items() if v > cap}

    if not capped_datasets:
        return weights

    # Cap the largest dataset
    for dataset in capped_datasets:
        excess = weights[dataset] - cap
        weights[dataset] = cap

        # Redistribute excess proportionally to uncapped datasets
        uncapped = {k: v for k, v in weights.items() if k != dataset}
        total_uncapped = sum(uncapped.values())

        for k in uncapped:
            weights[k] += excess * (uncapped[k] / total_uncapped)

    return weights


def calculate_proportional_weights(datasets: Dict) -> Dict[str, float]:
    """Calculate proportional weights based on token counts."""
    tokens = {k: v['tokens'] for k, v in datasets.items()}
    total = sum(tokens.values())
    return {k: v / total for k, v in tokens.items()}


def calculate_uniform_weights(datasets: Dict) -> Dict[str, float]:
    """Calculate uniform weights."""
    n = len(datasets)
    return {k: 1.0 / n for k in datasets}


def calculate_temperature_weights(datasets: Dict, temperature: float = 2.0) -> Dict[str, float]:
    """Calculate temperature-based weights (T5-style)."""
    tokens = {k: v['tokens'] for k, v in datasets.items()}
    temp_values = {k: v ** (1 / temperature) for k, v in tokens.items()}
    total = sum(temp_values.values())
    return {k: v / total for k, v in temp_values.items()}


def calculate_all_strategies(datasets: Dict) -> pd.DataFrame:
    """Calculate weights for all mixing strategies."""
    strategies = {
        '50% Cap + √': apply_50_cap(calculate_sqrt_weights(datasets)),
        'Pure √': calculate_sqrt_weights(datasets),
        'Proportional': calculate_proportional_weights(datasets),
        'Uniform': calculate_uniform_weights(datasets),
        'Temperature (τ=2)': calculate_temperature_weights(datasets, 2.0),
        'Temperature (τ=3)': calculate_temperature_weights(datasets, 3.0),
    }

    df = pd.DataFrame(strategies).T
    df = df * 100  # Convert to percentages
    return df


def calculate_entropy(weights: Dict[str, float]) -> float:
    """Calculate entropy of the distribution."""
    w = np.array(list(weights.values()))
    w = w[w > 0]  # Remove zeros to avoid log(0)
    return -np.sum(w * np.log2(w))


def calculate_balance_score(weights: Dict[str, float]) -> float:
    """Calculate balance score (0 = imbalanced, 1 = uniform)."""
    n = len(weights)
    uniform = 1.0 / n
    deviations = sum(abs(w - uniform) for w in weights.values())
    max_deviation = 2 * (1 - uniform)
    return 1 - (deviations / max_deviation)


def print_detailed_analysis():
    """Print detailed analysis of mixture ratios."""
    print("=" * 80)
    print("DETAILED MIXTURE RATIO ANALYSIS")
    print("=" * 80)

    for name, datasets in [("7 Financial Datasets", DATASETS_7),
                           ("8 Datasets (with WikiText)", DATASETS_8)]:
        print(f"\n{name}")
        print("-" * 40)

        # Dataset statistics
        total_tokens = sum(d['tokens'] for d in datasets.values())
        total_examples = sum(d['examples'] for d in datasets.values())

        print(f"Total tokens: {total_tokens:.2f}M")
        print(f"Total examples: {total_examples:,}")

        # Calculate strategies
        strategies = {
            '50% Cap + √': apply_50_cap(calculate_sqrt_weights(datasets)),
            'Pure √': calculate_sqrt_weights(datasets),
            'Proportional': calculate_proportional_weights(datasets),
            'Uniform': calculate_uniform_weights(datasets),
        }

        print("\nMixing Strategies Comparison:")
        print("-" * 40)

        for strategy_name, weights in strategies.items():
            entropy = calculate_entropy(weights)
            balance = calculate_balance_score(weights)
            max_weight = max(weights.values())
            min_weight = min(weights.values())

            print(f"\n{strategy_name}:")
            print(f"  Entropy: {entropy:.3f} (max: {np.log2(len(datasets)):.3f})")
            print(f"  Balance: {balance:.3f}")
            print(f"  Max weight: {max_weight:.3f} ({list(weights.keys())[list(weights.values()).index(max_weight)]})")
            print(f"  Min weight: {min_weight:.3f} ({list(weights.keys())[list(weights.values()).index(min_weight)]})")

            # Show top 3 datasets
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3 datasets:")
            for dataset, weight in sorted_weights[:3]:
                print(f"    - {dataset}: {weight*100:.1f}%")


def visualize_mixture_ratios():
    """Create visualizations of mixture ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: 7 datasets comparison
    ax = axes[0, 0]
    df_7 = calculate_all_strategies(DATASETS_7)
    df_7[['50% Cap + √', 'Pure √', 'Proportional', 'Uniform']].plot(kind='bar', ax=ax)
    ax.set_title('Mixture Ratios - 7 Financial Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Weight (%)')
    ax.legend(title='Strategy', loc='upper right')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% cap')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: 8 datasets comparison
    ax = axes[0, 1]
    df_8 = calculate_all_strategies(DATASETS_8)
    df_8[['50% Cap + √', 'Pure √', 'Proportional', 'Uniform']].plot(kind='bar', ax=ax)
    ax.set_title('Mixture Ratios - 8 Datasets (with WikiText)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Weight (%)')
    ax.legend(title='Strategy', loc='upper right')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% cap')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Pie chart for 50% cap strategy (7 datasets)
    ax = axes[1, 0]
    weights_7 = apply_50_cap(calculate_sqrt_weights(DATASETS_7))
    colors_7 = [DATASETS_7[k]['color'] for k in weights_7.keys()]
    wedges, texts, autotexts = ax.pie(weights_7.values(),
                                       labels=weights_7.keys(),
                                       colors=colors_7,
                                       autopct='%1.1f%%',
                                       startangle=90)
    ax.set_title('50% Cap Strategy - 7 Financial Datasets', fontsize=14, fontweight='bold')

    # Plot 4: Pie chart for 50% cap strategy (8 datasets)
    ax = axes[1, 1]
    weights_8 = apply_50_cap(calculate_sqrt_weights(DATASETS_8))
    colors_8 = [DATASETS_8[k]['color'] for k in weights_8.keys()]
    wedges, texts, autotexts = ax.pie(weights_8.values(),
                                       labels=weights_8.keys(),
                                       colors=colors_8,
                                       autopct='%1.1f%%',
                                       startangle=90)
    ax.set_title('50% Cap Strategy - 8 Datasets (with WikiText)', fontsize=14, fontweight='bold')

    plt.suptitle('Dataset Mixture Ratio Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = '/Users/mengzhao/PycharmProjects/lancy_thesis/docs/mixture_ratios_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Show plot
    plt.show()


def generate_latex_table():
    """Generate LaTeX table for thesis."""
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR THESIS")
    print("=" * 80)

    # Create table for 7 datasets
    df_7 = calculate_all_strategies(DATASETS_7)

    print("\n% Table: Mixture Ratios for 7 Financial Datasets")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Mixture Ratios for Different Strategies (7 Financial Datasets)}")
    print("\\begin{tabular}{l" + "r" * len(df_7.columns) + "}")
    print("\\hline")

    # Header
    header = "Strategy & " + " & ".join(df_7.columns) + " \\\\"
    print(header)
    print("\\hline")

    # Data rows
    for strategy in df_7.index:
        row_data = [f"{df_7.loc[strategy, col]:.1f}\\%" for col in df_7.columns]
        row = f"{strategy} & " + " & ".join(row_data) + " \\\\"
        print(row)

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # Create table for 8 datasets
    df_8 = calculate_all_strategies(DATASETS_8)

    print("\n% Table: Mixture Ratios for 8 Datasets (with WikiText)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Mixture Ratios for Different Strategies (8 Datasets with WikiText)}")
    print("\\begin{tabular}{l" + "r" * min(5, len(df_8.columns)) + "...}")
    print("\\hline")

    # Header (abbreviated for space)
    header_cols = list(df_8.columns)[:5]
    header = "Strategy & " + " & ".join(header_cols[:5]) + " & ... \\\\"
    print(header)
    print("\\hline")

    # Data rows
    for strategy in ['50% Cap + √', 'Proportional']:
        row_data = [f"{df_8.loc[strategy, col]:.1f}\\%" for col in header_cols]
        row = f"{strategy} & " + " & ".join(row_data) + " & ... \\\\"
        print(row)

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Main execution function."""
    print("Starting mixture ratio analysis...")

    # Print detailed analysis
    print_detailed_analysis()

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_mixture_ratios()

    # Generate LaTeX tables
    generate_latex_table()

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate final ratios for implementation
    weights_7 = apply_50_cap(calculate_sqrt_weights(DATASETS_7))
    weights_8 = apply_50_cap(calculate_sqrt_weights(DATASETS_8))

    print("\nFinal Implementation Values:")
    print("\n7 Financial Datasets:")
    rates_7 = [f"{weights_7[k]:.3f}" for k in DATASETS_7.keys()]
    print(f'MIXTURE_RATES="{" ".join(rates_7)}"')

    print("\n8 Datasets (with WikiText):")
    rates_8 = [f"{weights_8[k]:.3f}" for k in DATASETS_8.keys()]
    print(f'MIXTURE_RATES="{" ".join(rates_8)}"')

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()