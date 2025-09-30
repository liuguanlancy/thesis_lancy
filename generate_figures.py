#!/usr/bin/env python3
"""
Generate figures from experimental results showing scaling behavior.
Creates plots with model size on x-axis and metrics on y-axis.
Includes support for learning rate adjusted experiments.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model sizes and their display names
MODEL_SIZES = ['0.6B', '1.7B', '4B']
MODEL_SIZE_VALUES = [0.6, 1.7, 4.0]  # For x-axis positioning

# Datasets with LR adjustments
LR_ADJUSTED_DATASETS = ['financial_qa', 'twitter', 'wikitext']

def parse_results_file(filepath):
    """Parse a results markdown file and extract metrics."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract dataset name from filename
    dataset_name = filepath.stem.replace('results_', '').replace('_', ' ').title()
    dataset_key = filepath.stem.replace('results_', '')

    # Parse perplexity table
    perplexity_section = re.search(r'## Perplexity Metrics\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)',
                                   content, re.DOTALL)

    perplexities = {}
    if perplexity_section:
        lines = perplexity_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*')
                try:
                    perplexities[eval_dataset] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ]
                except ValueError:
                    continue

    # Parse loss table
    loss_section = re.search(r'## Loss Metrics\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)',
                            content, re.DOTALL)

    losses = {}
    if loss_section:
        lines = loss_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*')
                try:
                    losses[eval_dataset] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ]
                except ValueError:
                    continue

    # Parse LR comparison tables if this dataset has LR adjustments
    lr_perplexities = {}
    lr_losses = {}
    if dataset_key in LR_ADJUSTED_DATASETS:
        lr_perplexities, lr_losses = parse_lr_comparison(content, dataset_key)

    return dataset_name, perplexities, losses, lr_perplexities, lr_losses

def parse_lr_comparison(content, dataset_key):
    """Parse learning rate comparison tables."""
    lr_perplexities = {}
    lr_losses = {}

    # Parse LR comparison perplexity table
    lr_ppl_section = re.search(
        r'#### Perplexity Metrics Comparison\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)',
        content, re.DOTALL
    )

    if lr_ppl_section:
        lines = lr_ppl_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 6:  # Should have 6 columns for LR comparison
                eval_dataset = parts[0].strip('*')
                try:
                    # Extract values, handling special cases like infinity
                    values = []
                    for val in parts[1:]:
                        # Strip markdown bold markers
                        val = val.strip('*')
                        if '∞' in val or 'inf' in val.lower():
                            values.append(float('inf'))
                        else:
                            values.append(float(val))

                    if len(values) == 5:  # 0.6B, 1.7B orig, 1.7B adj, 4B orig, 4B adj
                        lr_perplexities[eval_dataset] = values
                except (ValueError, IndexError):
                    continue

    # Parse LR comparison loss table
    lr_loss_section = re.search(
        r'#### Loss Metrics Comparison\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)',
        content, re.DOTALL
    )

    if lr_loss_section:
        lines = lr_loss_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 6:
                eval_dataset = parts[0].strip('*')
                try:
                    # Strip markdown bold markers before converting
                    values = [float(val.strip('*')) for val in parts[1:6]]
                    lr_losses[eval_dataset] = values
                except (ValueError, IndexError):
                    continue

    return lr_perplexities, lr_losses

def get_lr_labels(dataset_key):
    """Get learning rate labels for a specific dataset."""
    lr_configs = {
        'financial_qa': {
            '1.7B_orig': '1.7B (LR=2e-5)',
            '1.7B_adj': '1.7B (LR=1e-5)',
            '4B_orig': '4B (LR=2e-5)',
            '4B_adj': '4B (LR=5e-6)'
        },
        'twitter': {
            '1.7B_orig': '1.7B (LR=2e-5)',
            '1.7B_adj': '1.7B (LR=1e-5)',
            '4B_orig': '4B (LR=2e-5)',
            '4B_adj': '4B (LR=5e-6)'
        },
        'wikitext': {
            '1.7B_orig': '1.7B (LR=2e-5)',
            '1.7B_adj': '1.7B (LR=5e-6)',
            '4B_orig': '4B (LR=2e-5)',
            '4B_adj': '4B (LR=3e-6)'
        }
    }
    return lr_configs.get(dataset_key, {})

def create_scaling_plot(dataset_name, perplexities, losses, lr_perplexities, lr_losses, output_dir):
    """Create a figure showing scaling behavior for a dataset."""

    dataset_key = dataset_name.lower().replace(' ', '_')
    has_lr_data = bool(lr_perplexities or lr_losses)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title = f'{dataset_name} Dataset: Model Scaling Performance'
    if has_lr_data:
        title += ' (with LR Adjustments)'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Define colors for different model sizes
    model_colors = {
        '0.6B': '#1f77b4',  # Blue
        '1.7B': '#ff7f0e',  # Orange
        '4B': '#2ca02c'     # Green
    }

    # Plot 1: Perplexity scaling
    if perplexities and not has_lr_data:
        # Original plotting logic for datasets without LR adjustments
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for idx, (eval_name, values) in enumerate(perplexities.items()):
            if eval_name.lower() != 'average':
                ax1.plot(MODEL_SIZE_VALUES, values, marker='o',
                        label=eval_name.replace('_', ' ').title(),
                        color=colors[idx % len(colors)], linewidth=2, markersize=8)

        if 'Average' in perplexities:
            ax1.plot(MODEL_SIZE_VALUES, perplexities['Average'],
                    marker='s', label='Average (Original)', color='black',
                    linewidth=3, markersize=10, linestyle='-')

    elif has_lr_data and 'Average' in lr_perplexities:
        # Plot with LR adjustments - focus on average lines
        avg_values = lr_perplexities['Average']
        # avg_values format: [0.6B, 1.7B_orig, 1.7B_adj, 4B_orig, 4B_adj]

        # Plot 0.6B (no LR adjustment)
        ax1.plot([MODEL_SIZE_VALUES[0]], [avg_values[0]],
                marker='o', color=model_colors['0.6B'],
                linewidth=2.5, markersize=10, label='0.6B (LR=2e-5)')

        # Plot 1.7B original
        ax1.plot(MODEL_SIZE_VALUES[:2], [avg_values[0], avg_values[1]],
                marker='o', color=model_colors['1.7B'], linestyle='-',
                linewidth=2.5, markersize=10, label='1.7B (LR=2e-5)')

        # Plot 1.7B adjusted
        ax1.plot(MODEL_SIZE_VALUES[:2], [avg_values[0], avg_values[2]],
                marker='s', color=model_colors['1.7B'], linestyle='--',
                linewidth=2.5, markersize=10,
                label=get_lr_labels(dataset_key).get('1.7B_adj', '1.7B (adj)'))

        # Plot 4B original
        ax1.plot(MODEL_SIZE_VALUES, [avg_values[0], avg_values[1], avg_values[3]],
                marker='o', color=model_colors['4B'], linestyle='-',
                linewidth=2.5, markersize=10, label='4B (LR=2e-5)')

        # Plot 4B adjusted
        ax1.plot(MODEL_SIZE_VALUES, [avg_values[0], avg_values[2], avg_values[4]],
                marker='s', color=model_colors['4B'], linestyle='--',
                linewidth=2.5, markersize=10,
                label=get_lr_labels(dataset_key).get('4B_adj', '4B (adj)'))

    ax1.set_xlabel('Model Size (Billions of Parameters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Perplexity', fontsize=11, fontweight='bold')
    ax1.set_title('Perplexity vs Model Size', fontsize=12)
    ax1.set_xticks(MODEL_SIZE_VALUES)
    ax1.set_xticklabels(MODEL_SIZES)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=9, loc='best')
    ax1.set_yscale('log')

    # Plot 2: Loss scaling
    if losses and not has_lr_data:
        # Original plotting logic
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for idx, (eval_name, values) in enumerate(losses.items()):
            if eval_name.lower() != 'average':
                ax2.plot(MODEL_SIZE_VALUES, values, marker='o',
                        label=eval_name.replace('_', ' ').title(),
                        color=colors[idx % len(colors)], linewidth=2, markersize=8)

        if 'Average' in losses:
            ax2.plot(MODEL_SIZE_VALUES, losses['Average'],
                    marker='s', label='Average (Original)', color='black',
                    linewidth=3, markersize=10, linestyle='-')

    elif has_lr_data and 'Average' in lr_losses:
        # Plot with LR adjustments
        avg_values = lr_losses['Average']

        # Plot 0.6B
        ax2.plot([MODEL_SIZE_VALUES[0]], [avg_values[0]],
                marker='o', color=model_colors['0.6B'],
                linewidth=2.5, markersize=10, label='0.6B (LR=2e-5)')

        # Plot 1.7B original
        ax2.plot(MODEL_SIZE_VALUES[:2], [avg_values[0], avg_values[1]],
                marker='o', color=model_colors['1.7B'], linestyle='-',
                linewidth=2.5, markersize=10, label='1.7B (LR=2e-5)')

        # Plot 1.7B adjusted
        ax2.plot(MODEL_SIZE_VALUES[:2], [avg_values[0], avg_values[2]],
                marker='s', color=model_colors['1.7B'], linestyle='--',
                linewidth=2.5, markersize=10,
                label=get_lr_labels(dataset_key).get('1.7B_adj', '1.7B (adj)'))

        # Plot 4B original
        ax2.plot(MODEL_SIZE_VALUES, [avg_values[0], avg_values[1], avg_values[3]],
                marker='o', color=model_colors['4B'], linestyle='-',
                linewidth=2.5, markersize=10, label='4B (LR=2e-5)')

        # Plot 4B adjusted
        ax2.plot(MODEL_SIZE_VALUES, [avg_values[0], avg_values[2], avg_values[4]],
                marker='s', color=model_colors['4B'], linestyle='--',
                linewidth=2.5, markersize=10,
                label=get_lr_labels(dataset_key).get('4B_adj', '4B (adj)'))

    ax2.set_xlabel('Model Size (Billions of Parameters)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Loss vs Model Size', fontsize=12)
    ax2.set_xticks(MODEL_SIZE_VALUES)
    ax2.set_xticklabels(MODEL_SIZES)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'scaling_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    if has_lr_data:
        print(f"  ✓ Included LR adjustment data")
    plt.close()

def create_comparison_plot(all_results, output_dir):
    """Create a comparison plot showing average performance across all datasets."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Dataset Comparison: Model Scaling Performance',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Plot average perplexities (use best available results)
    for idx, (dataset_name, perplexities, losses, lr_ppl, lr_loss) in enumerate(all_results):
        # Use LR-adjusted results if available, otherwise use original
        if 'Average' in lr_ppl:
            # Use best results: 0.6B original, 1.7B adjusted, 4B adjusted
            best_ppl = [lr_ppl['Average'][0], lr_ppl['Average'][2], lr_ppl['Average'][4]]
            ax1.plot(MODEL_SIZE_VALUES, best_ppl,
                    marker='o', label=f"{dataset_name} (best)",
                    color=colors[idx], linewidth=2, markersize=8)
        elif 'Average' in perplexities:
            ax1.plot(MODEL_SIZE_VALUES, perplexities['Average'],
                    marker='o', label=dataset_name,
                    color=colors[idx], linewidth=2, markersize=8)

    ax1.set_xlabel('Model Size (Billions of Parameters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Perplexity', fontsize=11, fontweight='bold')
    ax1.set_title('Average Perplexity Across Datasets', fontsize=12)
    ax1.set_xticks(MODEL_SIZE_VALUES)
    ax1.set_xticklabels(MODEL_SIZES)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=8, loc='best')
    ax1.set_yscale('log')

    # Plot average losses (use best available results)
    for idx, (dataset_name, perplexities, losses, lr_ppl, lr_loss) in enumerate(all_results):
        # Use LR-adjusted results if available
        if 'Average' in lr_loss:
            best_loss = [lr_loss['Average'][0], lr_loss['Average'][2], lr_loss['Average'][4]]
            ax2.plot(MODEL_SIZE_VALUES, best_loss,
                    marker='o', label=f"{dataset_name} (best)",
                    color=colors[idx], linewidth=2, markersize=8)
        elif 'Average' in losses:
            ax2.plot(MODEL_SIZE_VALUES, losses['Average'],
                    marker='o', label=dataset_name,
                    color=colors[idx], linewidth=2, markersize=8)

    ax2.set_xlabel('Model Size (Billions of Parameters)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Average Loss Across Datasets', fontsize=12)
    ax2.set_xticks(MODEL_SIZE_VALUES)
    ax2.set_xticklabels(MODEL_SIZES)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=8, loc='best')

    plt.tight_layout()

    output_path = output_dir / 'scaling_comparison_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    # Setup paths
    results_dir = Path('/Users/mengzhao/thesis_lancy/experimental_results')
    output_dir = Path('/Users/mengzhao/thesis_lancy/thesis/figures')
    output_dir.mkdir(exist_ok=True)

    # Process all result files
    result_files = sorted(results_dir.glob('results_*.md'))
    all_results = []

    print(f"Processing {len(result_files)} result files...")
    print()

    for filepath in result_files:
        print(f"Processing: {filepath.name}")
        dataset_name, perplexities, losses, lr_ppl, lr_loss = parse_results_file(filepath)

        if perplexities or losses or lr_ppl or lr_loss:
            create_scaling_plot(dataset_name, perplexities, losses, lr_ppl, lr_loss, output_dir)
            all_results.append((dataset_name, perplexities, losses, lr_ppl, lr_loss))
        else:
            print(f"  Warning: No data found in {filepath.name}")
        print()

    # Create comparison plot
    if all_results:
        print("Creating comparison plot across all datasets...")
        create_comparison_plot(all_results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
    print(f"Total figures created: {len(all_results) + 1}")
    print(f"Datasets with LR adjustments: {', '.join(LR_ADJUSTED_DATASETS)}")

if __name__ == '__main__':
    main()