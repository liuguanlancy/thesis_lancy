#!/usr/bin/env python3
"""
Generate cross-dataset comparison tables for thesis.

For each evaluation dataset, shows which training dataset (including LR variants)
performs best across different model sizes. Best values are bolded.
"""

import os
import re
from typing import Dict, List, Tuple

# Configuration
RESULTS_DIR = "experimental_results"
OUTPUT_DIR = "thesis/tables"
MODEL_SIZES = ['0.6B', '1.7B', '4B']

# Datasets with LR adjustments (training set perspective)
LR_ADJUSTED_DATASETS = {
    'financial_qa': {
        '1.7B': '1e-5',
        '4B': '5e-6'
    },
    'twitter': {
        '1.7B': '1e-5',
        '4B': '5e-6'
    },
    'wikitext': {
        '1.7B': '5e-6',
        '4B': '3e-6'
    }
}

# Dataset name mappings for display
DATASET_DISPLAY_NAMES = {
    'alpaca': 'Alpaca',
    'financial_qa': 'Financial QA',
    'fingpt': 'FinGPT',
    'fiqa': 'FiQA',
    'mixed_financial': 'Mixed Financial',
    'mixed_wiki_financial': 'Mixed Wiki+Financial',
    'news_articles': 'Financial News',
    'sec_reports': 'SEC Reports',
    'twitter': 'Twitter Financial',
    'wikitext': 'WikiText',
    'financial_news': 'Financial News',
    'financial_repor': 'SEC Reports'
}


def parse_results_file(file_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Parse a results file to extract loss and perplexity metrics.

    Returns:
        (loss_metrics, ppl_metrics, lr_loss_metrics, lr_ppl_metrics)
        lr_*_metrics contain adjusted LR results if available
    """
    with open(file_path, 'r') as f:
        content = f.read()

    loss_metrics = {}
    ppl_metrics = {}
    lr_loss_metrics = {}
    lr_ppl_metrics = {}

    # Parse standard Loss Metrics table
    loss_section = re.search(
        r'## Loss Metrics\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content, re.DOTALL
    )

    if loss_section:
        lines = loss_section.group(1).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*').lower().replace(' ', '_')
                if eval_dataset == 'average':
                    continue
                values = []
                for val in parts[1:]:
                    val = val.strip('*')
                    try:
                        values.append(float(val))
                    except:
                        values.append(None)
                if len(values) == 3:
                    loss_metrics[eval_dataset] = dict(zip(MODEL_SIZES, values))

    # Parse standard Perplexity Metrics table
    ppl_section = re.search(
        r'## Perplexity Metrics\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content, re.DOTALL
    )

    if ppl_section:
        lines = ppl_section.group(1).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*').lower().replace(' ', '_')
                if eval_dataset == 'average':
                    continue
                values = []
                for val in parts[1:]:
                    val = val.strip('*')
                    if '∞' in val or 'inf' in val.lower():
                        values.append(float('inf'))
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(None)
                if len(values) == 3:
                    ppl_metrics[eval_dataset] = dict(zip(MODEL_SIZES, values))

    # Check for LR comparison sections
    lr_loss_section = re.search(
        r'#### Loss Metrics Comparison\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content, re.DOTALL
    )

    if lr_loss_section:
        lines = lr_loss_section.group(1).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 6:
                eval_dataset = parts[0].strip('*').lower().replace(' ', '_')
                if eval_dataset == 'average':
                    continue
                # Parts: [dataset, 0.6B, 1.7B_orig, 1.7B_adj, 4B_orig, 4B_adj]
                try:
                    lr_loss_metrics[eval_dataset] = {
                        '0.6B': float(parts[1].strip('*')),
                        '1.7B': float(parts[3].strip('*')),  # adjusted value
                        '4B': float(parts[5].strip('*'))     # adjusted value
                    }
                except:
                    pass

    lr_ppl_section = re.search(
        r'#### Perplexity Metrics Comparison\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content, re.DOTALL
    )

    if lr_ppl_section:
        lines = lr_ppl_section.group(1).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 6:
                eval_dataset = parts[0].strip('*').lower().replace(' ', '_')
                if eval_dataset == 'average':
                    continue
                try:
                    values = []
                    for idx in [1, 3, 5]:  # 0.6B, 1.7B_adj, 4B_adj
                        val = parts[idx].strip('*')
                        if '∞' in val or 'inf' in val.lower():
                            values.append(float('inf'))
                        else:
                            values.append(float(val))

                    lr_ppl_metrics[eval_dataset] = {
                        '0.6B': values[0],
                        '1.7B': values[1],
                        '4B': values[2]
                    }
                except:
                    pass

    return loss_metrics, ppl_metrics, lr_loss_metrics, lr_ppl_metrics


def collect_all_training_results() -> Dict:
    """Collect results from all training datasets.

    Returns:
        Dict mapping training_dataset -> {
            'original': {eval_dataset -> {'0.6B': {'loss': X, 'ppl': Y}, ...}},
            'adjusted': {eval_dataset -> {'0.6B': {'loss': X, 'ppl': Y}, ...}}  # if applicable
        }
    """
    all_results = {}

    for filename in os.listdir(RESULTS_DIR):
        if not filename.startswith('results_') or not filename.endswith('.md'):
            continue

        # Extract training dataset name
        training_dataset = filename.replace('results_', '').replace('.md', '')

        file_path = os.path.join(RESULTS_DIR, filename)
        loss_metrics, ppl_metrics, lr_loss_metrics, lr_ppl_metrics = parse_results_file(file_path)

        # Store original results
        all_results[training_dataset] = {'original': {}}

        for eval_dataset in loss_metrics.keys():
            all_results[training_dataset]['original'][eval_dataset] = {}
            for model_size in MODEL_SIZES:
                all_results[training_dataset]['original'][eval_dataset][model_size] = {
                    'loss': loss_metrics[eval_dataset].get(model_size),
                    'ppl': ppl_metrics[eval_dataset].get(model_size)
                }

        # Store adjusted LR results if available
        if lr_loss_metrics and lr_ppl_metrics:
            all_results[training_dataset]['adjusted'] = {}
            for eval_dataset in lr_loss_metrics.keys():
                all_results[training_dataset]['adjusted'][eval_dataset] = {}
                for model_size in MODEL_SIZES:
                    all_results[training_dataset]['adjusted'][eval_dataset][model_size] = {
                        'loss': lr_loss_metrics[eval_dataset].get(model_size),
                        'ppl': lr_ppl_metrics[eval_dataset].get(model_size)
                    }

    return all_results


def format_lr_label(training_dataset: str) -> str:
    """Format LR adjustment label for display."""
    if training_dataset not in LR_ADJUSTED_DATASETS:
        return ""

    lr_info = LR_ADJUSTED_DATASETS[training_dataset]
    parts = []
    if '1.7B' in lr_info:
        parts.append(f"1.7B: {lr_info['1.7B']}")
    if '4B' in lr_info:
        parts.append(f"4B: {lr_info['4B']}")

    return f" ({', '.join(parts)})"


def generate_cross_dataset_table(eval_dataset: str, all_results: Dict) -> str:
    """Generate LaTeX table for a specific evaluation dataset."""

    # Collect all rows with their metrics
    rows = []

    for training_dataset in sorted(all_results.keys()):
        # Add original LR row
        if eval_dataset in all_results[training_dataset]['original']:
            row_data = {
                'label': f"{DATASET_DISPLAY_NAMES.get(training_dataset, training_dataset)} (2e-5)",
                'metrics': {}
            }
            for model_size in MODEL_SIZES:
                metrics = all_results[training_dataset]['original'][eval_dataset].get(model_size, {})
                row_data['metrics'][model_size] = {
                    'loss': metrics.get('loss'),
                    'ppl': metrics.get('ppl')
                }
            rows.append(row_data)

        # Add adjusted LR row if available
        if 'adjusted' in all_results[training_dataset] and eval_dataset in all_results[training_dataset]['adjusted']:
            row_data = {
                'label': f"{DATASET_DISPLAY_NAMES.get(training_dataset, training_dataset)}{format_lr_label(training_dataset)}",
                'metrics': {}
            }
            for model_size in MODEL_SIZES:
                metrics = all_results[training_dataset]['adjusted'][eval_dataset].get(model_size, {})
                row_data['metrics'][model_size] = {
                    'loss': metrics.get('loss'),
                    'ppl': metrics.get('ppl')
                }
            rows.append(row_data)

    # Find minimum values for each column (for bolding)
    min_values = {}
    for model_size in MODEL_SIZES:
        for metric in ['loss', 'ppl']:
            key = f"{model_size}_{metric}"
            values = []
            for row in rows:
                val = row['metrics'].get(model_size, {}).get(metric)
                if val is not None and val != float('inf'):
                    values.append(val)
            min_values[key] = min(values) if values else None

    # Generate LaTeX table
    eval_display_name = DATASET_DISPLAY_NAMES.get(eval_dataset, eval_dataset.replace('_', ' ').title())

    latex = f"""% Cross-Dataset Comparison: {eval_display_name} as Evaluation Dataset
% Shows which training dataset performs best on {eval_display_name}
% Bold values indicate best performance for each model size

\\begin{{table}}[h]
\\centering
\\caption{{{eval_display_name} Evaluation: Performance Across Training Datasets}}
\\label{{tab:cross_{eval_dataset}}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|ccc|ccc}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Training Dataset}}}} &
\\multicolumn{{3}}{{c|}}{{\\textbf{{Cross-Entropy Loss}}}} &
\\multicolumn{{3}}{{c}}{{\\textbf{{Perplexity}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
& \\textbf{{0.6B}} & \\textbf{{1.7B}} & \\textbf{{4B}} & \\textbf{{0.6B}} & \\textbf{{1.7B}} & \\textbf{{4B}} \\\\
\\midrule
"""

    for row in rows:
        label = row['label']
        line_parts = [label]

        # Add loss values
        for model_size in MODEL_SIZES:
            val = row['metrics'].get(model_size, {}).get('loss')
            if val is None:
                line_parts.append('-')
            else:
                key = f"{model_size}_loss"
                if min_values.get(key) == val:
                    line_parts.append(f"\\textbf{{{val:.2f}}}")
                else:
                    line_parts.append(f"{val:.2f}")

        # Add perplexity values
        for model_size in MODEL_SIZES:
            val = row['metrics'].get(model_size, {}).get('ppl')
            if val is None:
                line_parts.append('-')
            elif val == float('inf'):
                line_parts.append('$\\infty$')
            else:
                key = f"{model_size}_ppl"
                if min_values.get(key) == val:
                    line_parts.append(f"\\textbf{{{val:.2f}}}")
                else:
                    line_parts.append(f"{val:.2f}")

        latex += " & ".join(line_parts) + " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
}
\\end{table}

"""

    return latex


def main():
    """Generate all cross-dataset comparison tables."""
    print("Collecting results from all training datasets...")
    all_results = collect_all_training_results()

    # Get list of all evaluation datasets
    eval_datasets = set()
    for training_dataset in all_results.values():
        eval_datasets.update(training_dataset['original'].keys())

    print(f"Found {len(eval_datasets)} evaluation datasets")
    print(f"Found {len(all_results)} training datasets")

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate table for each evaluation dataset
    for eval_dataset in sorted(eval_datasets):
        print(f"Generating table for {eval_dataset}...")
        latex_table = generate_cross_dataset_table(eval_dataset, all_results)

        output_file = os.path.join(OUTPUT_DIR, f"table_cross_{eval_dataset}.tex")
        with open(output_file, 'w') as f:
            f.write(latex_table)

        print(f"  Saved to {output_file}")

    print(f"\nGenerated {len(eval_datasets)} cross-dataset comparison tables")


if __name__ == '__main__':
    main()