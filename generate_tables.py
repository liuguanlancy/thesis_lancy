#!/usr/bin/env python3
"""
Generate LaTeX tables from experimental results.
Creates comprehensive evaluation tables for each dataset.
"""

import re
from pathlib import Path

# Datasets with LR adjustments
LR_ADJUSTED_DATASETS = {
    'financial_qa': {
        '1.7B': ['2e-5', '1e-5'],
        '4B': ['2e-5', '5e-6']
    },
    'twitter': {
        '1.7B': ['2e-5', '1e-5'],
        '4B': ['2e-5', '5e-6']
    },
    'wikitext': {
        '1.7B': ['2e-5', '5e-6'],
        '4B': ['2e-5', '3e-6']
    }
}

# Dataset metadata
DATASET_INFO = {
    'alpaca': {'name': 'Finance Alpaca', 'hf_id': 'gbharti/finance-alpaca', 'size': '17M tokens', 'examples': '68K'},
    'financial_qa': {'name': 'Financial QA 10K', 'hf_id': 'virattt/financial-qa-10K', 'size': '3.5M tokens', 'examples': '7K'},
    'fingpt': {'name': 'FinGPT Sentiment', 'hf_id': 'FinGPT/fingpt-sentiment-train', 'size': '19M tokens', 'examples': '76.8K'},
    'fiqa': {'name': 'FiQA', 'hf_id': 'FiQA dataset', 'size': '4M tokens', 'examples': '~10K'},
    'mixed_financial': {'name': 'Mixed Financial', 'hf_id': '7 datasets mixed', 'size': '322M tokens', 'examples': 'Multiple'},
    'mixed_wiki_financial': {'name': 'Mixed Wiki+Financial', 'hf_id': 'WikiText + 7 financial datasets', 'size': '~400M tokens', 'examples': 'Multiple'},
    'news_articles': {'name': 'Financial News', 'hf_id': 'Financial news articles', 'size': '197M tokens', 'examples': '~300K'},
    'sec_reports': {'name': 'SEC Reports', 'hf_id': 'SEC 10-K/10-Q filings', 'size': '80M tokens', 'examples': '~50K'},
    'twitter': {'name': 'Twitter Financial', 'hf_id': 'Financial tweets', 'size': '0.3M tokens', 'examples': '~1K'},
    'wikitext': {'name': 'WikiText', 'hf_id': 'WikiText-103', 'size': '100M tokens', 'examples': '~28K'}
}

def parse_results_file(filepath):
    """Parse results markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()

    dataset_key = filepath.stem.replace('results_', '')

    # Parse basic metrics
    perplexities = {}
    losses = {}

    # Parse perplexity table
    ppl_section = re.search(r'## Perplexity Metrics\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if ppl_section:
        lines = ppl_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*')
                try:
                    perplexities[eval_dataset] = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    continue

    # Parse loss table
    loss_section = re.search(r'## Loss Metrics\s+\|.*?\n\|(.*?)\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if loss_section:
        lines = loss_section.group(2).strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                eval_dataset = parts[0].strip('*')
                try:
                    losses[eval_dataset] = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    continue

    # Parse LR comparison if available
    lr_perplexities = {}
    lr_losses = {}
    if dataset_key in LR_ADJUSTED_DATASETS:
        # Find loss comparison table
        loss_table_match = re.search(
            r'#### Loss Metrics Comparison\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
            content, re.DOTALL
        )
        if loss_table_match:
            lines = loss_table_match.group(1).strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 6:
                    eval_dataset = parts[0].strip('*')
                    try:
                        values = [float(val.strip('*')) for val in parts[1:6]]
                        lr_losses[eval_dataset] = values
                    except (ValueError, IndexError):
                        continue

        # Find perplexity comparison table
        ppl_table_match = re.search(
            r'#### Perplexity Metrics Comparison\s*\n+\| Eval Dataset.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
            content, re.DOTALL
        )
        if ppl_table_match:
            lines = ppl_table_match.group(1).strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 6:
                    eval_dataset = parts[0].strip('*')
                    try:
                        values = []
                        for val in parts[1:6]:
                            val = val.strip('*')
                            if '∞' in val or 'inf' in val.lower():
                                values.append(float('inf'))
                            else:
                                values.append(float(val))
                        lr_perplexities[eval_dataset] = values
                    except (ValueError, IndexError):
                        continue

    return dataset_key, perplexities, losses, lr_perplexities, lr_losses

def format_value(val, is_inf_ok=False):
    """Format a numeric value for LaTeX, handling infinity."""
    if isinstance(val, float) and (val == float('inf') or val == float('-inf')):
        if is_inf_ok:
            return r'$\infty$'
        else:
            return r'\textit{fail}'
    return f"{val:.2f}" if val < 100 else f"{val:.1f}"

def create_simple_table(dataset_key, perplexities, losses):
    """Create simple table without LR adjustments."""
    info = DATASET_INFO[dataset_key]
    dataset_display = info['name']

    # Build rows for evaluation datasets
    rows = []
    train_dataset = dataset_key

    for eval_name in sorted(perplexities.keys()):
        if eval_name.lower() == 'average':
            continue

        ppl_vals = perplexities[eval_name]
        loss_vals = losses.get(eval_name, [0, 0, 0])

        # Highlight training dataset
        if eval_name.lower() == train_dataset or eval_name.lower().replace('_', ' ') == train_dataset:
            row_start = r'\rowcolor{gray!20} \textbf{' + eval_name.replace('_', ' ').title() + ' (train)}'
        else:
            row_start = eval_name.replace('_', ' ').title()

        row = f"{row_start} & {format_value(loss_vals[0])} & {format_value(loss_vals[1])} & {format_value(loss_vals[2])} & " + \
              f"{format_value(ppl_vals[0])} & {format_value(ppl_vals[1])} & {format_value(ppl_vals[2])} \\\\"
        rows.append(row)

    # Add average row
    if 'Average' in perplexities:
        avg_ppl = perplexities['Average']
        avg_loss = losses.get('Average', [0, 0, 0])
        avg_row = r'\rowcolor{blue!10} \textbf{Average} & ' + \
                  f"\\textbf{{{format_value(avg_loss[0])}}} & \\textbf{{{format_value(avg_loss[1])}}} & \\textbf{{{format_value(avg_loss[2])}}} & " + \
                  f"\\textbf{{{format_value(avg_ppl[0])}}} & \\textbf{{{format_value(avg_ppl[1])}}} & \\textbf{{{format_value(avg_ppl[2])}}} \\\\"
        rows.append(avg_row)

    table_rows = '\n'.join(rows)

    latex = f"""% {dataset_display} Dataset: Evaluation Results
% Training: {info['name']} ({info['hf_id']}, {info['size']})
% All models trained with LR=2e-5

\\begin{{table}}[h]
\\centering
\\caption{{{dataset_display} Dataset: Evaluation Across Multiple Datasets}}
\\label{{tab:{dataset_key}_results}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|ccc|ccc}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Eval Dataset}}}} &
\\multicolumn{{3}}{{c|}}{{\\textbf{{Cross-Entropy Loss}}}} &
\\multicolumn{{3}}{{c}}{{\\textbf{{Perplexity}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
& \\textbf{{0.6B}} & \\textbf{{1.7B}} & \\textbf{{4B}} & \\textbf{{0.6B}} & \\textbf{{1.7B}} & \\textbf{{4B}} \\\\
\\midrule
{table_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}

"""
    return latex

def create_lr_comparison_table(dataset_key, lr_perplexities, lr_losses):
    """Create table with LR adjustments."""
    info = DATASET_INFO[dataset_key]
    dataset_display = info['name']
    lr_config = LR_ADJUSTED_DATASETS[dataset_key]

    # Build rows
    rows = []
    train_dataset = dataset_key

    for eval_name in sorted(lr_perplexities.keys()):
        if eval_name.lower() == 'average':
            continue

        ppl_vals = lr_perplexities[eval_name]  # [0.6B, 1.7B_orig, 1.7B_adj, 4B_orig, 4B_adj]
        loss_vals = lr_losses.get(eval_name, [0, 0, 0, 0, 0])

        # Highlight training dataset
        if eval_name.lower() == train_dataset or eval_name.lower().replace('_', ' ') == train_dataset:
            row_start = r'\rowcolor{gray!20} \textbf{' + eval_name.replace('_', ' ').title() + ' (train)}'
        else:
            row_start = eval_name.replace('_', ' ').title()

        # Loss row
        loss_row = f"{row_start} & {format_value(loss_vals[0])} & {format_value(loss_vals[1])} & {format_value(loss_vals[2])} & " + \
                   f"{format_value(loss_vals[3])} & {format_value(loss_vals[4])} & " + \
                   f"{format_value(ppl_vals[0])} & {format_value(ppl_vals[1], True)} & {format_value(ppl_vals[2], True)} & " + \
                   f"{format_value(ppl_vals[3], True)} & {format_value(ppl_vals[4], True)} \\\\"
        rows.append(loss_row)

    # Add average row
    if 'Average' in lr_perplexities:
        avg_ppl = lr_perplexities['Average']
        avg_loss = lr_losses.get('Average', [0, 0, 0, 0, 0])
        avg_row = r'\rowcolor{blue!10} \textbf{Average} & ' + \
                  f"\\textbf{{{format_value(avg_loss[0])}}} & \\textbf{{{format_value(avg_loss[1])}}} & \\textbf{{{format_value(avg_loss[2])}}} & " + \
                  f"\\textbf{{{format_value(avg_loss[3])}}} & \\textbf{{{format_value(avg_loss[4])}}} & " + \
                  f"\\textbf{{{format_value(avg_ppl[0], True)}}} & \\textbf{{{format_value(avg_ppl[1], True)}}} & \\textbf{{{format_value(avg_ppl[2], True)}}} & " + \
                  f"\\textbf{{{format_value(avg_ppl[3], True)}}} & \\textbf{{{format_value(avg_ppl[4], True)}}} \\\\"
        rows.append(avg_row)

    table_rows = '\n'.join(rows)

    latex = f"""% {dataset_display} Dataset: Evaluation Results with LR Adjustments
% Training: {info['name']} ({info['hf_id']}, {info['size']})
% LR Adjustments: 1.7B ({lr_config['1.7B'][0]} → {lr_config['1.7B'][1]}), 4B ({lr_config['4B'][0]} → {lr_config['4B'][1]})

\\begin{{table}}[h]
\\centering
\\caption{{{dataset_display} Dataset: Impact of Learning Rate Adjustments}}
\\label{{tab:{dataset_key}_lr_comparison}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|c|cc|cc|c|cc|cc}}
\\toprule
\\multirow{{3}}{{*}}{{\\textbf{{Eval Dataset}}}} &
\\multicolumn{{5}}{{c|}}{{\\textbf{{Cross-Entropy Loss}}}} &
\\multicolumn{{5}}{{c}}{{\\textbf{{Perplexity}}}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}}
& \\textbf{{0.6B}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{1.7B}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{4B}}}} &
\\textbf{{0.6B}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{1.7B}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{4B}}}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{8-9}} \\cmidrule(lr){{10-11}}
& \\textbf{{{lr_config['1.7B'][0]}}} & \\textbf{{{lr_config['1.7B'][0]}}} & \\textbf{{{lr_config['1.7B'][1]}}} & \\textbf{{{lr_config['4B'][0]}}} & \\textbf{{{lr_config['4B'][1]}}} &
\\textbf{{{lr_config['1.7B'][0]}}} & \\textbf{{{lr_config['1.7B'][0]}}} & \\textbf{{{lr_config['1.7B'][1]}}} & \\textbf{{{lr_config['4B'][0]}}} & \\textbf{{{lr_config['4B'][1]}}} \\\\
\\midrule
{table_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}

"""
    return latex

def main():
    results_dir = Path('/Users/mengzhao/thesis_lancy/experimental_results')
    output_dir = Path('/Users/mengzhao/thesis_lancy/thesis/tables')
    output_dir.mkdir(exist_ok=True)

    result_files = sorted(results_dir.glob('results_*.md'))

    print(f"Generating tables for {len(result_files)} datasets...")
    print()

    for filepath in result_files:
        print(f"Processing: {filepath.name}")
        dataset_key, perplexities, losses, lr_ppl, lr_loss = parse_results_file(filepath)

        if not perplexities and not lr_ppl:
            print(f"  Warning: No data found")
            continue

        # Generate appropriate table
        has_lr_data = bool(lr_ppl and lr_loss and 'Average' in lr_ppl and 'Average' in lr_loss)
        if has_lr_data:
            latex = create_lr_comparison_table(dataset_key, lr_ppl, lr_loss)
            output_file = output_dir / f'table_{dataset_key}_lr_comparison.tex'
            print(f"  → Created LR comparison table")
        else:
            latex = create_simple_table(dataset_key, perplexities, losses)
            output_file = output_dir / f'table_{dataset_key}_results.tex'
            print(f"  → Created results table")
            if lr_ppl or lr_loss:
                print(f"  Warning: Partial LR data found but incomplete")

        with open(output_file, 'w') as f:
            f.write(latex)

        print(f"  Saved: {output_file.name}")
        print()

    print(f"All tables saved to: {output_dir}")

if __name__ == '__main__':
    main()