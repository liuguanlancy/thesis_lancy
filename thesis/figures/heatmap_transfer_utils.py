#!/usr/bin/env python3
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np


# Canonical column order and label mapping
CANONICAL_ORDER = [
    "alpaca",
    "financial_news",
    "financial_qa",
    "financial_repor",  # SEC reports
    "fingpt",
    "fiqa",
    "twitter",
    "wikitext",
]

CANONICAL_LABELS = {
    "alpaca": "Alpaca",
    "financial_news": "Fin News",
    "financial_qa": "Fin QA",
    "financial_repor": "SEC",
    "fingpt": "FinGPT",
    "fiqa": "FiQA",
    "twitter": "Twitter",
    "wikitext": "WikiText",
}


def _read_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _find_section(lines: List[str], header: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.strip().lower() == header.strip().lower():
            return i
    return None


def _parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[List[str], List[Dict[str, str]]]:
    # Parse a GitHub-style markdown table starting at start_idx (header line)
    headers_line = lines[start_idx]
    sep_line = lines[start_idx + 1] if start_idx + 1 < len(lines) else ""
    if "|" not in headers_line or "-" not in sep_line:
        raise ValueError("Not a markdown table at start index")

    headers = [h.strip() for h in headers_line.strip().strip("|").split("|")]
    rows: List[Dict[str, str]] = []
    i = start_idx + 2
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or not line.strip().startswith("|"):
            break
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != len(headers):
            # end of table or malformed
            break
        row = {headers[j]: parts[j] for j in range(len(headers))}
        rows.append(row)
        i += 1
    return headers, rows


def _coerce_value(val: str) -> float:
    v = val.strip().replace("**", "")
    if v in {"∞", "inf", "Inf", "INF"}:
        return float("inf")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _reorder_to_canonical(row_map: Dict[str, float]) -> List[float]:
    return [row_map.get(k, float("nan")) for k in CANONICAL_ORDER]


def parse_original_perplexities(md_path: str, size_label: str) -> Tuple[List[str], List[float]]:
    """Return (eval_labels_canonical, values) for a given model size from the Perplexity Metrics table.

    size_label: one of "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"
    """
    lines = _read_file(md_path)
    sec_idx = _find_section(lines, "## Perplexity Metrics")
    if sec_idx is None:
        raise FileNotFoundError(f"Perplexity Metrics section not found in {md_path}")
    # Find the first table after this header
    # It should be at sec_idx+3: header, sep, first row lines
    # But we scan forward to the first '|' header line
    i = sec_idx + 1
    while i < len(lines) and "|" not in lines[i]:
        i += 1
    headers, rows = _parse_markdown_table(lines, i)

    # Identify the eval dataset column and the requested size column
    eval_col = headers[0]
    if eval_col.lower() != "eval dataset":
        # Some tables may bold names; still accept
        pass
    if size_label not in headers:
        raise KeyError(f"Size column {size_label} not in table headers {headers}")

    values_map: Dict[str, float] = {}
    for r in rows:
        name = r[eval_col].strip().strip("**")
        if name.lower() == "average":
            continue
        values_map[name] = _coerce_value(r[size_label])

    # Map dataset keys to canonical
    rename = {
        "financial_qa": "financial_qa",
        "fingpt": "fingpt",
        "alpaca": "alpaca",
        "fiqa": "fiqa",
        "twitter": "twitter",
        "financial_repor": "financial_repor",
        "financial_reports": "financial_repor",
        "financial_news": "financial_news",
        "wikitext": "wikitext",
    }

    mapped: Dict[str, float] = {}
    for raw_key, val in values_map.items():
        k = rename.get(raw_key.lower(), None)
        if k is not None:
            mapped[k] = val

    return [CANONICAL_LABELS[k] for k in CANONICAL_ORDER], _reorder_to_canonical(mapped)


def parse_adjusted_perplexities(md_path: str, size_label: str) -> Tuple[Optional[str], List[float]]:
    """Parse the Perplexity Metrics Comparison table and return (lr_label, values) for the given size.

    Returns (None, [nan,...]) if no adjusted column exists for that size.
    lr_label is the exact LR text from the header, e.g., 'LR=5e-6'.
    """
    lines = _read_file(md_path)
    sec_idx = _find_section(lines, "#### Perplexity Metrics Comparison")
    if sec_idx is None:
        return None, [float("nan")] * len(CANONICAL_ORDER)

    i = sec_idx + 1
    while i < len(lines) and "|" not in lines[i]:
        i += 1
    headers, rows = _parse_markdown_table(lines, i)

    # Find the adjusted column for the given size (the one that has parentheses with LR)
    # e.g., 'Qwen3-4B (LR=5e-6)'
    size_prefix = f"{size_label} (LR="
    matched_cols = [h for h in headers if h.startswith(size_prefix)]
    # Require at least two columns (original and adjusted). If only one exists, it's the original.
    if len(matched_cols) < 2:
        return None, [float("nan")] * len(CANONICAL_ORDER)
    adj_col = matched_cols[-1]  # choose the last if multiple

    # Extract LR label within parentheses
    m = re.search(r"\(LR=([^\)]+)\)", adj_col)
    lr_text = f"LR={m.group(1)}" if m else "LR=*"

    eval_col = headers[0]
    values_map: Dict[str, float] = {}
    for r in rows:
        name = r[eval_col].strip().strip("**")
        if name.lower() == "average":
            continue
        values_map[name] = _coerce_value(r.get(adj_col, "nan"))

    # Map dataset keys to canonical
    rename = {
        "financial_qa": "financial_qa",
        "fingpt": "fingpt",
        "alpaca": "alpaca",
        "fiqa": "fiqa",
        "twitter": "twitter",
        "financial_repor": "financial_repor",
        "financial_reports": "financial_repor",
        "financial_news": "financial_news",
        "wikitext": "wikitext",
    }

    mapped: Dict[str, float] = {}
    for raw_key, val in values_map.items():
        k = rename.get(raw_key.lower(), None)
        if k is not None:
            mapped[k] = val

    return lr_text, _reorder_to_canonical(mapped)


def build_rows_for_size(size_short: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Build (train_labels, eval_labels, data_matrix) for the given size.

    size_short: one of {'06b','17b','4b'}
    """
    size_map = {
        "06b": "Qwen3-0.6B",
        "17b": "Qwen3-1.7B",
        "4b": "Qwen3-4B",
    }
    if size_short not in size_map:
        raise ValueError("size_short must be one of {'06b','17b','4b'}")
    size_label = size_map[size_short]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experimental_results"))
    files = [
        ("WikiText", os.path.join(base_dir, "results_wikitext.md")),
        ("Fin QA", os.path.join(base_dir, "results_financial_qa.md")),
        ("Twitter", os.path.join(base_dir, "results_twitter.md")),
        ("News", os.path.join(base_dir, "results_news_articles.md")),
        ("SEC", os.path.join(base_dir, "results_sec_reports.md")),
        ("FinGPT", os.path.join(base_dir, "results_fingpt.md")),
        ("Alpaca", os.path.join(base_dir, "results_alpaca.md")),
        ("FiQA", os.path.join(base_dir, "results_fiqa.md")),
        ("Mixed Fin", os.path.join(base_dir, "results_mixed_financial.md")),
        ("Mixed Wiki+Fin", os.path.join(base_dir, "results_mixed_wiki_financial.md")),
    ]

    train_labels: List[str] = []
    rows: List[List[float]] = []

    # Build original rows for each dataset
    for label, path in files:
        eval_labs, vals = parse_original_perplexities(path, size_label)
        train_labels.append(f"{label} (2e-5)")
        rows.append(vals)

        # If there is an adjusted column for this size, add it as a separate row
        adj_lr, adj_vals = parse_adjusted_perplexities(path, size_label)
        if adj_lr is not None and not all(np.isnan(adj_vals)):
            train_labels.append(f"{label} ({adj_lr.split('=')[1]})")
            rows.append(adj_vals)

    data = np.array(rows, dtype=float)
    # Canonical eval labels (same for all rows)
    eval_labels = [CANONICAL_LABELS[k] for k in CANONICAL_ORDER]
    return train_labels, eval_labels, data
