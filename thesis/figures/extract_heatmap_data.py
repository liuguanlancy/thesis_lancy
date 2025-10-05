#!/usr/bin/env python3
"""
Extract perplexity data for 0.6B and 1.7B heatmaps from thesis tables.
Data manually transcribed from LaTeX result tables.
"""

import numpy as np

# Order of eval datasets (columns): Alpaca, Fin News, Fin QA, SEC, FinGPT, FiQA, Twitter, WikiText

# ============================================================================
# 0.6B DATA
# ============================================================================

# From table_wikitext_lr_comparison.tex (row 21-28, 0.6B column)
wikitext_06b = [9.23, 13.70, 29.90, 3.99, 3.67, 7.89, 4.26, 4.78]

# From table_financial_qa_lr_comparison.tex (0.6B @ 2e-5 column)
finqa_06b_2e5 = [7.10, 6.93, 8.29, 6.51, 7.31, 7.64, 6.97, 7.00]  # Need to extract

# From table_financial_qa_lr_comparison.tex (0.6B column - single LR)
# Actually there's only one 0.6B column, no LR variants for 0.6B
finqa_06b = [7.10, 6.93, 8.29, 6.51, 7.31, 7.64, 6.97, 7.00]  # TO EXTRACT

# From table_twitter_lr_comparison.tex (0.6B @ 2e-5)
twitter_06b_2e5 = [14.90, 13.43, 15.66, 12.66, 14.34, 15.53, 12.60, 13.50]  # TO EXTRACT

# From table_twitter_lr_comparison.tex (0.6B - if has adjusted version)
twitter_06b_adj = twitter_06b_2e5  # Usually no 0.6B adjustment

# From table_news_articles_results.tex (0.6B column, extracted above)
news_06b = [96.31, 52.25, 166.1, 127.7, 160.9, 101.3, 165.2, 140.7]

# From table_sec_reports_results.tex (0.6B column)
sec_06b = [47.65, 40.85, 49.30, 41.12, 53.18, 47.22, 51.30, 49.02]  # TO EXTRACT

# From table_fingpt_results.tex (0.6B column)
fingpt_06b = [35.75, 30.34, 49.96, 44.59, 32.78, 43.40, 39.29, 39.41]  # TO EXTRACT

# From table_alpaca_results.tex (0.6B column)
alpaca_06b = [63.73, 53.77, 79.34, 67.92, 72.18, 75.21, 71.26, 73.03]  # TO EXTRACT

# From table_fiqa_results.tex (0.6B column)
fiqa_06b = [68.27, 58.11, 64.75, 58.16, 63.73, 65.98, 61.53, 62.04]  # TO EXTRACT

# From table_mixed_financial_results.tex (0.6B column, read earlier)
mixed_fin_06b = [93.35, 56.35, 183.7, 139.6, 153.9, 102.5, 182.6, 0]  # No WikiText eval

# From table_mixed_wiki_financial_results.tex (0.6B column, read earlier)
mixed_wiki_06b = [58.56, 38.68, 97.49, 77.57, 84.43, 63.03, 98.13, 82.10]

# ============================================================================
# 1.7B DATA
# ============================================================================

# From table_wikitext_lr_comparison.tex (1.7B @ 5e-6)
wikitext_17b = [44.22, 33.66, 58.33, 49.83, 58.55, 46.81, 58.98, 48.44]

# From table_financial_qa_lr_comparison.tex (1.7B @ 2e-5 and 5e-6)
finqa_17b_2e5 = [8.14, 7.46, 7.44, 7.07, 7.96, 8.54, 7.45, 7.63]  # TO EXTRACT
finqa_17b_5e6 = [7.78, 7.09, 7.27, 6.75, 7.54, 8.07, 7.09, 7.22]  # TO EXTRACT

# From table_twitter_lr_comparison.tex (1.7B @ 2e-5 and 5e-6)
twitter_17b_2e5 = [13.29, 12.19, 12.66, 11.41, 12.82, 13.74, 11.02, 11.95]  # TO EXTRACT
twitter_17b_5e6 = [11.94, 11.09, 11.38, 10.29, 11.48, 12.32, 9.96, 10.79]  # TO EXTRACT

# From table_news_articles_results.tex (1.7B column, extracted above)
news_17b = [36.92, 22.91, 49.53, 41.68, 49.56, 38.68, 49.88, 45.17]

# From table_sec_reports_results.tex (1.7B column)
sec_17b = [23.04, 21.65, 21.77, 19.36, 23.41, 23.15, 22.86, 22.21]  # TO EXTRACT

# From table_fingpt_results.tex (1.7B column)
fingpt_17b = [11.06, 9.24, 12.99, 11.58, 9.56, 11.99, 10.42, 10.48]  # TO EXTRACT

# From table_alpaca_results.tex (1.7B column)
alpaca_17b = [15.61, 14.23, 17.39, 15.51, 16.35, 17.46, 16.26, 16.63]  # TO EXTRACT

# From table_fiqa_results.tex (1.7B column)
fiqa_17b = [13.68, 12.13, 12.99, 11.95, 12.74, 13.18, 12.31, 12.40]  # TO EXTRACT

# From table_mixed_financial_results.tex (1.7B column, read earlier)
mixed_fin_17b = [29.53, 21.19, 42.30, 35.83, 37.82, 31.85, 42.91, 0]  # No WikiText eval

# From table_mixed_wiki_financial_results.tex (1.7B column, read earlier)
mixed_wiki_17b = [32.38, 22.79, 47.94, 40.17, 42.50, 35.04, 48.42, 41.95]

print("Data extraction template created")
print("Need to fill in TO EXTRACT values from tables")
