# Dataset Statistics Correction Report

## Data Sources Used

1. **scripts/analyze_mixture_ratios.py** - Contains the canonical dataset statistics used for training
2. **scripts/phase2b_financial_pretraining.sh** - Confirms example counts in comments
3. **verify_dataset_stats.py** - Direct HuggingFace API verification (for reference only)

## Canonical Values (from analyze_mixture_ratios.py)

These are the values that were actually used in the training experiments:

| Dataset | Examples | Tokens (M) |
|---------|----------|------------|
| Financial Q&A (virattt/financial-qa-10K) | 7,000 | 0.70 |
| FinGPT (FinGPT/fingpt-sentiment-train) | 76,000 | 4.14 |
| Finance Alpaca (gbharti/finance-alpaca) | 68,000 | 8.46 |
| FiQA (LLukas22/fiqa) | 15,000 | 3.60 |
| Twitter (zeroshot/twitter-financial-news-sentiment) | 10,000 | 0.28 |
| SEC Reports (JanosAudran/financial-reports-sec:small_lite) | 200,000 | 8.12 |
| News Articles (ashraq/financial-news-articles) | 306,000 | 197.38 |
| WikiText (wikitext:wikitext-103-v1) | 1,801,350 | 103.00 |

## Current Table Values (INCORRECT)

### Table 3.3 (Financial Datasets)
| Dataset | Current Examples | Current Tokens | Correct Examples | Correct Tokens | Status |
|---------|------------------|----------------|------------------|----------------|--------|
| Financial News Articles | 300K | 197M | 306K | 197.4M | ✗ Examples off |
| SEC Reports | 54.3K | 80M | 200K | 8.1M | ✗✗ Both wrong |
| FinGPT Sentiment | 76.8K | 19.1M | 76K | 4.1M | ✗ Tokens off |
| Finance Alpaca | 68.9K | 17.2M | 68K | 8.5M | ✗ Tokens off |
| FiQA | 17.4K | 4.3M | 15K | 3.6M | ✗ Both off |
| Financial QA 10K | 7.1K | 3.5M | 7K | 0.7M | ✗✗ Tokens off |
| Twitter Financial Sentiment | 1.1K | 0.3M | 10K | 0.28M | ✗✗ Both wrong |

### Table 3.4 (WikiText)
| Dataset | Current Examples | Current Tokens | Correct Examples | Correct Tokens | Status |
|---------|------------------|----------------|------------------|----------------|--------|
| WikiText-103 | 103K | 103M | 1.8M | 103M | ✗ Examples wrong |

## Major Issues Identified

### 1. **SEC Reports - SEVERE ERROR**
- **Current**: 54.3K examples, 80M tokens
- **Correct**: 200K examples, 8.1M tokens
- **Problem**: Token count is 10x too high (80M vs 8.1M)!
- **Impact**: This affects mixture calculations and the 50cap discussion in Section 3.3

### 2. **Financial QA 10K - SEVERE ERROR**
- **Current**: 7.1K examples, 3.5M tokens
- **Correct**: 7K examples, 0.7M tokens
- **Problem**: Token count is 5x too high (3.5M vs 0.7M)!

### 3. **Twitter - SEVERE ERROR**
- **Current**: 1.1K examples, 0.3M tokens
- **Correct**: 10K examples, 0.28M tokens
- **Problem**: Example count is 9x too low (1.1K vs 10K)!

### 4. **FinGPT & Finance Alpaca - Token Count Errors**
- **FinGPT Current**: 76.8K examples, 19.1M tokens
- **FinGPT Correct**: 76K examples, 4.1M tokens
- Token count is 4.6x too high!

- **Alpaca Current**: 68.9K examples, 17.2M tokens
- **Alpaca Correct**: 68K examples, 8.5M tokens
- Token count is 2x too high!

### 5. **WikiText-103 - Example Count Error**
- **Current**: 103K examples, 103M tokens
- **Correct**: 1.8M examples, 103M tokens
- **Problem**: Using article count instead of line/example count

## Total Token Counts

### Financial Datasets (7 total)
- **Current table sum**: 321.4M tokens
- **Correct sum**: 222.68M tokens
- **Difference**: -98.72M tokens (30.7% error!)

### With WikiText (8 total)
- **Current**: 424.4M tokens
- **Correct**: 325.68M tokens
- **Difference**: -98.72M tokens (23.3% error!)

## Root Cause Analysis

The errors likely stem from:
1. **SEC Reports**: Counted sentence-level tokens incorrectly or used wrong config
2. **Financial QA**: Misunderstood dataset structure (maybe counted context separately?)
3. **Twitter**: Used test set instead of train set (test set has ~1.1K examples)
4. **FinGPT/Alpaca**: Token counting method included formatting/templates multiple times?
5. **WikiText**: Confused "103" in name (WikiText-103) with example count

## Tokenizer Used

The canonical values in `analyze_mixture_ratios.py` were calculated using:
- **Tokenizer**: Qwen/Qwen3-0.6B-Base
- **Method**: Full dataset processing with dataset-specific text formatting
- **Script**: count_dataset_tokens.py

## Verification Against HuggingFace API

The verification script results (sampling 1000 examples):
- Most example counts match HuggingFace API
- Token counts vary due to:
  - Different tokenizer (Qwen2.5-0.5B vs Qwen3-0.6B-Base)
  - Sampling vs full count
  - Text formatting differences

## Recommended Actions

1. **Update Table 3.3** with correct token/example counts from analyze_mixture_ratios.py
2. **Update Table 3.4** with correct WikiText example count (1.8M, not 103K)
3. **Verify Section 3.3.2** discussion of mixture totals (currently says 321.4M, should be 222.68M)
4. **Check Section 3.3.3** 50cap calculations - the percentages may need updating
5. **Review any epoch calculations** that depend on total token counts

## Corrected LaTeX Snippets

### For Table 3.3 (Financial Datasets)

```latex
Financial News Articles\\\footnotesize \url{ashraq/financial-news-articles} & 306K & 197.4M & Journalism & [TBD] & Long-form articles on markets, earnings, policy \\
\midrule
SEC Financial Reports\\\footnotesize \url{JanosAudran/financial-reports-sec:small_lite} & 200K & 8.1M & Regulatory & 1993–2020 & 10-K annual filings with formal disclosures, legal language \\
\midrule
FinGPT Sentiment\\\footnotesize \url{FinGPT/fingpt-sentiment-train} & 76K & 4.1M & Instruction & 2006–2022 & Headlines + sentiment labels in conversational format \\
\midrule
Finance Alpaca\\\footnotesize \url{gbharti/finance-alpaca} & 68K & 8.5M & Q\&A & N/A$^*$ & Instruction-response pairs on financial concepts \\
\midrule
FiQA\\\footnotesize \url{LLukas22/fiqa} & 15K & 3.6M & Forum & 2009–2017 & User-generated Q\&A from Stack Exchange Investment topic \\
\midrule
Financial QA 10K\\\footnotesize \url{virattt/financial-qa-10K} & 7K & 0.7M & Document & FY 2021–2023 & Questions on recent 10-K filings requiring tabular reasoning \\
\midrule
Twitter Financial News Sentiment\\\footnotesize \url{zeroshot/twitter-financial-news-sentiment} & 10K & 0.28M & Social Media & 2019–2022 & Labeled tweets ($<$280 chars) with informal language \\
```

### For Table 3.4 (WikiText)

```latex
WikiText-103\\\footnotesize \url{wikitext:wikitext-103-v1} & 1.8M & 103M & Encyclopedia & 2016 snapshot & Verified Wikipedia articles from 2016 snapshot with formal register, broad topical coverage, clean preprocessing \\
```

### Total Token Count Update (Section 3.3.1)

**Current text (line 38)**:
> We curate 7 financial datasets spanning diverse tasks, document types, and data scales (total: 321.4M tokens)

**Corrected**:
> We curate 7 financial datasets spanning diverse tasks, document types, and data scales (total: 222.7M tokens)

## Summary Statistics (Corrected)

### Token Distribution by Size
- **Large (>100M)**: News Articles (197.4M)
- **Medium (5-10M)**: Finance Alpaca (8.5M), SEC Reports (8.1M)
- **Small (2-5M)**: FinGPT (4.1M), FiQA (3.6M)
- **Tiny (<1M)**: Financial QA (0.7M), Twitter (0.28M)

### Example Distribution
- **Very Large (>100K)**: WikiText (1.8M), SEC Reports (200K)
- **Large (50-100K)**: News Articles (306K), FinGPT (76K), Finance Alpaca (68K)
- **Small (<20K)**: FiQA (15K), Twitter (10K), Financial QA (7K)
