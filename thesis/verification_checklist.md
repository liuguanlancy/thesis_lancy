# Verification Checklist – Chapter 1 (Introduction)

For this Extreme pass, no inline verification notes are left in the LaTeX. Please confirm the following items against your sources, logs, and figures/tables before finalizing.

- Token/config counts
  - Confirm “10 pretraining configurations across three sizes” (thesis/chapters/chapter1_introduction.tex:9) matches `tables/table_experimental_settings.tex` and Chapter 3 description.
  - Confirm “30 trained models, 240 evaluations, eight held-out test sets” (thesis/chapters/chapter1_introduction.tex:31) match Chapter 3 and Chapter 4 summaries.

- Headline metrics and thresholds
  - Mixed Financial 4B: 21.55 ppl; Mixed Wiki+Financial 4B: 26.69 ppl; WikiText: 48.7 ppl (thesis/chapters/chapter1_introduction.tex:20) match the reported values in Chapter 4 tables/figures (e.g., `tab:mixed_financial_results`, `tab:mixed_wiki_financial_results`, `fig:scaling_comparison_all`).
  - Approximate ratio “About 2.3× worse” for 48.7 vs 21.55 (thesis/chapters/chapter1_introduction.tex:38) is acceptable given rounding; confirm if you prefer exact phrasing.
  - Dataset-size thresholds: tokens (≥100M stable; <20M severe overtraining, variance 89–97%) referenced at thesis/chapters/chapter1_introduction.tex:26 align with analyses and figures in Chapters 3–5 (`fig:scaling_news_articles`, `fig:scaling_sec_reports`, `fig:scaling_financial_qa`, `fig:scaling_twitter`).

- Learning rate details
  - Main LR=2e-5 and follow-up LRs of 1e-5 and 5e-6 (thesis/chapters/chapter1_introduction.tex:23) match Chapter 3 “LR adjustments” and LR-comparison tables (`tab:financial_qa_lr_comparison`, `tab:twitter_lr_comparison`).

- Figure and table counts
  - “11 scaling figures” and “18 tables” claim (thesis/chapters/chapter1_introduction.tex:38, 69) matches the actual counts in `thesis/figures` and `thesis/tables` used in Chapter 4.

- Mixture strategy
  - 50cap description in Chapter 3 is consistent with the Chapter 1 summary statements (thesis/chapters/chapter1_introduction.tex:81).

- Code release statement
  - “We release a complete codebase” (thesis/chapters/chapter1_introduction.tex:59) — ensure the repository link and licensing are set appropriately in the final thesis or appendix, or adjust wording if code will not be public.

- GDPR mention
  - The GDPR reference (thesis/chapters/chapter1_introduction.tex:5) is correct in context; verify institutional guidance on phrasing for compliance discussions.

If any item differs from your final datasets, runs, or counts, please update Chapter 1 wording to match the final artifacts (numbers must not drift from the reported results).

Ultra Pass note: Chapter 1 edits further vary cadence and transitions but do not change any numbers, labels, figure/table refs, or citations. Please re-check the exact values listed above post-edit.

---

# Verification Checklist – Chapter 2 (Background and Related Work)

- Citations: Confirm financial NLP and model references align with the intended sources (FinBERT, BloombergGPT, FinGPT; curriculum/mixture/transfer papers).
- Causal LM description: Ensure the objective/architecture description matches the models actually used (decoder-only; attention/FFN details).
- Compute/memory setup: Verify use of bfloat16 with gradient accumulation and activation checkpointing reflects training scripts/logs.
- Learning rate: Confirm “main runs used LR=2e-5; some reduced for stability” matches Chapter 3 configs and LR comparison tables.
- Mixture performance numbers: Reconfirm 4B Mixed Financial (21.55 ppl) vs Mixed Wiki+Financial (26.69 ppl; ~24% worse) agree with `tab:mixed_financial_results`, `tab:mixed_wiki_financial_results`, and `fig:scaling_comparison_all`.
- Scope note: Only stylistic and connector adjustments were made; no new claims, datasets, or citations were introduced.

---

# Verification Checklist – Chapter 3 (Methodology)

- Experimental design: 10 configurations; 3 sizes; 100M-token budget; 8 held-out test sets match `tables/table_experimental_settings.tex` and Chapter 4.
- Model specs table: Parameters, layers, hidden sizes, heads, GQA, memory values for Qwen3 (0.6B/1.7B/4B) are correct.
- Dataset table: Example and token counts for 7 financial datasets and WikiText-103 match source preprocessing logs and table captions.
- Mixture strategy (50cap): Step 1/Step 2 description matches your sampling implementation.
- Training setup: Optimizer, LR, schedule, warmup, batch, precision, sequence length match run configs.
- LR adjustments: Only the three affected cases (WikiText, Financial QA, Twitter at larger sizes) used smaller LRs; cross-check with LR comparison tables.
- Compute budget: 36 runs × 100M tokens = 3.6B tokens; runtime estimates on Lambda Labs A100 are consistent with logs.
- Ultra Pass note: Only narrative cadence and connector changes; all values, labels, and tables remain identical to prior pass.

---

# Verification Checklist – Chapter 4 (Results)

- Overview table values: Verify perplexities for each experiment/best model match the tables included under `thesis/tables`.
- Mixed Financial: mean/CV values and per-set perplexities (News 15.2; SEC 18.7; FinGPT 19.4; Alpaca 21.8; FiQA 14.6; Financial QA 23.1; Twitter 25.9; WikiText 33.7) correspond to plotted figures and CSV sources.
- “Cross-dataset consistency” phrasing aligns with the same 55% CV value; no metric change implied.
- Large datasets section: News and SEC results (perplexities, spreads) match `table_news_articles_results.tex` and `table_sec_reports_results.tex`.
- Medium/small datasets: Perplexity numbers, epochs ranges, and CV percentages match corresponding tables and figures.
- Reverse scaling notes: LR reductions and recovered metrics (e.g., Twitter 4B to 12.35 ppl) match plots (`figures/scaling_twitter.png`) and LR tables.
- Variance-performance discussion: Ensure the CV numbers and mean PPL cited in that section are consistent with the listed tables.
- Ultra Pass note: Only phrasing/connector tweaks; no changes to any results, figures, tables, or labels.

---

# Verification Checklist – Chapter 5 (Discussion)

- Finding summaries: Check the exact values (21.55, 26.69, 48.7; 55%/62%/78% CV; 2.3× gap) against Chapter 4.
- Correlations and variance claims: r = 0.82 (News↔SEC), r = 0.68–0.73 (instruction cluster), 89% variance for Twitter, log(tokens) vs variance r = -0.78.
- Trade-offs: “29% financial performance for 16% general improvement” and other percentages reflect your computed deltas.
- Ultra Pass note: Added brief connective phrases and emphasis only; all quantitative references and citations unchanged.
- No new claims introduced; all numerical statements sourced from Chapter 4 tables/figures.

---

# Verification Checklist – Chapter 6 (Conclusion)

- Thresholds: ≥100M stable; <20M severe; CV values by dataset size category agree with prior chapters.
- Asymmetry statements: WikiText 48.7 vs Mixed Financial 21.55 and general→financial failure are consistent with Chapter 4.
- Model size guidance: 1.7B “recommended” rationale matches performance/memory/time numbers; 0.6B/4B trade-offs reflect actual runs.
- Industry guidance: Claims about privacy/compliance and cost must align with your institutional policy and any publicly shareable cost estimates.
- Future work items (dynamic schedules, adaptation stages) are framed as suggestions; no implied new results.
- Ultra Pass note: Only style cadence/phrasing changes; all metrics, names, and references preserved.
