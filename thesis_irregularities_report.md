**Thesis Irregularities Report**

- Author: Guanlan Liu
- Scope: Thesis LaTeX sources under `thesis/` and related tables
- Goal: Flag inconsistencies, contradictions, and polish issues (no changes applied)

**Executive Summary**

- Critical: Evaluation coverage and totals are inconsistent across chapters (30/237 vs 36/288; mixed models claim “unified eight‑dataset evaluation” but exclude WikiText).
- Critical: Dataset token counts conflict between methodology and tables (e.g., SEC 8.1M vs 80M; FinGPT 4.1M vs 19M; Alpaca 8.5M vs 17M; FiQA 3.6M vs 4M).
- Critical: Several cross‑dataset tables contain implausible/performance‑inverting values (e.g., tiny Financial QA model outperforming SEC model on SEC test set by 2×+), contradicting single‑dataset tables and the narrative.
- High: Mixed Financial total tokens vary (220–322M); financial news size varies (194–197M).
- High: README content conflicts with manuscript (findings summary and counts).
- Medium: Typos/labels and LaTeX warnings (overfull boxes, float specifiers).

**Evaluation Coverage and Totals**

- Mixed Financial excludes WikiText despite “unified eight‑dataset evaluation” claim.
  - `thesis/main.tex:33`
  - `thesis/tables/table_experimental_settings.tex:22`
  - `thesis/chapters/chapter4_results.tex:5`

- Conflicting totals for models/evaluations:
  - “36 models, 288 evaluations” vs “30 models, 237 evaluations”.
  - `thesis/chapters/chapter4_results.tex:5`
  - `thesis/chapters/chapter4_results.tex:474`
  - `thesis/chapters/chapter5_discussion.tex:7`
  - `thesis/README.md:93`

**Dataset Token Count Inconsistencies**

- SEC Reports: 8.1M vs 80M
  - 8.1M: `thesis/chapters/chapter3_methodology.tex:51`, `thesis/chapters/chapter3_methodology.tex:68`
  - 80M (comment): `thesis/tables/table_sec_reports_results.tex:2`

- FinGPT Sentiment: 4.1M vs 19M
  - 4.1M: `thesis/chapters/chapter3_methodology.tex:53`, `thesis/chapters/chapter3_methodology.tex:68`
  - 19M (comment): `thesis/tables/table_fingpt_results.tex:2`

- Finance Alpaca: 8.5M vs 17M
  - 8.5M: `thesis/chapters/chapter3_methodology.tex:55`, `thesis/chapters/chapter3_methodology.tex:68`
  - 17M (comment): `thesis/tables/table_alpaca_results.tex:2`

- FiQA: 3.6M vs 4M (caption)
  - 3.6M: `thesis/chapters/chapter3_methodology.tex:59`, `thesis/chapters/chapter3_methodology.tex:68`
  - 4M: `thesis/chapters/chapter4_results.tex:166`

- Financial QA 10K: 0.7M vs 3.5M (comment)
  - 0.7M: `thesis/chapters/chapter3_methodology.tex:59`, `thesis/chapters/chapter3_methodology.tex:68`
  - 3.5M (comment): `thesis/tables/table_financial_qa_results.tex:2`

- Twitter Financial Sentiment: 0.28M vs 0.3M (comment)
  - 0.28M: `thesis/chapters/chapter3_methodology.tex:61`, `thesis/chapters/chapter3_methodology.tex:68`
  - 0.3M (comment): `thesis/tables/table_twitter_results.tex:2`

- Financial News Articles: 194–197M
  - 194.5M: `thesis/chapters/chapter3_methodology.tex:49`
  - 197M: `thesis/tables/table_news_articles_results.tex:2`

- Mixed Financial total tokens: 220–322M
  - 220M: `thesis/chapters/chapter4_results.tex:23`
  - 322M (comment): `thesis/tables/table_mixed_financial_results.tex:2`

**Cross‑Table and Metric Anomalies**

- SEC evaluation cross‑table shows tiny or general‑domain training outperforming SEC‑trained models by large margins, contradicting dataset‑specific results and narrative.
  - SEC cross table: `thesis/tables/table_cross_financial_repor.tex:16` (e.g., WikiText 0.6B 3.99 PPL; Financial QA 0.6B 8.21 PPL) vs SEC table `thesis/tables/table_sec_reports_results.tex:16` (4B 15.91 PPL)

- Financial News cross table shows FiQA 4B at 7.43 PPL on News, better than News‑trained 17.47 PPL; conflicts with claims that document models do best on document test sets.
  - `thesis/tables/table_cross_financial_news.tex:16`
  - `thesis/tables/table_news_articles_results.tex:12`

- WikiText cross table indicates extreme reverse scaling (0.6B 4.78; 1.7B 30.63; 4B 27.19) — ensure this matches raw logs and that the LR adjustment rows are consistent with your described interventions.
  - `thesis/tables/table_cross_wikitext.tex:18`

- Overview table in results (averages, spreads) asserts “medium datasets dominate” and seems aligned with per‑dataset tables; however, the small‑dataset and cross‑dataset tables above undermine that narrative unless those anomalous entries are corrected or explained.
  - `thesis/chapters/chapter4_results.tex:9`

**Methodological Clarity**

- If Mixed Financial excludes WikiText evaluation, explicitly state the unified evaluation is “7 financial + WikiText, with WikiText omitted only for the pure‑financial mixture rows” or adjust the claim and totals throughout.
  - Affects: `thesis/main.tex:33`, `thesis/chapters/chapter3_methodology.tex:149`, `thesis/chapters/chapter4_results.tex:5`

- Reconcile “36 models, 288 evaluations” vs “30 models, 237 evaluations” by defining which follow‑up LR runs are included in totals, and whether Mixed Financial rows are counted with 7 or 8 evaluations.

**Repository Documentation Conflicts**

- README contradicts manuscript:
  - Claims Mixed Financial >> WikiText in key findings; manuscript now frames medium individual datasets as best.
  - Also carries 30/237 totals.
  - `thesis/README.md:93`

**Typos / Labels**

- Label/file typo: “repor” instead of “report”.
  - `thesis/tables/table_cross_financial_repor.tex:6`
  - References across chapters (e.g., `thesis/chapters/chapter4_results.tex:171`)

- “three-folded” phrasing is consistent with kmbart style but non‑standard; acceptable if intentional.

**LaTeX Warnings (non‑blocking)**

- Overfull \hbox and float placement adjustments were reported during build; optional polish.
  - Overfull lines: `thesis/chapters/chapter4_results.tex:458`, `thesis/chapters/chapter6_conclusion.tex:11`, `thesis/chapters/chapter6_conclusion.tex:95`
  - Many “`h` float specifier changed to `ht`” warnings (multiple tables/figures).

**Placeholders / Metadata**

- Title page has a placeholder submission date.
  - `thesis/preamble.tex:126`

**Suggested Fix Plan (no edits performed)**

- Pick a single source of truth for dataset sizes (tokens) and update all mentions (tables, captions, narrative) to match.
- Decide and document evaluation coverage (7 vs 8 test sets) for Mixed Financial; recompute totals and adjust all counts consistently.
- Audit and correct cross‑dataset tables with anomalous values (SEC and News cases above). Ensure they align with per‑dataset tables and narrative.
- Align README with the manuscript’s final findings and totals, or mark it as outdated.
- Optionally resolve LaTeX warnings (microtype/urls, tweak long lines).

