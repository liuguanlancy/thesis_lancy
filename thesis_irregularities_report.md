**Thesis Irregularities Report**

- Author: Guanlan Liu
- Scope: LaTeX sources under `thesis/` (chapters, tables, figures)
- Goal: Flag contradictions, inconsistencies, and polish items (no edits applied)

**Build Status**

- Compiles successfully to 70 pages; no undefined references/citations found. See `thesis/main.log`.
- Warnings summary from `thesis/main.log`:
  - Overfull lines: 15 occurrences (various chapters)
  - Underfull lines: 5 occurrences
  - Float placement changed: 28 times (“h” -> “ht”)

**High-Priority Contradictions**

- Mixed Financial: “Inferior on all metrics” vs “best scaling/best overall.”
  - Claims inferiority: `thesis/chapters/chapter4_results.tex:43` (section title), `thesis/chapters/chapter4_results.tex:53` (Key Insight paragraph)
  - Claims best performance/generalization later: `thesis/chapters/chapter4_results.tex:267`, `thesis/chapters/chapter4_results.tex:343`, `thesis/chapters/chapter4_results.tex:361`, `thesis/chapters/chapter4_results.tex:476`
  - Action: Resolve the narrative to one consistent position (either “medium individual datasets dominate” or “mixed financial dominates”). Update summary paragraphs and captions accordingly.

**Figures, Tables, Assets**

- Images: All `\includegraphics{...}` targets exist and load (15 entries verified).
- Tables: All `\input{tables/...}` files exist (19 entries verified).

**Labels and Cross-References**

- Referenced labels appear defined; no undefined references in the log.
- Labels defined but not referenced (optional to fix):
  - `tab:financial_qa_results`, `tab:news_articles_results`, `tab:twitter_results`, `tab:wikitext_results`
- Action: Either add references in text (e.g., “see Table ...”) or remove labels if not needed.

**Terminology Consistency**

- “WikiText” vs “Wikitext” mixed casing in tables vs text.
  - Examples: `thesis/tables/table_wikitext_results.tex:7` (WikiText) vs `thesis/tables/table_wikitext_results.tex:21` (Wikitext); `thesis/tables/table_news_articles_results.tex:22` (Wikitext)
  - Action: Standardize to “WikiText” throughout tables/captions for consistency.

**LaTeX Style Warnings**

- Float placement: many figures use `[h]`, triggering “changed to ht” warnings.
  - Examples: `thesis/chapters/chapter4_results.tex:55,78,101,124,141,148,173,180,187,210,217,234,363,478`; `thesis/chapters/chapter3_methodology.tex:99`
  - Action: Prefer `[htbp]` or `[H]` (with `\usepackage{float}` already present).
- Overfull/underfull boxes: present in chapters 4 and 6 (and bibliography).
  - Action: Typical fixes include `\usepackage[protrusion=true,expansion=true]{microtype}`; `\emergencystretch=2em`; rephrasing long inline math/URLs; allowing page breaks with manual `\\` or `\allowdisplaybreaks` (already set for math).

**Placeholders and Minor Wording**

- Submission date placeholder on title page: `thesis/preamble.tex:172` (“Date of Submission: [ Date ]”).
- Non-standard phrase “three-folded” in abstract: `thesis/main.tex:35`. Suggest “threefold” or “three main contributions.”

**Counts and Coverage (for clarity, not errors)**

- Totals used consistently as “30 models, 237 evaluations” across chapters: `thesis/chapters/chapter4_results.tex:5,474`; `thesis/chapters/chapter5_discussion.tex:7`.
- Mixed-corpus sizes are internally consistent:
  - Mixed Financial ≈ 219.77M tokens (e.g., `thesis/chapters/chapter3_methodology.tex:95`, `thesis/tables/table_mixed_financial_results.tex:2`)
  - Mixed Wiki+Financial ≈ 343.35M tokens (e.g., `thesis/tables/table_mixed_wiki_financial_results.tex:2`, `thesis/chapters/chapter4_results.tex:66`)

**Recommended Fix Plan**

- Unify the core conclusion for Mixed Financial vs Medium Individual datasets (update chapter 4 summary sections and any conflicting captions).
- Standardize “WikiText” casing in all tables and captions.
- Replace `[h]` with `[htbp]` (or `[H]` where exact placement is required) for figures/tables listed above.
- Add `\usepackage[protrusion=true,expansion=true]{microtype}` to `thesis/preamble.tex` and consider `\emergencystretch=2em` to reduce overfull boxes.
- Fill in the submission date on the title page and revise “three-folded” wording.

If you want, I can apply these edits directly (or draft a patch touching only the exact lines referenced above).
