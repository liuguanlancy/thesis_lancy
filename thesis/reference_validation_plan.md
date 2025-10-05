# Thesis Reference Validation Plan

Purpose: ensure every citation is accurate, complete, consistent, and appropriate for the claim it supports in the thesis.

Inputs
- LaTeX sources: `thesis/main.tex`, `thesis/chapters/*.tex`, `thesis/tables/*.tex`
- Bibliography: `thesis/references.bib`

Tools (suggested)
- Search: `rg` (ripgrep), `grep`, `sed`, `awk`
- LaTeX build: `latexmk` (or your workflow) to surface citation warnings
- Lookups: DBLP, Crossref, OpenAlex, arXiv, ACL Anthology, IEEE Xplore, ACM DL
- Optional helpers: Zotero + Zotero Connector, `bibtex-tidy`, `bibtool`

Workflow

1) Inventory citations and bibliography
- Extract all cite keys used in `.tex` files.
- Extract all entry keys present in `thesis/references.bib`.
- Identify missing entries (cited but not in `.bib`) and unused entries (in `.bib` but never cited).
- Commands
  - List cite keys used: `rg -n "\\cite{[^}]+}" thesis/**/*.tex | sed -E 's/.*\\cite\{([^}]*)\}.*/\1/g' | tr ',' '\n' | tr -d ' ' | sort -u`
  - List bib keys: `rg -n "^@.+\{[[:alnum:]:_\-]+," thesis/references.bib | sed -E 's/^@.+\{([^,]+),.*/\1/g' | sort -u`
  - Quick diff: save both lists to files (e.g., `used.txt`, `bib.txt`) then `comm -23 used.txt bib.txt` (missing) and `comm -13 used.txt bib.txt` (unused).

2) Metadata verification per entry
- Title
  - Exact match with publisher record; preserve intended capitalization for acronyms/names using braces: e.g., `{LLM}`, `{GPT-4}`, `{BERT}`.
- Authors
  - Correct order, spelling, diacritics; prefer full names when available.
- Venue and type
  - Correct venue name (conference/journal/workshop); include series and year where applicable (e.g., NeurIPS 2020, ACL 2023).
  - If a preprint later appeared at a venue, prefer the published version over arXiv unless you explicitly discuss the preprint.
- Year and details
  - Correct publication year; add volume/issue/pages for journals and page range for proceedings if available.
- Identifiers
  - `doi` present and resolves; `url` used mainly for arXiv and preprints unless style requires URLs.
  - For arXiv: include `eprint`, `archivePrefix = {arXiv}`, and `primaryClass`.
- Entry type alignment
  - Ensure correct BibTeX type: `@inproceedings` (conferences), `@article` (journals), `@incollection`/`@book` as needed; avoid using `@misc` when a specific type fits.
- How to verify
  - DBLP/ACL Anthology/IEEE/ACM pages for CS; Crossref/OpenAlex/Publisher sites for DOI and canonical metadata.
  - Spot-check with: `curl -I https://doi.org/<doi>` expecting HTTP 200.

3) Citation–context validation (appropriateness)
- Purpose tagging
  - For each citation occurrence, tag purpose: Background, Method, Dataset, Tool, Baseline/Comparison, Result/Claim, Limitation.
- Claim alignment
  - Ensure the cited work actually supports the specific claim at that location (e.g., if stating SOTA, cite the source that reports SOTA for that task/time).
- Primary sources
  - Prefer the primary, peer-reviewed source; add secondary surveys only for high-level context.
- Up-to-date venue
  - Replace arXiv-only cites with published versions where available and appropriate for the claim.
- Multi-citation hygiene
  - For broad claims, use multiple representative citations; for specific claims, avoid over-citation.
- Practical review method
  - Generate a locations list per cite key and review the surrounding lines in the chapter file for alignment.

4) Style and consistency
- Key naming
  - Enforce a consistent key scheme (e.g., `AuthorYearVenue` or `AuthorYYVenue`), avoid duplicates.
- Venue naming
  - Choose a consistent style: full names vs. common acronyms (e.g., "In Proceedings of the 37th NeurIPS" vs. "NeurIPS 2023").
- Capitalization
  - Protect proper nouns/acronyms in `title` with braces; check for unintended downcasing by BibTeX styles.
- Fields by type
  - `@inproceedings`: `booktitle`, `pages`, `year`, `publisher` if required.
  - `@article`: `journal`, `volume`, `number`, `pages`, `year`.
- URLs and DOIs
  - Prefer DOIs over URLs; keep both only when helpful (e.g., arXiv with `doi` missing).
- Deduplication
  - Merge duplicate entries and pick canonical metadata; remove unused `.bib` items.

5) Automation (optional but recommended)
- Undefined/unused citations during build
  - Build and capture warnings: `latexmk -pdf -interaction=nonstopmode -halt-on-error thesis/main.tex`
  - Address `Undefined citation` and `There were undefined references` warnings.
- Lint/tidy
  - Use `bibtex-tidy` to normalize fields and formatting if available.
  - Optionally `bibtool` to enforce field presence or rename keys.
- DOI resolution check (example)
  - Extract DOIs: `rg -No "doi\s*=\s*\{([^}]+)\}" thesis/references.bib | sed -E 's/.*\{([^}]+)\}.*/\1/' | sort -u > doi.txt`
  - Test resolution: `while read d; do curl -s -o /dev/null -w "%{http_code} %{redirect_url}\\n" https://doi.org/$d; done < doi.txt`

6) Deliverables
- Inventory report
  - `used_vs_bib.md` summarizing missing and unused keys.
- Metadata log
  - `reference_audit.md` table with columns: `key`, `status (ok/fix)`, `issue`, `fix`, `source (URL)`, `checked-by`, `date`.
- Context log
  - `citation_context.md` table listing each cite occurrence and purpose tag; note any misalignment and resolution.
- Clean bibliography
  - Updated `thesis/references.bib` with corrected metadata, removed duplicates, and consistent style.

7) Review cadence and sign-off
- Pass 1: Inventory and quick fixes (missing/unused, obvious metadata errors).
- Pass 2: Deep metadata verification (title/authors/venue/year/DOI) against authoritative sources.
- Pass 3: Citation–context review in each chapter (`chapters/`), tag purposes, fix misaligned cites.
- Pass 4: Style unification and final LaTeX compile with zero citation warnings.
- Final: Freeze `.bib`, rerun build, spot-check PDF bibliography for formatting.

Templates

- Per-entry checklist
  - [ ] Key present in `.bib` and used where intended
  - [ ] Correct type (`@article`/`@inproceedings`/…)
  - [ ] Title correct; protected capitalization for acronyms
  - [ ] Authors correct order and spelling
  - [ ] Venue name and type correct
  - [ ] Year/volume/issue/pages present (as applicable)
  - [ ] DOI resolves; arXiv fields present if applicable
  - [ ] URL only when needed/required
  - [ ] Consistent key naming and no duplicates

- Citation context row (example)
  - `key`: 
  - `file`: 
  - `line`: 
  - `purpose`: Background | Method | Dataset | Baseline | Result | Limitation
  - `claim snippet`: 
  - `appropriateness`: ok | revise | replace
  - `action`: 

Helpful commands
- Find all uses of a key: `rg -n "\\cite\{(<KEY>|[^}]*,<KEY>|<KEY>,[^}]*)\}" thesis/**/*.tex`
- Show surrounding lines: `rg -n "<KEY>" thesis/**/*.tex -C 2`
- Count cite frequency: `rg -o "\\cite\{[^}]+\}" thesis/**/*.tex | wc -l`
- List unused bib entries: see Inventory step with `comm`.

Notes specific to this repo
- Primary sources live under `thesis/chapters/` and `thesis/tables/`; keep reviews chapter-by-chapter.
- The main bibliography file is `thesis/references.bib`; keep all authoritative fixes there and avoid per-chapter `.bib`s.

