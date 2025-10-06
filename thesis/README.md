# Master's Thesis: Data Mixture Effects in Financial Language Model Pretraining

Author: Guanlan Liu
Supervisor: Prof. Dr. Markus Leippold
University of Zurich

## Structure

```
thesis/
├── thesis_guanlan_liu_19-768-837.tex    # Main LaTeX file
├── preamble.tex                # Preamble with packages and settings
├── references.bib              # Bibliography file
├── chapters/
│   ├── chapter1_introduction.tex      # ✅ COMPLETED (3 pages)
│   ├── chapter2_background.tex        # ⏳ TO WRITE (5 pages)
│   ├── chapter3_methodology.tex       # ⏳ TO WRITE (6 pages)
│   ├── chapter4_results.tex           # ⏳ TO WRITE (10 pages)
│   ├── chapter5_discussion.tex        # ⏳ TO WRITE (4 pages)
│   └── chapter6_conclusion.tex        # ⏳ TO WRITE (2 pages)
├── figures/                    # Figures directory
└── tables/                     # Tables directory
```

## Compilation

To compile the thesis (pdfLaTeX + Biber):

```bash
cd thesis/
pdflatex thesis_guanlan_liu_19-768-837.tex
biber thesis_guanlan_liu_19-768-837
pdflatex thesis_guanlan_liu_19-768-837.tex
pdflatex thesis_guanlan_liu_19-768-837.tex
```

Or use your preferred LaTeX editor (VSCode with LaTeX Workshop, TeXMaker, Overleaf).

## Chapter Status

### Chapter 1: Introduction ✅ COMPLETED
- 3 pages written covering:
  - Motivation (privacy-preserving financial AI)
  - Research questions (4 core questions)
  - Contributions (6 major contributions)
  - Thesis organization
  - Scope and limitations

### Chapter 2: Background and Related Work ⏳ TO WRITE
- 5 pages planned:
  - Financial NLP (1 page)
  - Language Model Pretraining (1.5 pages)
  - Data Mixture Strategies (1.5 pages)
  - Domain Adaptation and Transfer Learning (1 page)

### Chapter 3: Methodology ⏳ TO WRITE
- 6 pages planned:
  - Experimental Design Overview (1 page)
  - Model Architecture (0.75 page)
  - Datasets (2 pages)
  - Training Setup and Hyperparameter Tuning (1.25 pages)
  - Evaluation Protocol (1 page)

### Chapter 4: Results ⏳ TO WRITE
- 10 pages planned:
  - Overview (0.5 page)
  - Data Mixture Effects (3 pages)
  - Individual Dataset Analysis (2 pages)
  - Training Dynamics and LR Scaling (2.5 pages)
  - Domain Transfer Patterns (1.5 pages)
  - Summary (0.5 page)

### Chapter 5: Discussion ⏳ TO WRITE
- 4 pages planned:
  - Key Empirical Findings (1 page)
  - Interpretation of Data Interactions (1.5 pages)
  - Practical Guidelines (1 page)
  - Limitations (0.5 page)

### Chapter 6: Conclusion ⏳ TO WRITE
- 2 pages planned:
  - Summary of Contributions (1 page)
  - Implications (0.5 page)
  - Future Directions (0.5 page)

## Total: 30 pages

## Key Findings to Highlight

1. **Data Mixture**: Mixed Financial (21.55 ppl, mean across financial evaluations) >> WikiText (31.54 ppl average across evaluations; 27.19 ppl on WikiText test set; 41.96 ppl mean across financial evaluations after LR adjustment)
2. **LR Scaling**: 0.6B (2e-5) → 1.7B (1e-5) → 4B (5e-6)
3. **Dataset Size**: <20K samples require mixing, >100K viable standalone
4. **Experiments**: 10 experiments, 30 models, 237 evaluations (Mixed Financial excludes WikiText evaluation)

## Next Steps

1. Write Chapter 2 (Background) - review literature
2. Write Chapter 3 (Methodology) - document experiments
3. Write Chapter 4 (Results) - analyze experimental_results/
4. Write Chapter 5 (Discussion) - interpret findings
5. Write Chapter 6 (Conclusion) - summarize contributions
6. Create figures and tables
7. Proofread and format
8. Final submission
