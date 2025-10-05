# Thesis Overview: Cross-Domain Data Interactions in Language Model Pretraining

## Title
**Understanding Data Mixture Effects in Financial Language Model Pretraining: A Study of Domain-Specific and High-Quality General Corpora**

## Research Question
How do different data sources—both in-domain financial data and out-of-domain high-quality corpora—interact during pretraining, and what are the optimal mixture strategies for developing specialized financial language models?

## Motivation

### 1. **Privacy-Preserving Finance AI**
Financial institutions and individuals handle highly sensitive data (transactions, portfolios, trading strategies) that cannot be sent to external APIs. A lightweight, locally-runnable financial LM addresses this critical privacy concern while maintaining performance.

### 2. **Data Efficiency in Specialized Domains**
Training from scratch requires massive compute. Understanding how to leverage both domain-specific financial data and high-quality general corpora (e.g., WikiText) can dramatically reduce training costs while improving domain adaptation.

### 3. **Cross-Domain Knowledge Transfer Mystery**
Current literature lacks systematic studies on how high-quality out-of-domain data (Wikipedia, curated web text) affects in-domain performance. Does WikiText help financial tasks? Or does it dilute domain specialization?

### 4. **Mixture Composition Optimization**
When combining datasets, what are the optimal mixture rates? Do different model sizes (0.6B vs 4B) require different data strategies? This thesis provides empirical answers.

### 5. **Lightweight Model Viability**
Can small models (0.6B-4B parameters) achieve acceptable financial NLP performance when pretrained on the right data mixtures? This is crucial for edge deployment (laptops, mobile devices).

### 6. **Reverse Scaling Phenomenon and Training Dynamics**
Initial experiments reveal surprising patterns: sometimes smaller models outperform larger ones on specific data regimes (e.g., 0.6B on WikiText, 1.7B on Twitter). Understanding *why* this happens—and how to fix it—is scientifically valuable and practically important for model training at scale.

## Research Approach

### Core Experimental Design
- **Model Family**: Qwen3 (0.6B, 1.7B, 4B) - three scales to study size-dependent effects
- **In-Domain Data**: 7 financial datasets (207M tokens total)
  1. `virattt/financial-qa-10K` - Financial Q&A (7.1K samples, ~3.5M tokens)
  2. `FinGPT/fingpt-sentiment-train` - Financial sentiment (76.8K samples, ~19.1M tokens)
  3. `gbharti/finance-alpaca` - Financial instruction following (68.9K samples, ~17.2M tokens)
  4. `LLukas22/fiqa` - FiQA question answering (17.4K samples, ~4.3M tokens)
  5. `zeroshot/twitter-financial-news-sentiment` - Twitter financial sentiment (1.1K samples, ~0.3M tokens)
  6. `JanosAudran/financial-reports-sec` - SEC financial reports (54.3K samples, ~80M tokens)
  7. `Lettria/financial_news_articles` - Financial news articles (300K samples, ~197M tokens)
- **Out-of-Domain Data**: `wikitext` - High-quality curated Wikipedia text (103K samples, ~100M tokens estimated)
- **Mixture Strategies**:
  - Pure financial pretraining (individual datasets)
  - Pure WikiText pretraining (baseline for general knowledge)
  - Mixed financial datasets (50cap strategy: caps largest dataset at 50%)
  - Mixed WikiText + financial datasets (balanced 8-dataset mixture)

### Key Investigations
1. **Single-Domain Baselines**: How do models pretrained purely on financial or general text perform?
2. **In-Domain Diversity**: Does mixing multiple financial datasets improve robustness vs single-dataset overfitting?
3. **Cross-Domain Synergy**: Do WikiText+Financial mixtures outperform pure financial training?
4. **Scale Dependency**: Do larger models benefit more/less from data mixing?
5. **Generalization Analysis**: Which training regimes produce models that generalize best across unseen financial tasks?

## Key Hypotheses

### H1: Domain Diversity Benefit
**Hypothesis**: Pretraining on mixed financial datasets will outperform single-dataset training by improving cross-task robustness.

**Rationale**: Financial texts vary widely (news articles, Q&A, sentiment tweets, formal reports). Exposure to diverse formats prevents overfitting to dataset-specific artifacts.

### H2: High-Quality General Data Complementarity
**Hypothesis**: Adding WikiText to financial mixtures will improve model stability and linguistic quality without harming domain performance.

**Rationale**: WikiText provides clean, grammatically correct, encyclopedic knowledge that can serve as a "regularizer" against noisy financial data, while improving general language understanding.

### H3: Scale-Dependent Training Dynamics
**Hypothesis**: Larger models (4B) require more careful hyperparameter tuning (especially learning rate) than smaller models (0.6B), but when properly tuned, will show better performance and lower variance across data mixtures.

**Rationale**: Larger models have greater capacity to learn multiple data distributions simultaneously, but also have different optimizer dynamics that require adjusted hyperparameters. Initial training instabilities or "reverse scaling" may occur with naive hyperparameter transfer from smaller models.

### H4: Evaluation-Training Domain Mismatch
**Hypothesis**: Models trained on WikiText will show poor financial task performance due to domain shift, demonstrating the necessity of domain-specific pretraining.

**Rationale**: General text lacks financial terminology, reasoning patterns, and discourse structures, making pure WikiText pretraining insufficient for financial NLP.

## Evaluation Framework

### Multi-Dataset Evaluation
All pretrained models are evaluated on **8 held-out test sets** spanning:
- Financial sentiment (Twitter, FinGPT, FiQA)
- Financial Q&A (Alpaca, Financial QA)
- Financial documents (SEC Reports, Financial News)
- General text (WikiText)

### Metrics
1. **Cross-Entropy Loss**: Primary metric for language modeling quality
2. **Perplexity**: Interpretable measure of prediction confidence
3. **Cross-Dataset Variance**: Measures model robustness (low variance = better generalization)
4. **Relative Spread**: Coefficient of variation across evaluation sets

## Expected Contributions

### 1. Empirical Data Mixture Guidelines
Concrete recommendations for financial LM pretraining: optimal mixture rates, dataset selection, scale-dependent strategies. Evidence that in-domain diversity outweighs high-quality general corpora for specialized domains.

### 2. Learning Rate Scaling Laws for 0.6B-4B Models
Discovery that larger models require 50-85% learning rate reduction compared to smaller models to avoid training instabilities and reverse scaling. Demonstrated that proper hyperparameter scaling resolves apparent performance regressions, with systematic guidelines for LR adjustment by model size.

### 3. Dataset Size Effects on Pretraining
Empirical relationship between dataset size and overtraining patterns: datasets < 20K samples show extreme overtraining (67-249 epochs) and require mixing, while large datasets (> 100K samples) provide stable pretraining with minimal epochs (2-30).

### 4. Cross-Domain Interaction Analysis
First systematic study of how high-quality general corpora (WikiText) interact with domain-specific financial data during pretraining. Evidence that WikiText provides minimal benefit and sometimes degrades financial task performance.

### 5. Lightweight Financial Model Feasibility
Demonstration that 0.6B-4B models can achieve practical financial NLP performance with appropriate data mixtures and hyperparameter tuning, enabling privacy-preserving edge deployment.

### 6. Open-Source Training Pipeline
Reproducible codebase for mixture-based pretraining with comprehensive evaluation framework across 10 experiments and 30 trained models.

## Thesis Structure (30 Pages Total)

### Chapter 1: Introduction (3 pages)
- **1.1 Motivation** (1 page): Privacy-preserving financial AI, lightweight deployment needs, training efficiency
- **1.2 Research Questions** (0.5 page): Core questions about data mixture effects and model scaling
- **1.3 Contributions** (0.75 page):
  - Data mixture guidelines (in-domain diversity > general quality)
  - Learning rate scaling laws for 0.6B-4B models
  - Dataset size effects and overtraining patterns
  - Cross-domain interaction evidence
- **1.4 Thesis Organization** (0.5 page): Chapter overview
- **1.5 Scope and Limitations** (0.25 page): What this thesis covers and excludes

### Chapter 2: Background and Related Work (5 pages)
- **2.1 Financial NLP** (1 page): Task landscape, existing models, domain challenges
- **2.2 Language Model Pretraining** (1.5 pages): Objectives, scaling laws, architectural considerations
- **2.3 Data Mixture Strategies** (1.5 pages): Curriculum learning, domain mixing, related empirical studies
- **2.4 Domain Adaptation and Transfer Learning** (1 page): Cross-domain challenges, catastrophic forgetting, distribution shift

### Chapter 3: Methodology (6 pages)
- **3.1 Experimental Design Overview** (1 page): High-level approach, research framework, 10 experiments
- **3.2 Model Architecture** (0.75 page): Qwen3 family (0.6B/1.7B/4B), architectural details
- **3.3 Datasets** (2 pages):
  - 3.3.1 Financial datasets: descriptions, statistics, preprocessing (7 datasets, 207M tokens)
  - 3.3.2 WikiText: characteristics, role as high-quality general corpus
  - 3.3.3 Mixture strategies: 50cap algorithm, sampling procedures
- **3.4 Training Setup and Hyperparameter Tuning** (1.25 pages):
  - 3.4.1 Initial configuration: uniform LR = 2e-5, hardware, optimizer
  - 3.4.2 Discovery of reverse scaling in WikiText, Financial QA, Twitter experiments
  - 3.4.3 Systematic LR adjustment experiments (1e-5, 5e-6, 3e-6)
  - 3.4.4 Final LR recommendations by model size
  - 3.4.5 Other hyperparameters (batch size, warmup, epochs)
- **3.5 Evaluation Protocol** (1 page): Metrics (loss, perplexity, variance), multi-dataset evaluation strategy

### Chapter 4: Results (10 pages)
- **4.1 Overview of Experimental Results** (0.5 page):
  - Summary table: all 10 experiments (30 trained models, 240 evaluations)
  - Reading guide and key metrics explanation

- **4.2 Data Mixture Effects: The Core Finding** (3 pages):
  - 4.2.1 Mixed Financial datasets (best overall: 21.55 ppl @ 4B, 55% variance)
    - Performance across 8 evaluation sets
    - Generalization analysis and robustness
  - 4.2.2 Mixed Wiki+Financial (marginal benefit: 26.69 ppl @ 4B, 62% variance)
    - Comparison with pure financial mixture
    - WikiText inclusion trade-offs
  - 4.2.3 Pure WikiText baseline (poor financial transfer: 31.54 ppl @ 4B)
    - Domain mismatch evidence
    - Strong general performance but weak financial transfer
  - 4.2.4 Key takeaway: In-domain diversity > general corpus quality
    - Summary comparison table

- **4.3 Individual Dataset Analysis: Component Effects** (2 pages):
  - 4.3.1 Large datasets (News 197M, SEC 80M): 2-24 epochs, good generalization
    - Standalone viability
    - Transfer patterns (news ↔ SEC reports)
  - 4.3.2 Medium datasets (FinGPT 19M, Alpaca 17M, FiQA 4M): 6-30 epochs
    - Moderate overtraining
    - Task-specific strengths (instruction-following, Q&A)
  - 4.3.3 Small datasets (Financial QA 3.5M, Twitter 0.3M): 67-249 epochs
    - Extreme overtraining evidence
    - Why small datasets require mixing
  - 4.3.4 Dataset size vs generalization relationship
    - Empirical pattern: size correlates with robustness

- **4.4 Training Dynamics and Scaling Behavior** (2.5 pages):
  - 4.4.1 Normal scaling pattern (FiQA, FinGPT, News, SEC, Alpaca)
    - Expected improvements: 0.6B → 1.7B → 4B
    - Performance gains: 50-90% perplexity reduction
  - 4.4.2 Reverse scaling phenomenon (3 cases documented)
    - WikiText: 0.6B (9.68) < 4B (31.54) << 1.7B (infinity)
    - Financial QA: 1.7B (8.42) < 0.6B (9.69) < 4B (9.02)
    - Twitter: 1.7B (12.55) < 0.6B (16.28) < 4B (18.05)
  - 4.4.3 **Learning rate sensitivity by model size** (MAJOR FINDING)
    - Empirical LR scaling: 0.6B (2e-5) → 1.7B (1e-5) → 4B (5e-6)
    - Before/after comparisons showing performance recovery
    - Evidence: Financial QA 4B: 9.02 → 8.09 ppl (-10.3%)
    - Evidence: Twitter 4B: 18.05 → 12.35 ppl (-31.6%)
  - 4.4.4 Fixing reverse scaling: systematic LR adjustment guidelines
    - 1.7B models: 50% LR reduction
    - 4B models: 75-85% LR reduction
  - 4.4.5 Model stability analysis
    - Variance trends across experiments
    - Mixture vs individual training stability

- **4.5 Domain Transfer and Generalization Patterns** (1.5 pages):
  - 4.5.1 Cross-dataset evaluation analysis
    - Best generalizers: mixed financial, large individual datasets
    - Worst generalizers: small datasets, WikiText
  - 4.5.2 Document format and task type effects
    - Long-form transfer: news ↔ SEC reports (strong)
    - Instruction-following: Alpaca ↔ FiQA (moderate)
    - Short-form isolation: Twitter (weak transfer)
  - 4.5.3 Variance comparison across training regimes
    - Mixed financial: 55-62% relative spread
    - Individual large: 18-26% on own dataset, 65-90% cross-dataset
    - Individual small: 70-97% relative spread
  - 4.5.4 Domain-specific vs general knowledge transfer
    - WikiText → financial: poor (vocabulary mismatch)
    - Financial → WikiText: moderate (general language intact)

- **4.6 Summary and Key Results** (0.5 page):
  - Best configurations by use case table
  - Ranking: Mixed Financial > Large Individual > Mixed Wiki+Financial > Small Individual > WikiText
  - Critical findings recap

### Chapter 5: Discussion (4 pages)
- **5.1 Key Empirical Findings** (1 page):
  - Mixed financial datasets outperform single-domain training (21.55 vs 31.54 ppl)
  - WikiText provides minimal benefit for financial tasks (sometimes degrades performance)
  - **Learning rate must scale down 50-85% with model size (0.6B→4B)**
  - Dataset size critically affects pretraining viability (< 20K samples problematic)
  - Reverse scaling is a training artifact, not fundamental limitation

- **5.2 Interpretation of Data Interaction Effects** (1.5 pages):
  - 5.2.1 Why WikiText underperforms on financial tasks
    - Vocabulary mismatch (financial terminology absent)
    - Discourse pattern differences (encyclopedic vs financial analysis)
    - Evidence: 31.54 ppl vs 21.55 ppl for mixed financial
  - 5.2.2 Benefits of in-domain diversity
    - Robustness across task types (sentiment, Q&A, documents)
    - Generalization improvement: 55% variance vs 70-97% individual
    - Dataset complementarity effects
  - 5.2.3 Domain interference patterns
    - When cross-domain mixing hurts: adding WikiText to financial
    - Small dataset overfitting (249 epochs on Financial QA)
  - 5.2.4 **Scale-dependent training dynamics** (0.75 page)
    - Why larger models need smaller learning rates
      - Gradient magnitude scaling with parameter count
      - Optimizer momentum accumulation effects
      - Relationship to batch size and effective learning rate
    - Empirical LR scaling law: LR ∝ 1/√(model_size)
    - Implications for scaling laws in literature
      - Most assume proper hyperparameter tuning
      - Naive hyperparameter transfer causes "reverse scaling"
    - Connection to existing work on LR scheduling and warmup

- **5.3 Practical Guidelines for Financial LM Pretraining** (1 page):
  - **Data mixture strategies by use case**:
    - General financial: Use mixed 7-dataset approach (50cap)
    - Domain-specific: Use large individual dataset (> 100K samples)
    - Avoid: WikiText mixing, small individual datasets (< 20K samples)
  - **Model size selection**:
    - 0.6B: Fast, good for prototyping, high variance
    - 1.7B: Best balance, recommended for most use cases
    - 4B: Best performance but requires careful LR tuning
  - **Learning rate guidelines by model size**:
    - 0.6B: LR = 2e-5 (baseline)
    - 1.7B: LR = 1e-5 (50% reduction)
    - 4B: LR = 5e-6 (75% reduction)
    - Scale by √(model_size) ratio for other sizes
  - **Token budget allocation**:
    - Prefer diverse mixtures over deep single-dataset training
    - 100M tokens sufficient with proper mixing
    - Cap dominant datasets at 50% to ensure diversity

- **5.4 Limitations and Threats to Validity** (0.5 page):
  - Single model family (Qwen3) limits generalizability
    - LR scaling may differ for other architectures
    - Mixture benefits may be architecture-dependent
  - Fixed mixture strategy (50cap) not exhaustively tested
    - Other mixture algorithms unexplored
    - Dynamic mixing strategies future work
  - Evaluation on pretraining distributions only
    - No downstream task evaluation (sentiment, QA)
    - Perplexity may not reflect task performance
  - Hardware constraints limited exploration
    - Only tested up to 4B parameters
    - Larger models (7B+) may show different patterns

### Chapter 6: Conclusion (2 pages)
- **6.1 Summary of Contributions** (1 page):
  - **Data mixture guidelines**: Comprehensive empirical study (10 experiments, 30 models, 240 evaluations) demonstrating that in-domain diversity outweighs high-quality general corpora for specialized domains
    - Mixed financial: 21.55 ppl (best)
    - Pure WikiText: 31.54 ppl (poor financial transfer)
  - **Learning rate scaling laws**: Discovered empirical relationship between model size and optimal LR for 0.6B-4B models
    - 0.6B: 2e-5, 1.7B: 1e-5 (50% ↓), 4B: 5e-6 (75% ↓)
    - Resolved reverse scaling in 3 experiments
    - Generalizable beyond financial domain
  - **Dataset size effects**: Established relationship between dataset size and pretraining viability
    - Small (< 20K samples): extreme overtraining, requires mixing
    - Large (> 100K samples): viable standalone, 2-30 epochs
  - **Domain transfer patterns**: First systematic analysis of financial NLP cross-dataset generalization with evidence of format/task-specific transfer
  - **Open-source pipeline**: Reproducible codebase with comprehensive evaluation framework

- **6.2 Implications for Practice and Research** (0.5 page):
  - **For practitioners**:
    - Use diverse in-domain mixtures over generic corpora
    - Scale learning rate down 50-85% for larger models (1.7B-4B)
    - Avoid small datasets (< 20K samples) for individual pretraining
  - **For researchers**:
    - Hyperparameter scaling critical for valid model comparisons
    - Data composition matters more than data quality alone
    - Reverse scaling often indicates training issues, not model limitations
  - **For industry**:
    - Viable path to privacy-preserving financial AI
    - 0.6B-4B models sufficient with proper data/training
    - Edge deployment feasible with careful optimization

- **6.3 Future Research Directions** (0.5 page):
  - **Scaling up**: Extend LR scaling laws to 7B+ models, test other architectures (LLaMA, Gemma)
  - **Mixture optimization**: Ablate mixture ratios, explore dynamic mixing (curriculum learning)
  - **Downstream evaluation**: Test on financial NLP tasks (sentiment, QA, summarization, NER)
  - **Multi-stage pretraining**: Investigate general → domain adaptation pipelines
  - **LR theory**: Develop theoretical understanding of empirical LR scaling relationship

### Appendices (Optional, not counted in 30 pages)
- **Appendix A**: Full hyperparameter tables
- **Appendix B**: Additional experimental results
- **Appendix C**: Dataset statistics and preprocessing details
- **Appendix D**: Code repository structure and usage

---

**Page Allocation Summary:**
- Introduction: 3 pages (10%) - includes all 6 contributions
- Background: 5 pages (17%) - context on pretraining, mixtures, scaling
- Methodology: 6 pages (20%) - includes iterative LR discovery process
- Results: 10 pages (33%) - **restructured thematically** ← *Core contribution*
  - Data mixtures (3 pages)
  - Individual datasets (2 pages)
  - **Training dynamics & LR scaling (2.5 pages)** ← *Elevated*
  - Domain transfer (1.5 pages)
  - Summary (0.5 pages)
- Discussion: 4 pages (13%) - enhanced with LR scaling theory & guidelines
- Conclusion: 2 pages (7%) - comprehensive contributions summary

**Total: 30 pages**

**Key Structural Changes:**
- Chapter 4 reorganized from linear (by experiment) → thematic (by insight)
- Training dynamics/LR scaling elevated from 0.5 page → 2.5 pages
- Learning rate discovery process added to methodology
- LR scaling added as core contribution throughout

## Timeline and Current Status

✅ **Experiments Completed** (All training concluded):

**Infrastructure:**
- ✅ Full experimental pipeline with multi-dataset evaluation framework
- ✅ 8 held-out test sets for cross-dataset generalization analysis
- ✅ Automated logging and checkpointing system

**Mixture Experiments (3 experiments × 3 model sizes = 9 models):**
- ✅ Mixed Financial (7 datasets, 207M tokens) - Best overall: 21.55 ppl @ 4B
- ✅ Mixed Wiki+Financial (8 datasets) - Marginal benefit: 26.69 ppl @ 4B
- ✅ Pure WikiText (general baseline) - Poor financial transfer: 31.54 ppl @ 4B

**Individual Dataset Experiments (7 experiments × 3 model sizes = 21 models):**
- ✅ Large datasets: News Articles (197M), SEC Reports (80M) - 2-24 epochs
- ✅ Medium datasets: FinGPT (19M), Alpaca (17M), FiQA (4M) - 6-30 epochs
- ✅ Small datasets: Financial QA (3.5M), Twitter (0.3M) - 67-249 epochs

**Learning Rate Adjustment Experiments:**
- ✅ Discovered reverse scaling in 3 experiments (WikiText, Financial QA, Twitter)
- ✅ Systematic LR tuning: tested 1e-5, 5e-6, 3e-6 for 1.7B and 4B models
- ✅ Validated LR scaling law: 0.6B (2e-5) → 1.7B (1e-5) → 4B (5e-6)
- ✅ Performance recovery documented (e.g., Twitter 4B: 18.05 → 12.35 ppl)

**Total: 10 experiments, 30 trained models, 240 evaluation results**
**Results documentation:** All metrics in `experimental_results/` directory (10 files)

🔄 **Current Phase: Writing and Analysis**:

**Quantitative Analysis:**
- 📊 Perplexity trends across all 10 experiments and 3 model sizes
- 📈 Variance metrics: relative spread analysis (55% mixed vs 70-97% individual)
- 🔬 Dataset size correlation with overtraining patterns
- 🎯 LR scaling relationship validation (empirical law: LR ∝ 1/√size)

**Visualization and Tables:**
- Creating summary tables for all 10 experiments (30 models, 240 evaluations)
- Performance comparison plots: mixture vs individual vs WikiText
- LR sensitivity plots: before/after adjustment comparisons
- Dataset size vs epochs/generalization scatter plots
- Variance comparison across training regimes

**Chapter Writing:**
- **Chapter 4 (Results)**: Organizing 10 experiments thematically
  - Data mixture effects (core finding)
  - Individual dataset analysis (size effects)
  - Training dynamics & LR scaling (major discovery)
  - Domain transfer patterns
- **Chapter 5 (Discussion)**: Interpreting findings
  - Why WikiText fails for finance
  - LR scaling theory and implications
  - Practical guidelines for practitioners
- **Chapters 1-3, 6**: Introduction, methodology, conclusion

**No additional experiments or statistical testing planned** - focus is entirely on writing, visualization, and interpretation of completed results.

---

## Summary of Architectural Refinements

This thesis architecture has been refined based on comprehensive analysis of all 10 completed experiments. The key refinements elevate previously underemphasized findings to their proper prominence:

### Major Discoveries Elevated:

1. **Learning Rate Scaling Laws** (Now a core contribution #2)
   - Empirical finding: LR must scale down 50-85% as model size increases 0.6B→4B
   - Resolved "reverse scaling" in 3 experiments through systematic LR tuning
   - Generalizable beyond financial domain to any 0.6B-4B model training

2. **Dataset Size Effects** (Now a core contribution #3)
   - Clear relationship: datasets < 20K samples show extreme overtraining (67-249 epochs)
   - Large datasets (> 100K samples) viable standalone with 2-30 epochs
   - Critical insight: small datasets REQUIRE mixing for effective pretraining

3. **Training Dynamics Section** (Expanded from 0.5 → 2.5 pages in Chapter 4)
   - Documented 3 cases of reverse scaling
   - Systematic LR adjustment experiments and results
   - Before/after performance comparisons
   - Established empirical LR scaling guidelines

### Structural Improvements:

1. **Chapter 4 Reorganization**: From linear (by experiment type) to thematic (by scientific insight)
   - Better narrative flow: mixtures → individuals → training dynamics → transfer
   - Emphasizes "what we learned" over "what we did"

2. **Enhanced Methodology**: Documents iterative scientific process
   - Shows discovery of reverse scaling → investigation → solution
   - Adds credibility through intellectual honesty

3. **Strengthened Discussion**: Deep dive into LR scaling implications
   - Connects to optimizer dynamics and existing literature
   - Provides practical guidelines for practitioners
   - Discusses implications for scaling laws in general

### Result: A Richer Thesis

The refined architecture transforms the thesis from:
- "Data mixing works for financial NLP"

To:
- "Comprehensive study of data mixture effects and training dynamics for 0.6B-4B models, with generalizable findings on LR scaling, dataset size effects, and domain transfer patterns"

This reflects the true depth and breadth of the experimental work completed.