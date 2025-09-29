# Thesis Overview: Information-Secure Language Models for Finance Applications

**Author:** Guanlan Liu
**Institution:** University of Zurich
**Target Length:** 30 pages
**Focus:** Lightweight, privacy-preserving LLMs for financial applications

---

## Thesis Structure and Page Allocation

### Title Page, Abstract, Contents (2 pages)
- Title page with institutional information
- Abstract (200-300 words)
- Table of contents

### Chapter 1: Introduction (3 pages)
**Objective:** Establish motivation and research questions

#### 1.1 Motivation (1 page)
- Privacy concerns in financial data processing
- Need for on-device financial AI capabilities
- Current limitations of cloud-based solutions
- Regulatory requirements (GDPR, financial data protection)

#### 1.2 Problem Statement (0.5 page)
- Trade-off between model capability and resource constraints
- Domain-specific knowledge requirements for finance
- Lack of understanding of dataset interactions

#### 1.3 Research Questions (0.5 page)
1. How do different financial datasets contribute to model performance?
2. Does general knowledge (WikiText) enhance financial understanding?
3. What are optimal mixture strategies for financial pretraining?
4. Can lightweight models (0.6B-4B) achieve practical financial capabilities?

#### 1.4 Contributions (0.5 page)
- Systematic evaluation of financial dataset interactions
- Novel mixture strategies for financial domain pretraining
- Empirical evidence for cross-domain transfer effects
- Open-source framework for financial LLM training

#### 1.5 Thesis Structure (0.5 page)
- Overview of remaining chapters

### Chapter 2: Background and Related Work (4 pages)

#### 2.1 Language Models in Finance (1 page)
- Evolution: BERT → GPT → Specialized models
- FinBERT, BloombergGPT, FinGPT
- Current limitations and gaps

#### 2.2 Privacy-Preserving ML (1 page)
- On-device inference requirements
- Edge computing for sensitive data
- Quantization and model compression techniques

#### 2.3 Multi-Dataset Training (1 page)
- Dataset mixing strategies (T5, mT5, ExT5)
- Square root scaling and temperature sampling
- Transfer learning in NLP

#### 2.4 Efficient Fine-tuning Methods (1 page)
- LoRA and parameter-efficient methods
- Sequence packing for training efficiency
- Flash Attention and optimization techniques

### Chapter 3: Methodology (6 pages)

#### 3.1 Experimental Framework (1 page)
- Overview of training pipeline
- Hardware setup (RTX 4090, M1 Max comparisons)
- Evaluation methodology

#### 3.2 Datasets (2 pages)
##### Financial Datasets (1 page)
- **Financial Q&A** (7K examples, 0.7M tokens)
- **FinGPT Sentiment** (76K examples, 4.1M tokens)
- **Finance Alpaca** (68K examples, 8.5M tokens)
- **FiQA** (15K examples, 3.6M tokens)
- **Twitter Financial** (10K examples, 0.3M tokens)
- **SEC Reports** (200K examples, 8.1M tokens)
- **News Articles** (306K examples, 197.4M tokens)

##### General Knowledge Dataset (0.5 page)
- **WikiText-103** (1.8M examples, 103M tokens)
- Rationale for inclusion

##### Dataset Characteristics Analysis (0.5 page)
- Token distributions
- Content diversity
- Domain coverage

#### 3.3 Mixture Strategies (2 pages)
##### 3.3.1 Mathematical Formulation (0.5 page)
- Square root scaling: $w_i = \frac{\sqrt{t_i}}{\sum_j \sqrt{t_j}}$
- 50% capping rule and redistribution

##### 3.3.2 Mixture Configurations (0.5 page)
- **Mixed**: 7 financial datasets
- **Mixed-Wiki**: 7 financial + WikiText
- Detailed mixture rates and justification

##### 3.3.3 Implementation Details (1 page)
- Batch construction and sampling
- Sequence packing optimization
- Training hyperparameters

#### 3.4 Model Architecture (1 page)
- Qwen3 family (0.6B, 1.7B, 4B parameters)
- LoRA configuration (rank=32, alpha=64)
- Attention mechanisms (Flash Attention 2 vs Eager)

### Chapter 4: Experiments (8 pages)

#### 4.1 Experimental Setup (1 page)
- Training configurations
- Compute resources and time
- Evaluation metrics (perplexity, loss, spread)

#### 4.2 Single Dataset Baselines (2 pages)
- Individual dataset performance
- Learning curves and convergence
- Dataset-specific insights

#### 4.3 Mixed Dataset Experiments (2.5 pages)
##### 4.3.1 Mixed Financial (7 datasets)
- Performance across model sizes
- Cross-dataset evaluation results
- Dataset interaction effects

##### 4.3.2 Mixed-Wiki (8 datasets)
- Impact of adding general knowledge
- Comparative analysis with Mixed

#### 4.4 Cross-Dataset Transfer Analysis (2 pages)
- Transfer matrices (train on A, eval on B)
- Complementary vs redundant information
- Domain gap measurements

#### 4.5 Efficiency Analysis (0.5 page)
- Training time comparisons
- Memory usage
- Inference speed

### Chapter 5: Results and Analysis (5 pages)

#### 5.1 Key Findings (2 pages)
##### Dataset Interactions
- Synergistic effects between datasets
- Optimal mixture rates discovered
- WikiText contribution to financial understanding

##### Model Size Effects
- Performance scaling with parameters
- Efficiency trade-offs

#### 5.2 Statistical Analysis (1.5 pages)
- Significance tests
- Confidence intervals
- Correlation analysis between datasets

#### 5.3 Qualitative Analysis (1.5 pages)
- Case studies of specific improvements
- Error analysis
- Interpretation of transfer patterns

### Chapter 6: Discussion (3 pages)

#### 6.1 Implications for Financial AI (1 page)
- Practical deployment considerations
- Privacy benefits quantified
- Cost-effectiveness analysis

#### 6.2 Limitations (1 page)
- Pretraining metrics as proxies
- Dataset selection constraints
- Hardware limitations

#### 6.3 Comparison with Related Work (1 page)
- Performance vs BloombergGPT, FinGPT
- Efficiency advantages
- Novel contributions

### Chapter 7: Conclusion and Future Work (2 pages)

#### 7.1 Summary of Contributions (0.75 page)
- Answered research questions
- Key insights on dataset mixing
- Practical framework delivered

#### 7.2 Future Directions (0.75 page)
- Task-specific fine-tuning validation
- Larger model exploration
- Additional financial domains

#### 7.3 Closing Remarks (0.5 page)
- Vision for privacy-preserving financial AI
- Call for open financial datasets

### References (2 pages)
- ~40-50 references
- Academic papers, technical reports

### Appendices (3 pages)

#### Appendix A: Detailed Experimental Configurations (1 page)
- Hyperparameter tables
- Hardware specifications

#### Appendix B: Additional Results (1 page)
- Extended evaluation tables
- Supplementary figures

#### Appendix C: Code and Reproducibility (1 page)
- GitHub repository structure
- Reproduction instructions
- Computational requirements

---

## Total Page Distribution

| Section | Pages |
|---------|-------|
| Front Matter | 2 |
| Chapter 1: Introduction | 3 |
| Chapter 2: Background | 4 |
| Chapter 3: Methodology | 6 |
| Chapter 4: Experiments | 8 |
| Chapter 5: Results | 5 |
| Chapter 6: Discussion | 3 |
| Chapter 7: Conclusion | 2 |
| References | 2 |
| Appendices | 3 |
| **Total** | **30** |

---

## Key Figures and Tables to Include

### Essential Figures (8-10 total)
1. System architecture diagram
2. Dataset size and composition chart
3. Mixture rate visualization
4. Learning curves comparison
5. Cross-dataset transfer heatmap
6. Perplexity across configurations
7. Model size vs performance plot
8. Training efficiency comparison

### Essential Tables (6-8 total)
1. Dataset statistics summary
2. Model configurations
3. Mixture rates for all strategies
4. Main results table (perplexity)
5. Cross-dataset evaluation matrix
6. Computational requirements
7. Comparison with baselines

---

## Writing Timeline Suggestion

### Week 1-2: Data Collection and Analysis
- Complete all experiments
- Generate all figures and tables
- Statistical analysis

### Week 3: Core Chapters
- Methodology (Chapter 3)
- Experiments (Chapter 4)
- Results (Chapter 5)

### Week 4: Context Chapters
- Introduction (Chapter 1)
- Background (Chapter 2)
- Discussion (Chapter 6)

### Week 5: Polish and Submission
- Conclusion (Chapter 7)
- Abstract
- References
- Final proofreading

---

## Key Messages to Emphasize

1. **Privacy First**: On-device processing eliminates data leakage
2. **Efficiency Matters**: 0.6B-4B models are practical for edge deployment
3. **Smart Mixing**: Strategic dataset combination outperforms naive approaches
4. **Domain Transfer**: General knowledge enhances financial understanding
5. **Open Science**: Reproducible framework for financial AI research