# Thesis Writing Guidelines

## Writing Style and Tone

### Academic Voice
- **Use passive voice sparingly:** "We conduct experiments" > "Experiments were conducted"
- **Be precise:** "The model achieves 15.3% improvement" > "The model performs better"
- **Avoid hyperbole:** "demonstrates improvement" > "revolutionizes"
- **Stay objective:** Present findings before interpretation

### Technical Writing
- **Define acronyms on first use:** "Large Language Models (LLMs)"
- **Be consistent:** Choose "pretrain" or "pre-train" and stick with it
- **Explain technical terms:** Don't assume reader knows LoRA, Flash Attention
- **Use examples:** Abstract concepts benefit from concrete examples

---

## Page Budget Management

### Strategies for 30-Page Limit

#### What to Prioritize
1. **Novel contributions** (mixture strategies, cross-dataset analysis)
2. **Empirical results** with statistical validation
3. **Clear methodology** for reproducibility
4. **Practical implications** for real-world deployment

#### What to Minimize
1. **Extensive literature review** - Focus on directly relevant work
2. **Mathematical derivations** - Move to appendix if needed
3. **Redundant explanations** - State once clearly
4. **Excessive background** - Assume MS-level knowledge

#### Space-Saving Techniques
- **Combine related concepts:** Merge similar subsections
- **Use tables efficiently:** Multiple results in one table
- **Concise figure captions:** Essential info only
- **Footnotes for details:** Non-critical clarifications
- **Appendices strategically:** Detailed configs, extended results

---

## Figure and Table Guidelines

### Figures (8-10 total)
```
Essential Figures:
□ System architecture diagram (1 page)
□ Dataset composition pie chart (0.5 page)
□ Mixture rate comparison bar chart (0.5 page)
□ Learning curves (combined plot) (1 page)
□ Transfer matrix heatmap (1 page)
□ Model size vs performance (0.5 page)
```

### Tables (6-8 total)
```
Essential Tables:
□ Dataset statistics (0.5 page)
□ Experimental configurations (0.5 page)
□ Main results - perplexity (1 page)
□ Statistical significance (0.5 page)
□ Computational requirements (0.5 page)
```

### Design Principles
- **Information density:** Maximize data per figure/table
- **Self-contained:** Understandable without text
- **Consistent style:** Same fonts, colors, formatting
- **Publication quality:** Vector graphics, high DPI

---

## Section-Specific Guidelines

### Abstract (200-300 words)
**Structure:**
1. Problem statement (1-2 sentences)
2. Approach (2-3 sentences)
3. Key findings (3-4 sentences)
4. Implications (1-2 sentences)

**Example Opening:**
"Financial institutions require on-device language models to process sensitive data while maintaining privacy. We investigate how different financial datasets interact during pretraining and whether general knowledge enhances domain-specific performance."

### Introduction (3 pages)
**Page 1:** Hook → Problem → Why it matters
**Page 2:** Research questions → Contributions
**Page 3:** Thesis structure → Reader's guide

**Strong Opening Example:**
"Processing financial data in the cloud poses significant privacy risks, with 73% of financial institutions reporting data security as their primary AI adoption concern [citation]."

### Methodology (6 pages)
**Balance:**
- 1/3 on framework and approach
- 1/3 on datasets and mixing strategies
- 1/3 on implementation details

**Key Principle:** Enough detail for reproduction, but not a tutorial

### Results (5 pages)
**Organization:**
- Lead with most important finding
- Support with statistics
- Show, then tell
- Acknowledge limitations

**Result Presentation Template:**
"The mixed configuration achieves X±Y% improvement (p<0.05) over the baseline, with particularly strong gains in [specific area]."

---

## Common Pitfalls to Avoid

### Content Pitfalls
❌ **Over-explaining basics:** Don't define what a neural network is
❌ **Under-explaining novel:** Thoroughly explain your mixture strategy
❌ **Speculation without evidence:** Every claim needs support
❌ **Ignoring negative results:** Discuss what didn't work too

### Structure Pitfalls
❌ **Imbalanced chapters:** Keep proportions reasonable
❌ **Poor transitions:** Each section should flow to the next
❌ **Redundant content:** Say it once, say it well
❌ **Missing signposting:** Tell reader where they are

### Style Pitfalls
❌ **Inconsistent terminology:** Pick terms and stick with them
❌ **Excessive jargon:** Explain technical terms
❌ **Informal language:** Avoid contractions, colloquialisms
❌ **Dense paragraphs:** Break up text, use subsections

---

## Revision Checklist

### First Draft
- [ ] All experiments complete
- [ ] Results tables/figures ready
- [ ] Rough text for all sections
- [ ] Citations in place

### Second Draft
- [ ] Logical flow between sections
- [ ] Consistent terminology
- [ ] All claims supported
- [ ] Figures/tables referenced

### Final Draft
- [ ] Under 30 pages
- [ ] Grammar/spelling checked
- [ ] Format requirements met
- [ ] References complete

---

## Time Management

### Week-by-Week Plan

**Week 1: Analysis**
- Complete all experiments
- Generate all plots
- Statistical analysis
- Outline each chapter

**Week 2: Core Writing**
- Monday: Methodology
- Tuesday: Experiments
- Wednesday: Results
- Thursday: Analysis
- Friday: Revise core

**Week 3: Context Writing**
- Monday: Introduction
- Tuesday: Background
- Wednesday: Discussion
- Thursday: Conclusion
- Friday: Abstract

**Week 4: Polish**
- Monday-Tuesday: Full revision
- Wednesday: Format check
- Thursday: Final proofread
- Friday: Submit

---

## Key Messages by Chapter

### Chapter 1 (Introduction)
"Privacy-preserving financial AI is not just desirable—it's essential"

### Chapter 3 (Methodology)
"Strategic dataset mixing outperforms naive concatenation"

### Chapter 4 (Experiments)
"Systematic evaluation reveals surprising dataset interactions"

### Chapter 5 (Results)
"Small models + smart training = practical deployment"

### Chapter 6 (Discussion)
"This work enables real-world private financial AI"

### Chapter 7 (Conclusion)
"The path forward: open datasets, efficient models, privacy first"

---

## Final Tips

1. **Write the abstract last** - It's easier when you know your results
2. **Start with results** - Build the story around your findings
3. **Use version control** - Git for LaTeX files too
4. **Get feedback early** - Share drafts with advisor
5. **Leave time for formatting** - LaTeX issues always arise
6. **Print and proofread** - Errors hide on screens
7. **Check requirements** - University-specific format rules

---

## Emergency Compression Tactics

If over 30 pages:
1. **Combine figures:** Multi-panel plots
2. **Reduce spacing:** Adjust margins carefully
3. **Move to appendix:** Extended results, configurations
4. **Tighten prose:** Remove "the" where possible
5. **Smaller fonts:** Tables/figures can go to 9pt
6. **Single-space references:** Often allowed
7. **Inline short lists:** Save vertical space

Remember: Clear and concise > verbose and complete