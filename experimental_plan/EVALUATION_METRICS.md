# Evaluation Metrics for Financial Language Models

## Overview

This document defines comprehensive evaluation metrics for assessing the performance of lightweight financial language models. The metrics cover both task-specific performance and system-level efficiency, ensuring holistic evaluation of model capabilities.

## Metric Categories

### 1. Task-Specific Metrics

#### 1.1 Financial Question Answering

##### Exact Match (EM)
- **Definition**: Percentage of predictions that match ground truth exactly
- **Application**: Numerical answers, entity extraction
- **Formula**: `EM = (# exact matches) / (# total questions)`
- **Threshold**: Target ≥ 70% for financial facts

##### F1 Score
- **Definition**: Token-level overlap between prediction and ground truth
- **Application**: Longer text answers, explanations
- **Formula**: Harmonic mean of precision and recall
- **Threshold**: Target ≥ 80% for descriptive answers

##### Numerical Accuracy
- **Definition**: Accuracy within acceptable tolerance for numerical answers
- **Tolerances**:
  - Percentages: ±0.1%
  - Ratios: ±0.01
  - Currency: ±0.5% or $1000 (whichever is larger)
- **Special Cases**: Handle rounding, unit conversions

##### Execution Accuracy (for FinQA)
- **Definition**: Percentage of correct numerical reasoning programs
- **Components**: 
  - Program accuracy: Correct operations
  - Execution accuracy: Correct final answer
- **Evaluation**: Step-by-step reasoning validation

#### 1.2 Sentiment Analysis

##### Classification Accuracy
- **Definition**: Percentage of correctly classified sentiments
- **Classes**: Positive, Negative, Neutral
- **Weighted**: Account for class imbalance

##### Macro F1
- **Definition**: Average F1 across all sentiment classes
- **Advantage**: Treats all classes equally
- **Formula**: `Macro-F1 = mean(F1_positive, F1_negative, F1_neutral)`

##### Matthews Correlation Coefficient (MCC)
- **Definition**: Correlation between predicted and actual classifications
- **Range**: -1 to +1 (0 = random prediction)
- **Advantage**: Robust to class imbalance

##### Aspect-Based Accuracy
- **Definition**: Accuracy on entity-specific sentiments
- **Example**: "Apple's revenue grew (positive) but margins declined (negative)"
- **Measurement**: Per-aspect accuracy

#### 1.3 Document Summarization

##### ROUGE Scores
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-W**: Weighted longest common subsequence
- **Thresholds**:
  - ROUGE-1: ≥ 0.40
  - ROUGE-2: ≥ 0.18
  - ROUGE-L: ≥ 0.36

##### BERTScore
- **Definition**: Semantic similarity using BERT embeddings
- **Advantages**: Captures semantic meaning beyond n-grams
- **Threshold**: ≥ 0.85 for financial summaries

##### Factual Consistency
- **Method**: Fact extraction and verification
- **Metrics**:
  - Precision: % of facts in summary that are correct
  - Recall: % of important facts included
- **Verification**: Against source document

##### Key Information Retention
- **Financial Metrics**: Revenue, profit, ratios mentioned
- **Entities**: Companies, people, dates preserved
- **Quantitative**: Numbers accurately reported

#### 1.4 Financial Entity Recognition

##### Entity-Level F1
- **Precision**: Correct entities / predicted entities
- **Recall**: Correct entities / ground truth entities
- **Categories**:
  - Organizations (ORG)
  - Monetary values (MONEY)
  - Percentages (PERCENT)
  - Dates (DATE)
  - Financial metrics (FIN_METRIC)

##### Span-Level Accuracy
- **Exact Match**: Both boundaries correct
- **Partial Match**: At least one boundary correct
- **Type Accuracy**: Entity type correctly identified

#### 1.5 Numerical Reasoning

##### Calculation Accuracy
- **Arithmetic**: +, -, ×, ÷ operations
- **Percentage**: Growth rates, margins
- **Ratios**: P/E, debt-to-equity, etc.
- **Tolerance**: 0.1% for final answers

##### Multi-Step Reasoning
- **Step Accuracy**: % of correct intermediate steps
- **Final Accuracy**: Correct final answer
- **Reasoning Chain**: Logical flow evaluation

### 2. Generation Quality Metrics

#### 2.1 Fluency and Coherence

##### Perplexity
- **Definition**: Model's uncertainty in generation
- **Lower is better**: Target < 20 for financial domain
- **Domain-specific**: Measured on financial test set

##### Grammar Score
- **Tool**: LanguageTool or Grammarly API
- **Metrics**: Error rate per 100 words
- **Target**: < 2 errors per 100 words

##### Coherence Score
- **Method**: Sentence embedding similarity
- **Measurement**: Average cosine similarity between adjacent sentences
- **Target**: > 0.7

#### 2.2 Relevance and Informativeness

##### Relevance Score
- **Method**: Semantic similarity to query
- **Tool**: Sentence-BERT embeddings
- **Threshold**: > 0.8 cosine similarity

##### Information Density
- **Definition**: Unique financial facts per 100 words
- **Measurement**: Named entity and fact extraction
- **Target**: > 5 facts per 100 words

#### 2.3 Safety and Factuality

##### Hallucination Rate
- **Definition**: % of generated facts not grounded in context
- **Measurement**: Fact verification against source
- **Target**: < 5% hallucination rate

##### Toxicity Score
- **Tool**: Perspective API
- **Threshold**: < 0.1 toxicity score
- **Categories**: Profanity, threats, insults

### 3. System Performance Metrics

#### 3.1 Inference Efficiency

##### Latency Metrics
```python
latency_metrics = {
    "time_to_first_token": "Time from input to first generated token",
    "tokens_per_second": "Generation speed after first token",
    "total_inference_time": "Complete end-to-end time",
    "p50_latency": "Median response time",
    "p95_latency": "95th percentile response time",
    "p99_latency": "99th percentile response time"
}
```

**Target Performance**:
- Time to first token: < 500ms
- Tokens/second: > 10 (on CPU), > 50 (on GPU)
- P95 latency: < 5 seconds for 200-token response

#### 3.2 Resource Utilization

##### Memory Metrics
- **Model Size**: On-disk storage (GB)
- **RAM Usage**: Peak memory during inference
- **VRAM Usage**: GPU memory (if applicable)
- **Memory Efficiency**: Tokens processed per GB RAM

**Targets by Deployment**:
- Mobile: < 2GB RAM usage
- Laptop: < 4GB RAM usage
- Server: < 8GB RAM usage

##### Compute Metrics
- **CPU Utilization**: Average and peak percentage
- **GPU Utilization**: If available
- **Power Consumption**: Watts during inference
- **Battery Impact**: Drain rate on mobile/laptop

#### 3.3 Scalability Metrics

##### Throughput
- **Requests/second**: Maximum concurrent requests
- **Batch Processing**: Tokens/second with batching
- **Queue Time**: Average wait time under load

##### Reliability
- **Error Rate**: Failures per 1000 requests
- **Recovery Time**: Time to recover from errors
- **Availability**: Uptime percentage

### 4. Comparative Metrics

#### 4.1 Baseline Comparisons

##### Relative Performance
```python
relative_metrics = {
    "vs_gpt4": "Performance relative to GPT-4",
    "vs_gpt35": "Performance relative to GPT-3.5",
    "vs_claude": "Performance relative to Claude",
    "vs_finance_bert": "Performance relative to FinBERT",
    "vs_bloomberg_gpt": "Performance relative to BloombergGPT"
}
```

##### Performance Retention
- **After Quantization**: % performance retained (INT8, INT4)
- **After Pruning**: % performance retained
- **After Distillation**: % performance vs teacher model

#### 4.2 Cost-Efficiency Metrics

##### Performance per Dollar
- **Training Cost**: $ per point of accuracy improvement
- **Inference Cost**: $ per 1M tokens processed
- **Total Cost of Ownership**: Including hardware, energy

##### Performance per Watt
- **Energy Efficiency**: Accuracy per kWh consumed
- **Carbon Footprint**: CO2 per training run
- **Green AI Score**: Composite efficiency metric

### 5. Human Evaluation Metrics

#### 5.1 Expert Assessment

##### Accuracy Rating
- **Scale**: 1-5 (1=Very Inaccurate, 5=Very Accurate)
- **Evaluators**: Financial analysts, accountants
- **Domains**: Various financial subdomains

##### Usefulness Rating
- **Scale**: 1-5 (1=Not Useful, 5=Very Useful)
- **Context**: Real-world financial tasks
- **Comparison**: Against current tools

##### Trust and Confidence
- **Would use for decisions**: Yes/No/Maybe
- **Confidence level**: 0-100%
- **Recommendation likelihood**: NPS score

#### 5.2 User Study Metrics

##### Task Completion Rate
- **Definition**: % of users successfully completing financial tasks
- **Measurement**: With and without model assistance
- **Improvement**: % improvement with model

##### Time Savings
- **Measurement**: Time to complete financial analysis
- **Baseline**: Manual analysis time
- **Efficiency Gain**: % time saved

##### Error Reduction
- **Baseline**: Human error rate without assistance
- **With Model**: Error rate with model assistance
- **Improvement**: % reduction in errors

### 6. Evaluation Protocols

#### 6.1 Automated Evaluation Pipeline

```python
evaluation_pipeline = {
    "stage1": "Data preprocessing and formatting",
    "stage2": "Model inference on test sets",
    "stage3": "Metric calculation",
    "stage4": "Statistical significance testing",
    "stage5": "Report generation"
}
```

#### 6.2 A/B Testing Framework

##### Experiment Design
- **Control**: Baseline model or no model
- **Treatment**: New model version
- **Sample Size**: Minimum 1000 queries per variant
- **Duration**: Minimum 1 week

##### Success Criteria
- **Primary Metrics**: Must show significant improvement
- **Secondary Metrics**: Should not degrade
- **Guardrail Metrics**: Must not violate thresholds

#### 6.3 Continuous Evaluation

##### Monitoring Metrics
- Daily performance tracking
- Drift detection
- Error analysis
- User feedback integration

##### Regression Testing
- Test suite of 1000 critical examples
- Run before each deployment
- Automated rollback triggers

## Metric Aggregation

### Composite Scores

#### Financial AI Score (FAIS)
```python
FAIS = weighted_average([
    (qa_accuracy, 0.25),
    (sentiment_f1, 0.15),
    (summary_rouge, 0.15),
    (numerical_accuracy, 0.20),
    (latency_score, 0.15),
    (memory_efficiency, 0.10)
])
```

#### Deployment Readiness Score
```python
deployment_score = all([
    accuracy >= 0.80,
    latency_p95 < 5000,  # ms
    memory_usage < 4000,  # MB
    error_rate < 0.01,
    hallucination_rate < 0.05
])
```

## Reporting Format

### Performance Report Template

```markdown
## Model Performance Report

### Model Information
- Model: [Name and Version]
- Parameters: [Size]
- Training Data: [Dataset details]
- Evaluation Date: [Date]

### Task Performance
| Task | Metric | Score | Baseline | Improvement |
|------|--------|-------|----------|-------------|
| QA | F1 | X.XX | X.XX | +X.X% |
| Sentiment | Accuracy | X.XX | X.XX | +X.X% |

### System Performance
| Metric | Value | Target | Pass/Fail |
|--------|-------|--------|-----------|
| Latency (p95) | Xms | <5000ms | ✓/✗ |
| Memory | XGB | <4GB | ✓/✗ |

### Recommendations
- [Key findings]
- [Areas for improvement]
- [Deployment readiness]
```

## Statistical Considerations

### Significance Testing
- **Method**: Bootstrap confidence intervals
- **Confidence Level**: 95%
- **Multiple Comparisons**: Bonferroni correction
- **Effect Size**: Cohen's d for practical significance

### Sample Size Requirements
- **Minimum**: 100 examples per metric
- **Recommended**: 1000 examples for stable estimates
- **Power Analysis**: 80% power to detect 5% improvement

This comprehensive evaluation framework ensures thorough assessment of financial language models across all critical dimensions, from task performance to deployment efficiency.