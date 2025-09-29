# Dataset Selection for Privacy-Preserving Financial Chatbot

## Overview

This document outlines the dataset selection strategy for training a 2B parameter financial chatbot that runs entirely on-device. The focus is on publicly available financial data that can train the model to handle private/confidential documents without ever requiring external data transmission.

## Core Training Philosophy

**Key Principle**: Train on public data, deploy on private data

We train exclusively on publicly available financial documents to create a model that can then analyze users' private financial documents locally without any privacy concerns.

## Primary Training Datasets

### 1. Financial Documents Corpus (5GB Target)

#### SEC EDGAR Database (3GB)
**Purpose**: Core financial statement understanding

```python
edgar_data = {
    "10-K": {
        "description": "Annual reports",
        "companies": 1000,
        "years": 5,
        "size": "~1.5GB",
        "value": "Comprehensive financial statements, MD&A, risk factors"
    },
    "10-Q": {
        "description": "Quarterly reports",
        "companies": 500,
        "quarters": 8,
        "size": "~1GB",
        "value": "Recent financial trends, seasonal patterns"
    },
    "8-K": {
        "description": "Current reports",
        "count": 10000,
        "size": "~0.5GB",
        "value": "Material events, executive changes, acquisitions"
    }
}
```

#### Financial Textbooks & Guides (1GB)
**Purpose**: Financial concepts and terminology

- CFA Institute materials (public portions)
- Financial Accounting textbooks
- Corporate Finance texts
- Investment Analysis guides
- Financial modeling resources

#### Earnings & Analysis (1GB)
**Purpose**: Interpretation and communication skills

- Earnings call transcripts (S&P 500)
- Analyst reports (public sections)
- Financial news articles
- Market commentary

### 2. Task-Specific Instruction Datasets

#### Financial Q&A Datasets (Available on HuggingFace)

| Dataset | Size | Purpose | Integration Status |
|---------|------|---------|-------------------|
| FinQA | 8,281 examples | Numerical reasoning over financial data | ✅ Implemented |
| ConvFinQA | 3,892 conversations | Multi-turn financial dialogues | ✅ Implemented |
| virattt/financial-qa-10K | 7,000 pairs | 10-K report Q&A | ✅ Implemented |
| TAT-QA | 16,552 questions | Tabular & textual financial Q&A | 🔄 Pending |

#### Classification Datasets

| Dataset | Size | Purpose | Integration Status |
|---------|------|---------|-------------------|
| Financial PhraseBank | 4,840 sentences | Sentiment analysis | ✅ Implemented |
| FiQA Sentiment | 1,174 examples | Microblog sentiment | ✅ Implemented |
| Twitter Financial News | 11,932 tweets | Real-time sentiment | ✅ Implemented |
| FinGPT Sentiment | 76,772 examples | Instruction-based sentiment | ✅ Implemented |

### 3. Synthetic Data Generation

#### Custom Financial Tasks (10,000 examples)

```python
synthetic_tasks = {
    "balance_sheet_analysis": {
        "template": "Analyze this balance sheet and identify key metrics",
        "examples": 2000,
        "source": "Real 10-Ks with expert annotations"
    },
    "ratio_calculation": {
        "template": "Calculate {ratio} and explain its significance",
        "examples": 2000,
        "ratios": ["P/E", "ROE", "Debt/Equity", "Current Ratio", "Quick Ratio"]
    },
    "trend_identification": {
        "template": "Compare these financial statements and identify trends",
        "examples": 2000,
        "focus": "YoY growth, margin changes, efficiency improvements"
    },
    "risk_assessment": {
        "template": "Identify potential risks in this financial report",
        "examples": 2000,
        "risks": ["Liquidity", "Solvency", "Operational", "Market"]
    },
    "fraud_detection": {
        "template": "Review for accounting irregularities",
        "examples": 2000,
        "patterns": "Enron, WorldCom, Wirecard case studies"
    }
}
```

## Data Preprocessing Pipeline

### 1. Document Extraction
```python
def extract_financial_documents():
    """Extract structured data from raw financial documents"""
    steps = [
        "PDF to text conversion",
        "Table extraction (balance sheets, income statements)",
        "Section identification (MD&A, Risk Factors, Notes)",
        "Metadata extraction (company, period, filing date)"
    ]
    return structured_data
```

### 2. Privacy Scrubbing
```python
def ensure_privacy_safe_training():
    """Remove any potentially private information from training data"""
    filters = [
        "Remove email addresses",
        "Remove phone numbers",
        "Remove SSNs/tax IDs",
        "Anonymize executive names in examples",
        "Replace specific company names in synthetic data"
    ]
    return clean_data
```

### 3. Quality Filtering
```python
quality_criteria = {
    "min_length": 100,  # Minimum document length
    "max_length": 50000,  # Maximum document length
    "language": "english",  # English only for now
    "format": "clean_text",  # No HTML/XML artifacts
    "relevance": 0.8  # Financial content score
}
```

## Training Data Mix Strategy

### Phase 1: Domain Adaptation (Weeks 1-2)
```python
pretraining_mix = {
    "SEC_filings": 0.40,      # 40% - Core financial documents
    "textbooks": 0.20,        # 20% - Educational content
    "earnings_calls": 0.15,   # 15% - Natural financial language
    "news_articles": 0.15,    # 15% - Current events context
    "analyst_reports": 0.10   # 10% - Professional analysis
}
```

### Phase 2: Task Fine-tuning (Weeks 3-4)
```python
finetuning_mix = {
    "financial_qa": 0.30,        # 30% - Q&A capabilities
    "document_analysis": 0.25,   # 25% - Document understanding
    "ratio_calculation": 0.20,   # 20% - Numerical computation
    "sentiment_analysis": 0.15,  # 15% - Interpretation skills
    "risk_assessment": 0.10      # 10% - Critical analysis
}
```

## Validation Datasets

### Hold-out Test Sets
- 10% of each dataset reserved for validation
- Never seen during training
- Used for final performance evaluation

### Privacy Test Suite
```python
privacy_tests = {
    "no_memorization": "Ensure model doesn't memorize specific documents",
    "no_leakage": "Verify no training data appears in outputs",
    "generalization": "Test on completely novel financial statements",
    "adversarial": "Test with deliberately misleading queries"
}
```

## Data Augmentation Techniques

### 1. Template-Based Generation
```python
def generate_instruction_variants(base_instruction):
    """Create multiple phrasings of the same task"""
    variants = [
        "Analyze the following balance sheet",
        "Review this balance sheet and provide insights",
        "What are the key takeaways from this balance sheet?",
        "Examine the balance sheet below and identify important metrics"
    ]
    return variants
```

### 2. Financial Document Synthesis
```python
def create_synthetic_statements():
    """Generate realistic but fictional financial statements"""
    components = {
        "base": "Use real statement structure",
        "values": "Generate realistic but random numbers",
        "trends": "Apply common financial patterns",
        "anomalies": "Inject known red flags for detection training"
    }
    return synthetic_docs
```

## Implementation in Current Repository

### Datasets Currently Integrated (26 Total)

#### Successfully Integrated (High Priority)
1. **stanfordnlp/imdb** - Sentiment baseline
2. **glue** - Multi-task understanding
3. **virattt/financial-qa-10K** - Core financial Q&A ✅
4. **FinGPT/fingpt-sentiment-train** - Financial sentiment ✅
5. **AdaptLLM/finance-tasks** - Multiple financial tasks ✅
6. **zeroshot/twitter-financial-news-topic** - Topic classification ✅

#### Ready for Training
All datasets listed above are integrated into the training pipeline with:
- Proper tokenization functions
- Label mappings for classification
- Prompt templates for generation
- Task auto-detection

### Usage Examples

```bash
# Phase 1: Domain Adaptation
python main.py \
    --model google/gemma-2-2b \
    --dataset AdaptLLM/finance-tasks \
    --mode pretrain \
    --max_steps 10000 \
    --use_lora \
    --lora_r 16

# Phase 2: Task Fine-tuning
python main.py \
    --model google/gemma-2-2b \
    --dataset virattt/financial-qa-10K \
    --mode sft \
    --use_lora \
    --max_steps 5000 \
    --batch_size 4

# Phase 3: Multi-task Training
python main.py \
    --model google/gemma-2-2b \
    --datasets virattt/financial-qa-10K FinGPT/fingpt-sentiment-train \
    --mixture_rates 0.6 0.4 \
    --mode sft \
    --use_lora
```

## Data Storage and Management

### Storage Requirements
- Raw data: ~5GB
- Preprocessed data: ~3GB
- Tokenized cache: ~2GB
- **Total: ~10GB**

### Organization Structure
```
data/
├── raw/
│   ├── sec_filings/
│   ├── textbooks/
│   └── earnings/
├── processed/
│   ├── instruction_pairs/
│   ├── qa_datasets/
│   └── classification/
└── cache/
    └── tokenized/
```

## Privacy and Compliance

### Training Data Compliance
- ✅ All training data is publicly available
- ✅ No proprietary or confidential information used
- ✅ Compliant with SEC EDGAR terms of use
- ✅ Respects robots.txt and rate limits

### Deployment Privacy
- 🔒 Model never sends data externally
- 🔒 All inference happens on-device
- 🔒 No telemetry or usage tracking
- 🔒 User documents never used for training

## Future Dataset Considerations

### Potential Additions
1. **Multi-language Financial Data** - For global markets
2. **Historical Financial Crises** - For risk pattern recognition
3. **Regulatory Filings** - For compliance checking
4. **Industry-Specific Data** - For specialized sectors

### Federated Learning Potential
- Allow users to optionally contribute learnings
- Maintain complete privacy through differential privacy
- Improve model without centralizing data

## Conclusion

This dataset selection strategy ensures that our 2B parameter model can be trained entirely on public data while gaining the capabilities to analyze private financial documents. The focus on SEC filings and educational materials provides a strong foundation for understanding financial statements, while task-specific datasets enable practical capabilities like Q&A and analysis.

The privacy-first approach means users can confidently analyze their confidential financial documents knowing that the model was trained only on public data and operates entirely offline.