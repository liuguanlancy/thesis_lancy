# Qwen2-0.5B Experimental Plan Visualization

## Overview: Complete Experimental Pipeline (72 Experiments)

```mermaid
graph TB
    Start([Start: Qwen2-0.5B Base Model]) --> Phase1[Phase 1: Baseline Establishment<br/>10 Zero-shot Evaluations]
    
    Phase1 --> Phase2[Phase 2: Pretraining<br/>27 Experiments]
    Phase1 --> Phase3[Phase 3: Fine-tuning<br/>35 Experiments]
    
    Phase2 --> Phase2A[2A: General Pretraining<br/>9 experiments]
    Phase2 --> Phase2B[2B: Domain Pretraining<br/>12 experiments]
    Phase2 --> Phase2C[2C: Cross-Domain<br/>6 experiments]
    
    Phase3 --> Phase3A[3A: Direct Fine-tuning<br/>10 experiments]
    Phase3 --> Phase3B[3B: Multi-task<br/>10 experiments]
    Phase3 --> Phase3C[3C: Sequential<br/>15 experiments]
    
    Phase2A --> Eval[Evaluation on<br/>10 Tasks]
    Phase2B --> Eval
    Phase2C --> Eval
    Phase3A --> Eval
    Phase3B --> Eval
    Phase3C --> Eval
    
    Eval --> Results[Final Results:<br/>Training Taxonomy]
```

## Phase 1: Baseline Establishment (10 experiments)

```mermaid
graph LR
    Model[Qwen2-0.5B<br/>Base Model] --> ZS[Zero-shot Evaluation]
    
    ZS --> T1[Financial Sentiment]
    ZS --> T2[Tweet Sentiment]
    ZS --> T3[Topic Classification]
    ZS --> T4[Financial Q&A]
    ZS --> T5[Conversational Q&A]
    ZS --> T6[IMDB Sentiment]
    ZS --> T7[Math Reasoning]
    ZS --> T8[GLUE SST-2]
    ZS --> T9[Code Generation]
    ZS --> T10[NER]
    
    T1 --> Baseline[Baseline Scores<br/>for Comparison]
    T2 --> Baseline
    T3 --> Baseline
    T4 --> Baseline
    T5 --> Baseline
    T6 --> Baseline
    T7 --> Baseline
    T8 --> Baseline
    T9 --> Baseline
    T10 --> Baseline
```

## Phase 2: Pretraining Experiments (27 experiments)

### 2A: General Pretraining (9 experiments)

```mermaid
graph TD
    Base[Qwen2-0.5B] --> GP[General Pretraining<br/>with LoRA r=4]
    
    GP --> Single[Single Corpus<br/>OpenWebText]
    GP --> Double[Two Corpora<br/>OpenWebText + WikiText<br/>70:30 mixture]
    GP --> Triple[Three Corpora<br/>OpenWebText + WikiText + BookCorpus<br/>50:25:25 mixture]
    
    Single --> S1[10k steps]
    Single --> S2[25k steps]
    Single --> S3[50k steps]
    
    Double --> D1[10k steps]
    Double --> D2[25k steps]
    Double --> D3[50k steps]
    
    Triple --> T1[10k steps]
    Triple --> T2[25k steps]
    Triple --> T3[50k steps]
```

### 2B: Domain Continued Pretraining (12 experiments)

```mermaid
graph TD
    Base[Qwen2-0.5B] --> DP[Domain Pretraining<br/>Financial Texts as Corpus]
    
    DP --> QA[Financial Q&A<br/>2 scales: 5k, 10k]
    DP --> Sent[FinGPT Sentiment<br/>2 scales: 10k, 20k]
    DP --> Alpaca[Finance Alpaca<br/>2 scales: 10k, 20k]
    DP --> Mixed[Mixed Financial<br/>40:30:30 mixture<br/>2 scales: 10k, 20k]
    DP --> FiQA[FiQA Dataset<br/>2 scales: 5k, 10k]
    DP --> Twitter[Twitter Sentiment<br/>2 scales: 5k, 10k]
```

### 2C: Cross-Domain Pretraining (6 experiments)

```mermaid
graph LR
    Base[Qwen2-0.5B] --> CD[Cross-Domain<br/>Pretraining]
    
    CD --> Math1[GSM8K<br/>10k steps]
    CD --> Math2[DeepMind Math<br/>10k steps]
    CD --> Code[BigCodeBench<br/>10k steps]
    CD --> GLUE[GLUE MNLI<br/>10k steps]
    CD --> MMLU[MMLU-Pro<br/>10k steps]
    CD --> Mix[Math+Code Mix<br/>60:40 ratio<br/>10k steps]
```

## Phase 3: Fine-tuning Experiments (35 experiments)

### 3A: Direct Fine-tuning (10 experiments)

```mermaid
graph TD
    Base[Qwen2-0.5B] --> DF[Direct Fine-tuning<br/>with LoRA r=4<br/>5000 steps each]
    
    DF --> F1[Financial<br/>Sentiment]
    DF --> F2[Tweet<br/>Sentiment]
    DF --> F3[Topic<br/>Classification]
    DF --> F4[Financial<br/>Q&A]
    DF --> F5[Math<br/>Reasoning]
    DF --> F6[IMDB]
    DF --> F7[GLUE<br/>SST-2]
    DF --> F8[GLUE<br/>CoLA]
    DF --> F9[GLUE<br/>MRPC]
    DF --> F10[FiQA<br/>Q&A]
```

### 3B: Multi-task Fine-tuning (10 experiments)

```mermaid
graph TD
    Base[Qwen2-0.5B] --> MT[Multi-task Training<br/>with LoRA r=4]
    
    MT --> AllSent[All Sentiment<br/>3 datasets<br/>30:30:40 mix]
    MT --> AllQA[All Q&A<br/>3 datasets<br/>40:30:30 mix]
    MT --> MixClass[Mixed Classification<br/>3 datasets<br/>40:30:30 mix]
    MT --> FinGen[Financial+General<br/>2 datasets<br/>60:40 mix]
    MT --> MathFin[Math+Financial<br/>2 datasets<br/>50:50 mix]
    MT --> GLUEMulti[GLUE Tasks<br/>3 configs<br/>40:30:30 mix]
    MT --> FinComp[Financial Complete<br/>4 datasets<br/>25:25:25:25 mix]
    MT --> CrossDom[Cross-Domain<br/>3 datasets<br/>30:30:40 mix]
    MT --> Instruct[Instruction Following<br/>2 datasets<br/>50:50 mix]
    MT --> AdvReason[Advanced Reasoning<br/>2 datasets<br/>50:50 mix]
```

### 3C: Sequential Fine-tuning (15 experiments, 6 sequences)

```mermaid
graph LR
    subgraph Sequence 1: General to Specific
        IMDB[IMDB<br/>2k steps] --> FinSent[Financial Sentiment<br/>2k steps] --> TweetSent[Twitter Sentiment<br/>2k steps]
    end
    
    subgraph Sequence 2: Easy to Hard
        Binary[Binary Sentiment<br/>2k steps] --> Three[3-way Sentiment<br/>2k steps] --> Twenty[20-way Topics<br/>3k steps] --> QA[Financial Q&A<br/>3k steps]
    end
    
    subgraph Sequence 3: Pretrain then Finetune
        PreFin[Finance Pretrain<br/>10k steps] --> FineTune[Sentiment Finetune<br/>5k steps]
    end
    
    subgraph Sequence 4: Math to Financial
        MathPre[Math Pretrain<br/>5k steps] --> FinQA[Financial Q&A<br/>5k steps]
    end
    
    subgraph Sequence 5: Multi-domain Progression
        GenText[General Text<br/>5k steps] --> FinDomain[Financial Domain<br/>5k steps] --> TaskSpec[Task Specific<br/>3k steps]
    end
    
    subgraph Sequence 6: GLUE Progression
        GLUEMix[CoLA→SST2→MRPC<br/>6k steps total]
    end
```

## Evaluation Framework

```mermaid
graph TD
    Models[Trained Models<br/>from Each Experiment] --> Eval[Evaluation Suite]
    
    Eval --> FinTasks[Financial Tasks<br/>5 evaluations]
    Eval --> GenTasks[General Tasks<br/>3 evaluations]
    Eval --> TransTasks[Transfer Tasks<br/>2 evaluations]
    
    FinTasks --> Metrics1[Accuracy<br/>F1 Score<br/>Precision/Recall]
    GenTasks --> Metrics2[BLEU/ROUGE<br/>Perplexity<br/>Exact Match]
    TransTasks --> Metrics3[Cross-domain<br/>Performance<br/>Transfer Rate]
    
    Metrics1 --> Analysis[Comparative Analysis]
    Metrics2 --> Analysis
    Metrics3 --> Analysis
    
    Analysis --> Taxonomy[Training Taxonomy:<br/>Optimal Strategies]
```

## Resource Flow and Timeline

```mermaid
gantt
    title Qwen2-0.5B Experimental Timeline (342 GPU Hours)
    dateFormat  YYYY-MM-DD
    section Phase 1
    Baselines (10 exp)           :done, p1, 2024-01-01, 2h
    
    section Phase 2A
    General Pretrain 10k (3 exp) :p2a1, after p1, 15h
    General Pretrain 25k (3 exp) :p2a2, after p2a1, 38h
    General Pretrain 50k (3 exp) :p2a3, after p2a2, 75h
    
    section Phase 2B
    Domain Pretrain (12 exp)     :p2b, after p1, 68h
    
    section Phase 2C
    Cross-Domain (6 exp)          :p2c, after p1, 30h
    
    section Phase 3A
    Direct Fine-tuning (10 exp)   :p3a, after p1, 25h
    
    section Phase 3B
    Multi-task (10 exp)           :p3b, after p3a, 60h
    
    section Phase 3C
    Sequential (15 exp)           :p3c, after p3b, 30h
```

## Key Insights from Experimental Design

```mermaid
graph TD
    RQ[Research Questions] --> Design[Experimental Design]
    
    Design --> Q1[RQ1: Does continued pretraining help?<br/>→ Phase 2B experiments]
    Design --> Q2[RQ2: Optimal data mixture?<br/>→ Phase 3B multi-task]
    Design --> Q3[RQ3: Cross-task transfer?<br/>→ Phase 2C & 3C sequential]
    Design --> Q4[RQ4: Universal recipe?<br/>→ Compare all phases]
    Design --> Q5[RQ5: Min data/compute for 80% GPT-4?<br/>→ Scaling analysis across steps]
    
    Q1 --> Answer[Training Taxonomy]
    Q2 --> Answer
    Q3 --> Answer
    Q4 --> Answer
    Q5 --> Answer
```

## Cost-Optimized Path (82 GPU Hours)

```mermaid
graph TB
    Start[Qwen2-0.5B] --> Fast[Fast Track:<br/>82 GPU Hours]
    
    Fast --> P1F[Phase 1: All Baselines<br/>1.7 hours]
    Fast --> P2F[Phase 2: 6 Key Domain<br/>30 hours]
    Fast --> P3F[Phase 3: 10 Essential<br/>50 hours]
    
    P1F --> QuickResults[Initial Results<br/>in 3.4 days]
    P2F --> QuickResults
    P3F --> QuickResults
    
    QuickResults --> Decision{Promising?}
    Decision -->|Yes| Full[Run Full 342 Hours]
    Decision -->|No| Adjust[Adjust Strategy]
```

## Summary Statistics

| Phase | Experiments | GPU Hours | Key Question Answered |
|-------|------------|-----------|----------------------|
| 1 | 10 | 1.7 | What's the zero-shot baseline? |
| 2A | 9 | 127.5 | Does general pretraining help? |
| 2B | 12 | 67.5 | Does domain pretraining help? |
| 2C | 6 | 30.0 | Does cross-domain help? |
| 3A | 10 | 25.0 | How well does direct fine-tuning work? |
| 3B | 10 | 60.0 | Are multi-task approaches better? |
| 3C | 15 | 30.0 | Do sequential strategies help? |
| **Total** | **72** | **341.7** | **Complete Training Taxonomy** |