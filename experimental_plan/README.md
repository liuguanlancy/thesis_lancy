# Experimental Plan: Privacy-Preserving Financial Chatbot

## Thesis Overview

This experimental plan supports the Master's thesis: **"Information-Secure Language Models for Finance Applications"** at the University of Zurich.

### Core Innovation
Developing a **2B parameter financial chatbot** that runs entirely on personal devices (laptops/phones), enabling users to analyze sensitive financial documents without uploading data to external services like ChatGPT or Gemini.

## 🎯 Primary Goal

**Build a Gemma-2 2B based chatbot that achieves GPT-4 level performance on financial tasks while guaranteeing 100% data privacy through on-device processing.**

## 📊 Key Documents

1. **[EXPERIMENTAL_PLAN.md](./EXPERIMENTAL_PLAN.md)** - Complete 6-week implementation plan
2. **[DATASET_SELECTION.md](./DATASET_SELECTION.md)** - Public datasets for training private-capable model
3. **[EVALUATION_METRICS.md](./EVALUATION_METRICS.md)** - Performance and privacy benchmarks
4. **[HARDWARE_BENCHMARKING.md](./HARDWARE_BENCHMARKING.md)** - Device compatibility testing
5. **[TIMELINE_AND_MILESTONES.md](./TIMELINE_AND_MILESTONES.md)** - Week-by-week execution plan

## 🔑 Key Requirements

### Model Specifications
- **Size**: 2B parameters (reference: Apple Intelligence uses 3B)
- **Deployment**: Runs on 8GB RAM laptops and modern phones
- **Performance**: 85% of GPT-4's accuracy on financial tasks
- **Privacy**: Zero external data transmission

### Training Constraints  
- **Resources**: Free GPUs only (Google Colab, Kaggle)
- **Data**: Public financial documents only
- **Method**: LoRA fine-tuning (2% trainable parameters)
- **Time**: 6 weeks total development

## 🚀 Three-Phase Approach

### Phase 1: Financial Domain Injection (Weeks 1-2)
- Collect 5GB of public financial documents (SEC filings, textbooks)
- Continued pre-training on Gemma-2 2B
- Transform general model → financial expert

### Phase 2: Task Specialization (Weeks 3-4)
- LoRA fine-tuning on specific financial tasks
- Balance sheet analysis, ratio calculations, risk assessment
- Create 10K instruction-following examples

### Phase 3: Privacy & Deployment (Weeks 5-6)
- Quantization for edge devices (INT8/INT4)
- Build local application interface
- Validate privacy guarantees
- User acceptance testing

## 💡 Use Cases

The trained model will enable users to:
1. **Upload and analyze** confidential financial statements locally
2. **Calculate ratios** and identify trends without API calls
3. **Detect risks** and anomalies in pre-public documents
4. **Generate reports** from private company data
5. **Compare documents** across periods with complete privacy

## 🔒 Privacy Guarantee

### Training Privacy
- ✅ Trained exclusively on public data
- ✅ No proprietary information in training set
- ✅ SEC EDGAR compliance maintained

### Deployment Privacy
- 🔒 100% on-device processing
- 🔒 No network connections required
- 🔒 No data logging or telemetry
- 🔒 User documents never leave device

## 📈 Expected Outcomes

### Performance Targets
| Metric | Target | Baseline (GPT-4) |
|--------|--------|------------------|
| FinQA Accuracy | 75% | 85% |
| Sentiment F1 | 85% | 92% |
| Inference Speed | <2 sec | N/A (API) |
| Memory Usage | <4GB | N/A (Cloud) |
| Privacy Score | 100% | 0% |

### Deliverables
1. **Trained Models**
   - Gemma-2-Finance-FP16 (4GB)
   - Gemma-2-Finance-INT8 (2GB) 
   - Gemma-2-Finance-INT4 (1GB)

2. **Application**
   - Local web interface
   - Document upload capability
   - Real-time analysis
   - Export functionality

3. **Documentation**
   - Training methodology
   - Evaluation results
   - Deployment guide
   - Privacy validation

## 🛠 Technical Stack

### Models
- **Primary**: Google Gemma-2 2B
- **Fallback**: Meta LLaMA 3.2 1B
- **Baseline**: FinBERT (comparison only)

### Training
- **Framework**: HuggingFace Transformers + PEFT
- **Method**: LoRA (rank=16, alpha=32)
- **Resources**: Colab T4 GPU, Kaggle P100

### Deployment
- **Quantization**: GGML/ONNX
- **Interface**: Gradio/Streamlit
- **Runtime**: CPU/GPU inference

## 📚 Repository Integration

This experimental plan is implemented using the training pipeline in the parent repository:

```bash
# Example training command
python main.py \
    --model google/gemma-2-2b \
    --dataset virattt/financial-qa-10K \
    --mode sft \
    --use_lora \
    --lora_r 16 \
    --batch_size 4 \
    --max_steps 5000
```

## 🎓 Academic Context

### Research Questions
1. Can a 2B model match large model performance on domain-specific tasks?
2. What is the optimal privacy-performance tradeoff for financial AI?
3. How effective is LoRA for financial domain adaptation?

### Contributions
1. First sub-3B model achieving professional-grade financial analysis
2. Novel privacy-preserving architecture for regulated domains
3. Efficient training methodology using only free resources
4. Comprehensive benchmark for edge-deployed financial AI

## 📞 Contact

**Researcher**: Guanlan Liu  
**Institution**: University of Zurich  
**Email**: guanlan.liu@uzh.ch  
**Thesis**: Master's in Computer Science

## 🔮 Future Work

### Immediate Extensions
- Multi-language support for global markets
- Voice interface for accessibility
- Real-time market data integration

### Long-term Vision
- Federated learning framework
- Specialized hardware accelerators
- Expansion to healthcare/legal domains
- Industry partnerships for deployment

---

*This experimental plan demonstrates that powerful AI capabilities don't require massive models or cloud infrastructure, paving the way for privacy-first AI in sensitive domains.*