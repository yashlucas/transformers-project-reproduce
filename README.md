#  Reproducing DistilBERT: Smaller, Faster, Cheaper Transformers

This repository presents a **reproducibility study** of the paper:

> **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**  
> Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf  
> arXiv:1910.01108

The goal of this project is to **empirically verify the key claims** of the paper regarding
model compression, inference speed, and downstream task performance using modern tooling
and hardware.

---

##  Paper Background

DistilBERT is a compressed version of BERT obtained via **knowledge distillation during
pre-training**. The original paper claims that DistilBERT:

- Reduces parameter count by **~40%**
- Achieves **~60% faster inference**
- Retains **~97% of BERT’s performance**
- Performs competitively on **GLUE, IMDb, and SQuAD**

This repository evaluates whether these claims are **reproducible in practice**.

---

##  Experimental Setup

### Models Evaluated
- **BERT-base**
- **DistilBERT**
- **ELMo** (paper baseline; values copied from the original paper)

### Benchmarks
- **GLUE benchmark**
  - CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B, WNLI
- **IMDb** sentiment classification
- **SQuAD v1.1** question answering

### Framework
-  Hugging Face **Transformers**
- PyTorch backend
- Trainer API

### Training Protocol
- Fine-tuning for **3 epochs**
- Single-seed runs
- Default Trainer hyperparameters unless otherwise noted
- Pretrained checkpoints:
  - `bert-base-uncased`
  - `distilbert-base-uncased`

>  Due to computational constraints, **pre-training distillation was not reproduced**.
> This study focuses on **fine-tuning and evaluation**, which is standard for reproduction studies.

---


## 📊 Results

### 🔹 Model Size Comparison

| Model | Parameters (Millions) |
|------|------------------------|
| ELMo (paper baseline) | 180 |
| BERT-base | 108.31 |
| DistilBERT | 65.78 |

➡️ **DistilBERT achieves a ~39% parameter reduction**, closely matching the paper’s reported 40%.

---

### 🔹 Inference Speed (STS-B Full Pass)

| Model | Time (seconds) | Hardware |
|------|----------------|----------|
| BERT-base | 13.04 | GPU |
| DistilBERT | 6.61 | GPU |

➡️ **DistilBERT is ~1.97× faster than BERT-base**.

While the paper reports ~60% speedup on CPU, GPU-based evaluation yields an even closer
match to the theoretical **2× speedup** expected from halving the number of Transformer
layers.

---

### 🔹 GLUE Benchmark (Selected Results)

| Task | Metric | BERT | DistilBERT |
|------|--------|------|------------|
| CoLA | Matthews Corr. | 0.573 | 0.436 |
| MNLI | Accuracy | 0.844 | 0.822 |
| MRPC | F1 | 0.869 | **0.875** |
| QNLI | Accuracy | 0.919 | 0.880 |
| SST-2 | Accuracy | 0.926 | 0.905 |

**Observations**
- DistilBERT consistently underperforms BERT slightly
- Larger drops occur on syntax-sensitive tasks (e.g., CoLA)
- Smaller datasets (e.g., MRPC) exhibit higher variance

---

### 🔹 SQuAD v1.1 (BERT-base)

| Metric | Reproduced | Paper |
|--------|------------|-------|
| Exact Match | 81.79 | 81.2 |
| F1 | 88.87 | 88.5 |

➡️ Near-exact reproduction of the paper’s reported BERT-base performance.
