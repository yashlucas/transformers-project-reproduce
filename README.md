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


##  Results

###  Model Size Comparison

| Model | Parameters (Millions) |
|------|------------------------|
| ELMo (paper baseline) | 180 |
| BERT-base | 108.31 |
| DistilBERT | 65.78 |

 **DistilBERT achieves a ~39% parameter reduction**, closely matching the paper’s reported 40%.

---

###  Inference Speed (STS-B Full Pass)

| Model | Time (seconds) | Hardware |
|------|----------------|----------|
| BERT-base | 13.04 | GPU |
| DistilBERT | 6.61 | GPU |

 **DistilBERT is ~1.97× faster than BERT-base**.

While the paper reports ~60% speedup on CPU, GPU-based evaluation yields an even closer
match to the theoretical **2× speedup** expected from halving the number of Transformer
layers.

---

###  GLUE Benchmark (Full Comparison – All Tasks)

| Task | Metric | BERT | DistilBERT |
|------|--------|------|------------|
| CoLA | Matthews Corr. | 0.573 | 0.436 |
| MNLI | Accuracy | 0.844 | 0.822 |
| MRPC | F1 | 0.869 | **0.875** |
| QNLI | Accuracy | 0.919 | 0.880 |
| QQP | F1 | 0.881 | 0.869 |
| RTE | Accuracy | 0.617 | 0.585 |
| SST-2 | Accuracy | 0.926 | 0.905 |
| STS-B | Pearson Corr. | 0.881 | 0.858 |
| WNLI | Accuracy | 0.310 | 0.254 |

**Observations**
- DistilBERT retains performance close to BERT across all 9 GLUE tasks
- Larger drops occur on linguistically challenging or low-resource tasks (CoLA, RTE, WNLI)
- On MRPC, DistilBERT slightly outperforms BERT, likely due to dataset variance
- Overall trends align with the paper’s claim that DistilBERT retains ~97% of BERT’s performance
|

---

###  SQuAD v1.1 (BERT-base)

| Metric | Reproduced | Paper |
|--------|------------|-------|
| Exact Match | 81.79 | 81.2 |
| F1 | 88.87 | 88.5 |

 Near-exact reproduction of the paper’s reported BERT-base performance.

---

##  References

### DistilBERT Paper
Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020).  
**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**.  
arXiv preprint arXiv:1910.01108.  
https://arxiv.org/abs/1910.01108

---

### Transformers Library
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T.,
Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y.,
Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020).  
**Transformers: State-of-the-Art Natural Language Processing**.  
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing:
System Demonstrations, 38–45.  
https://www.aclweb.org/anthology/2020.emnlp-demos.6

---

### BibTeX

```bibtex
@article{sanh2020distilbert,
  title   = {DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author  = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal = {arXiv preprint arXiv:1910.01108},
  year    = {2020}
}

@inproceedings{wolf-etal-2020-transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and
               Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and
               Louf, R{\'e}mi and Funtowicz, Morgan and Davison, Joe and Shleifer, Sam and
               von Platen, Patrick and Ma, Clara and Jernite, Yacine and Plu, Julien and
               Xu, Canwen and Le Scao, Teven and Gugger, Sylvain and Drame, Mariama and
               Lhoest, Quentin and Rush, Alexander M.},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language
               Processing: System Demonstrations},
  year      = {2020},
  pages     = {38--45},
  url       = {https://www.aclweb.org/anthology/2020.emnlp-demos.6}
}


