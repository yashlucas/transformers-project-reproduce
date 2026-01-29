# Ablation Models and Experimental Details

This directory contains all models and artifacts used for the **ablation experiments** conducted as an extension to the paper  
**“DistilBERT: a distilled version of BERT”**.

The goal of these experiments is to perform a **lightweight, proxy ablation study** that qualitatively validates the trends reported in the original paper’s pre-training ablation (Table 4), under realistic computational constraints.

---

## 1. Overview of Ablation Strategy

The original paper performs ablation during **large-scale pre-training distillation**, which requires extensive computational resources.

In this project, we implement a **scaled-down proxy ablation** by:

- Training a **6-layer BERT-style student model**
- Using **knowledge distillation losses** during masked language modeling
- Running short pre-training experiments on a small text corpus
- Fine-tuning the resulting student models on downstream GLUE tasks

This approach allows us to study the **relative importance of different loss components and initialization strategies**, without reproducing full pre-training from scratch.

---

## 2. Models Used in Ablation

### Teacher Model
- **BERT-base (bert-base-uncased)**
- 12 Transformer layers
- ~110M parameters
- Used in evaluation mode only
- Provides:
  - Soft logits (for distillation loss)
  - Hidden states (for cosine embedding loss)

### Student Model
- **BERT-style Masked Language Model**
- 6 Transformer encoder layers
- Same hidden size, attention heads, and vocabulary as BERT-base
- ~67M parameters
- Initialized in two ways:
  - From teacher weights (layer-wise copying)
  - Random initialization (for ablation)

> Note: The student uses a BERT architecture with reduced depth, acting as a DistilBERT-style proxy rather than the exact DistilBERT class.

---

## 3. Pre-training Distillation Setup

Pre-training is performed using a **triple-loss objective**, inspired by the original DistilBERT paper:

- **MLM Loss**  
  Standard masked language modeling loss on student outputs

- **Distillation (KL) Loss**  
  Kullback–Leibler divergence between student and teacher logits, with temperature scaling

- **Cosine Embedding Loss**  
  Alignment between student and teacher hidden representations

Each ablation disables exactly one component or changes the initialization strategy.

---

## 4. Ablation Variants Implemented

The following ablation variants are implemented using `mini_distil_pretrain.py`:

| Variant | MLM | CE (Distillation) | Cosine | Init |
|------|----|----|----|----|
| Full (baseline) | ✓ | ✓ | ✓ | Teacher |
| No cosine loss | ✓ | ✓ | ✗ | Teacher |
| No distillation loss | ✓ | ✗ | ✓ | Teacher |
| No MLM loss | ✗ | ✓ | ✓ | Teacher |
| Random initialization | ✓ | ✓ | ✓ | Random |

Each variant produces a separate student checkpoint saved in this directory.

---

## 5. Pre-training Script

### `mini_distil_pretrain.py`

This script:
- Loads the teacher model and tokenizer
- Constructs a 6-layer student model
- Optionally initializes student weights from the teacher
- Applies configurable loss components
- Trains for a limited number of steps on a small text corpus
- Saves the resulting student model and tokenizer

Key configurable arguments include:
- `--use_mlm`, `--use_ce`, `--use_cos`
- `--init {teacher,random}`
- `--steps`, `--batch_size`, `--lr`

This script enables systematic ablation without modifying the core training logic.

---

## 6. Downstream Fine-Tuning Experiments

After pre-training, student models are fine-tuned on downstream tasks using the Hugging Face `Trainer` API.

The following task-specific folders contain fine-tuned checkpoints and results:

- `cola_bert/`, `cola_distilbert/`
- `mnli_bert/`, `mnli_distilbert/`
- `mrpc_bert/`
- `imdb_bert/`, `imdb_distilbert/`

Each folder corresponds to:
- A single downstream task
- A specific model (BERT-base or DistilBERT-style student)
- Standard fine-tuning configuration (3 epochs, fixed learning rate)

---

## 7. Evaluation and Metrics

- GLUE tasks evaluated using official metrics (Accuracy, F1, Matthews Corr.)
- IMDb evaluated using classification accuracy
- Comparisons focus on **relative trends**, not absolute performance
- Single-seed experiments are used for consistency

---

## 8. Relationship to the Original Paper

The original DistilBERT paper reports a full-scale ablation study during pre-training (Table 4).

This project:
- Does **not** reproduce full-scale pre-training ablations
- Implements a **computationally feasible proxy**
- Validates the **directional trends** reported in the paper
- Clearly distinguishes reproduced results from paper-reported results

---

## 9. Key Takeaway

These ablation experiments demonstrate that:
- Knowledge distillation losses significantly influence student performance
- Teacher-based initialization is critical for effective compression
- Lightweight proxy experiments can still provide meaningful insights into model behavior

---

## References

Wolf, T., et al. (2020).  
**Transformers: State-of-the-Art Natural Language Processing**.  
EMNLP System Demonstrations.

Sanh, V., et al. (2019).  
**DistilBERT: a distilled version of BERT**.
