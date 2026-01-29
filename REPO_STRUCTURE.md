## 📁 Repository Structure

This repository is organized to clearly separate **baseline reproductions**,  
**custom ablation models**, and **supporting scripts/results**.

---

### Core Documentation Files

- **README.md**  
  Main entry point for the project.  
  Describes the paper, reproduced results, system configuration, and instructions
  to run baseline experiments.

- **REPO_STRUCTURE.md**  
  Provides a detailed explanation of the entire repository layout, including
  baseline models, ablation models, scripts, and usage guidance.

---

### Result Files

- **table3_gpu_results.json**  
  Stores reproduced parameter counts and inference-time results (GPU-based),
  used for comparison with paper-reported values.

---

### Baseline Models (Pretrained Hugging Face)

These directories contain fine-tuned models initialized from Hugging Face
pretrained checkpoints and used for reproducing the paper’s main results.

- **cola_bert/** – BERT-base fine-tuned on CoLA  
- **cola_distilbert/** – DistilBERT fine-tuned on CoLA  
- **mnli_bert/** – BERT-base fine-tuned on MNLI  
- **mnli_distilbert/** – DistilBERT fine-tuned on MNLI  
- **mrpc_bert/** – BERT-base fine-tuned on MRPC  
- **imdb_bert/** – BERT-base fine-tuned on IMDb  
- **imdb_distilbert/** – DistilBERT fine-tuned on IMDb  

---

### Ablation & Distillation Models

- **Models for Ablation/**  
  Contains all **custom student models** trained specifically for ablation
  and distillation experiments in this project.  
  These models are created here and are **not standard Hugging Face
  pretrained checkpoints**.

---

### Training & Experiment Scripts

- **mini_distil_pretrain.py**  
  Implements lightweight distillation-based pre-training using a frozen
  BERT-base teacher and a reduced-depth student model with configurable
  loss components (MLM, distillation/KL, cosine embedding loss).

- **ablation_study.py**  
  Runs multiple ablation configurations by enabling or disabling different
  loss components and initialization strategies.

- **glue_tasks_all_abls.py**  
  Fine-tunes ablation-generated student models on GLUE tasks using a
  consistent evaluation setup for fair comparison.

---

### Summary

- **Baseline models**: Pretrained Hugging Face checkpoints, fine-tuned only  
- **Ablation models**: Custom student models trained in this project  
- **Teacher model**: Loaded from Hugging Face, frozen, never trained  

This structure cleanly separates **reproduction** and **extension** experiments.
