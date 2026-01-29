##  Repository Structure

The repository is organized to clearly separate **baseline reproductions**, **custom ablation models**, and **supporting scripts/results**.

├── README.md
│ └── Main entry point for the project.
│ Describes the paper, reproduced results, system configuration,
│ and instructions to run baseline experiments.
│
├── REPO_STRUCTURE.md
│ └── Provides a detailed explanation of the entire repository layout,
│ including baseline models, ablation models, scripts, and usage guidance.
│
├── table3_gpu_results.json
│ └── Stores reproduced parameter counts and inference-time results
│ (GPU-based) used for comparison with paper-reported values.
│
├── Models for Ablation/
│ └── Contains all custom student models trained specifically for
│ ablation and distillation experiments in this project.
│ These models are created here and are NOT standard
│ Hugging Face pretrained checkpoints.
│
├── cola_bert/
│ └── Baseline fine-tuned BERT-base model for the CoLA task
│ using a pretrained Hugging Face checkpoint.
│
├── cola_distilbert/
│ └── Baseline fine-tuned DistilBERT model for the CoLA task
│ using a pretrained Hugging Face checkpoint.
│
├── mnli_bert/
│ └── Baseline fine-tuned BERT-base model for MNLI.
│
├── mnli_distilbert/
│ └── Baseline fine-tuned DistilBERT model for MNLI.
│
├── mrpc_bert/
│ └── Baseline fine-tuned BERT-base model for MRPC.
│
├── imdb_bert/
│ └── Baseline fine-tuned BERT-base model for IMDb sentiment classification.
│
├── imdb_distilbert/
│ └── Baseline fine-tuned DistilBERT model for IMDb sentiment classification.
│
├── mini_distil_pretrain.py
│ └── Implements lightweight distillation-based pre-training.
│ Uses a frozen BERT-base teacher and trains a reduced-depth
│ student model with configurable loss components
│ (MLM, distillation/KL, cosine embedding loss).
│ Outputs student models into Models for Ablation/.
│
├── ablation_study.py
│ └── Runs multiple ablation configurations by enabling or disabling
│ different loss components and initialization strategies.
│ Used to study the qualitative impact of each component.
│
├── glue_tasks_all_abls.py
│ └── Fine-tunes ablation-generated student models on GLUE tasks
│ using a consistent evaluation setup for fair comparison.


---

### How to Use This Repository

- **To reproduce the paper’s main results**  
  Use the `*_bert/` and `*_distilbert/` directories, which rely on
  pretrained Hugging Face models and standard fine-tuning.

- **To run ablation and distillation extensions**  
  Use `mini_distil_pretrain.py`, followed by `ablation_study.py`
  and `glue_tasks_all_abls.py`.  
  Resulting student models are stored in `Models for Ablation/`.

---

### Key Distinction

- **Baseline models**: Pretrained Hugging Face checkpoints, fine-tuned only  
- **Ablation models**: Custom student models trained in this project  
- **Teacher model**: Loaded from Hugging Face, frozen, and never trained  



