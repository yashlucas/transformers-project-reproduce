# Project Setup and Reproduction Guide

This repository contains code to reproduce baseline GLUE results, measure model efficiency, and run lightweight distillation and ablation experiments.

---

## Step 1: Set Up Hugging Face Transformers Locally

Clone the Hugging Face Transformers repository and install dependencies in a virtual environment.

    git clone https://github.com/huggingface/transformers.git
    cd transformers

    python -m venv .venv
    source .venv/bin/activate        # Linux / macOS
    # or
    .\.venv\Scripts\Activate.ps1     # Windows

    pip install -U pip
    pip install -e .
    pip install torch datasets accelerate evaluate psutil

---

## Step 2: Clone This Project Repository

Clone this project and move into its directory.

    git clone <your-project-repository-url>
    cd <your-project-repository>

---

## Step 3: Run Baseline Reproduction (Paper Results)

Reproduce baseline GLUE results using BERT and DistilBERT.

### BERT

    python run_glue.py \
      --model_name_or_path bert-base-uncased \
      --task_name <TASK_NAME> \
      --do_train \
      --do_eval \
      --output_dir ./tmp/<TASK_NAME>_bert

### DistilBERT

    python run_glue.py \
      --model_name_or_path distilbert-base-uncased \
      --task_name <TASK_NAME> \
      --do_train \
      --do_eval \
      --output_dir ./tmp/<TASK_NAME>_distilbert

---

## Step 4: Measure Model Size and Inference Speed

GPU inference speed and model size results are stored in:

    table3_gpu_results.json

---

## Step 5: Run Lightweight Distillation (Ablation Extension)

Run lightweight distillation pretraining:

    python mini_distil_pretrain.py \
      --out_dir ./Models\ for\ Ablation/student_full \
      --steps 500 \
      --batch_size 8

---

## Step 6: Run Ablation Variants

### Ablation 1: Disable Cosine Embedding Loss

    python mini_distil_pretrain.py \
      --out_dir ./Models\ for\ Ablation/no_cosine \
      --use_cos 0

### Ablation 2: Disable Distillation (KL / Soft-Label Loss)

    python mini_distil_pretrain.py \
      --out_dir ./Models\ for\ Ablation/no_ce \
      --use_ce 0

### Ablation 3: Disable Masked Language Modeling Loss

    python mini_distil_pretrain.py \
      --out_dir ./Models\ for\ Ablation/no_mlm \
      --use_mlm 0

### Ablation 4: Random Weight Initialization

    python mini_distil_pretrain.py \
      --out_dir ./Models\ for\ Ablation/random_init \
      --rand_init 1

---

## Step 7: Fine-tune Ablation Models

Fine-tune and evaluate all ablation models on GLUE tasks:

    python glue_tasks_all_abls.py
