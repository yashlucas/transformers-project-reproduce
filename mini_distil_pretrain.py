import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    DistilBertConfig,
    DistilBertForMaskedLM,
    get_linear_schedule_with_warmup,
)

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def init_distil_student_from_bert_teacher(student: DistilBertForMaskedLM, teacher: BertForMaskedLM):
    """
    Initialize DistilBERT student from BERT teacher by taking one layer out of two:
      student.layer[i] <- teacher.encoder.layer[2*i]
    Copies:
      - word + position embeddings
      - transformer blocks (every other layer)
      - MLM head projection/decoder + layernorm (best-effort mapping)
    """
    # Embeddings
    student.distilbert.embeddings.word_embeddings.weight.copy_(
        teacher.bert.embeddings.word_embeddings.weight
    )
    student.distilbert.embeddings.position_embeddings.weight.copy_(
        teacher.bert.embeddings.position_embeddings.weight
    )
    student.distilbert.embeddings.LayerNorm.load_state_dict(
        teacher.bert.embeddings.LayerNorm.state_dict()
    )

    # Transformer blocks
    for i in range(student.config.n_layers):
        s_block = student.distilbert.transformer.layer[i]
        t_block = teacher.bert.encoder.layer[2 * i]

        # Attention
        s_block.attention.q_lin.weight.copy_(t_block.attention.self.query.weight)
        s_block.attention.q_lin.bias.copy_(t_block.attention.self.query.bias)
        s_block.attention.k_lin.weight.copy_(t_block.attention.self.key.weight)
        s_block.attention.k_lin.bias.copy_(t_block.attention.self.key.bias)
        s_block.attention.v_lin.weight.copy_(t_block.attention.self.value.weight)
        s_block.attention.v_lin.bias.copy_(t_block.attention.self.value.bias)

        s_block.attention.out_lin.weight.copy_(t_block.attention.output.dense.weight)
        s_block.attention.out_lin.bias.copy_(t_block.attention.output.dense.bias)

        s_block.sa_layer_norm.load_state_dict(t_block.attention.output.LayerNorm.state_dict())

        # FFN
        s_block.ffn.lin1.weight.copy_(t_block.intermediate.dense.weight)
        s_block.ffn.lin1.bias.copy_(t_block.intermediate.dense.bias)
        s_block.ffn.lin2.weight.copy_(t_block.output.dense.weight)
        s_block.ffn.lin2.bias.copy_(t_block.output.dense.bias)

        s_block.output_layer_norm.load_state_dict(t_block.output.LayerNorm.state_dict())

    # MLM head mapping (best effort)
    # DistilBERT MLM head: vocab_projector + vocab_layer_norm
    # BERT MLM head: cls.predictions.transform + decoder (+ bias)
    # Map decoder weights/bias directly, and LayerNorm if possible
    student.vocab_projector.weight.copy_(teacher.cls.predictions.decoder.weight)
    student.vocab_projector.bias.copy_(teacher.cls.predictions.decoder.bias)

    student.vocab_transform.weight.copy_(teacher.cls.predictions.transform.dense.weight)
    student.vocab_transform.bias.copy_(teacher.cls.predictions.transform.dense.bias)
    student.vocab_layer_norm.load_state_dict(teacher.cls.predictions.transform.LayerNorm.state_dict())

def cosine_loss(student_hid: torch.Tensor, teacher_hid: torch.Tensor) -> torch.Tensor:
    """
    student_hid, teacher_hid: [B, T, H]
    Average over tokens then cosine embedding loss over batch.
    """
    s = student_hid.mean(dim=1)
    t = teacher_hid.mean(dim=1)
    target = torch.ones(s.size(0), device=s.device)
    return F.cosine_embedding_loss(s, t, target)

def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s = student_logits / T
    t = teacher_logits / T
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean") * (T * T)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--teacher_model", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--use_mlm", type=int, default=1)
    ap.add_argument("--use_ce", type=int, default=1)
    ap.add_argument("--use_cos", type=int, default=1)
    ap.add_argument("--init", choices=["teacher", "random"], default="teacher")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--wiki_fraction", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)

    # Data
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    n = max(10_000, int(len(ds) * args.wiki_fraction))
    ds = ds.shuffle(seed=args.seed).select(range(n))

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length, padding="max_length")

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)

    # Teacher (BERT MLM)
    teacher = BertForMaskedLM.from_pretrained(args.teacher_model, output_hidden_states=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (DistilBERT MLM)
    tconf = teacher.config
    sconf = DistilBertConfig(
        vocab_size=tconf.vocab_size,
        max_position_embeddings=tconf.max_position_embeddings,
        n_layers=6,
        n_heads=tconf.num_attention_heads,
        dim=tconf.hidden_size,
        hidden_dim=tconf.intermediate_size,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
    )
    student = DistilBertForMaskedLM(sconf).to(device)

    if args.init == "teacher":
        init_distil_student_from_bert_teacher(student, teacher)

    print(f"Teacher params: {count_params(teacher)/1e6:.1f}M")
    print(f"Student params: {count_params(student)/1e6:.1f}M")
    print(f"Init: {args.init} | Losses: MLM={args.use_mlm} CE={args.use_ce} COS={args.use_cos}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    total_updates = math.ceil(args.steps / args.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_updates)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    student.train()
    step = 0
    update = 0
    t0 = time.time()

    dl_iter = iter(dl)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            with torch.cuda.amp.autocast(enabled=args.fp16):
                s_out = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"] if args.use_mlm else None,
                    output_hidden_states=True,
                )

                loss = 0.0
                if args.use_mlm:
                    loss = loss + s_out.loss
                if args.use_ce:
                    loss = loss + distill_kl(s_out.logits, t_out.logits, args.T)
                if args.use_cos:
                    loss = loss + cosine_loss(s_out.hidden_states[-1], t_out.hidden_states[-1])

                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            step += 1
            if step >= args.steps:
                break

        scaler.step(opt)
        scaler.update()
        sched.step()
        update += 1

        if update % 50 == 0:
            elapsed = time.time() - t0
            print(f"update={update:5d} step={step:5d} loss={accum_loss:.4f} elapsed={elapsed/60:.1f}m")

    student.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
