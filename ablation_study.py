import json
import os
import math

# Where your run_glue outputs are stored
BASE = os.path.join(
    "examples",
    "pytorch",
    "text-classification",
    "tmp",
    "table4_runs"
)


# Must match your folder names
ABLATIONS = [
    "abl_full",
    "abl_only_ce",
    "abl_only_cos",
    "abl_only_mlm",
    "abl_rand_init",
]

# 9 GLUE tasks
TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# Map your folders -> paper row names
PAPER_NAMES = {
    "abl_only_ce":   "∅  - Lcos - Lmlm",
    "abl_only_cos":  "Lce - ∅    - Lmlm",
    "abl_only_mlm":  "Lce - Lcos - ∅",
    "abl_rand_init": "Triple loss + random weights initialization",
}

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_metric(task: str, d: dict):
    """
    Pick a single metric per task (so macro avg is well-defined).
    This matches common HF GLUE practice:
      - CoLA: Matthews corr
      - STS-B: average of Pearson and Spearman (or combined_score if present)
      - MRPC/QQP: combined_score if present else F1 else accuracy
      - others: accuracy
    """
    if task == "cola":
        return d.get("eval_matthews_correlation")

    if task == "stsb":
        if "eval_combined_score" in d:
            return d["eval_combined_score"]
        p = d.get("eval_pearson")
        s = d.get("eval_spearmanr")
        if p is not None and s is not None:
            return (p + s) / 2.0
        return p if p is not None else s

    if task in ("mrpc", "qqp"):
        if "eval_combined_score" in d:
            return d["eval_combined_score"]
        return d.get("eval_f1") if d.get("eval_f1") is not None else d.get("eval_accuracy")

    return d.get("eval_accuracy")

def macro_score(abl: str) -> float:
    scores = []
    missing = []
    for task in TASKS:
        path = os.path.join(BASE, abl, task, "eval_results.json")
        if not os.path.exists(path):
            missing.append(task)
            continue
        d = read_json(path)
        m = pick_metric(task, d)
        if m is None:
            missing.append(task)
            continue
        scores.append(float(m))

    if missing:
        raise RuntimeError(f"{abl}: missing eval_results.json or metric for tasks: {missing}")

    return sum(scores) / len(scores)

def print_paper_table(rows):
    # rows: list of (ablation_name, delta)
    left_header = "Ablation"
    right_header = "Variation on GLUE macro-score"

    left_width = max(len(left_header), *(len(r[0]) for r in rows))
    right_width = max(len(right_header), 10)

    total_width = left_width + 3 + right_width  # " | " between

    # Paper-ish rules (top rule, mid rule, bottom rule)
    print("-" * total_width)
    print(f"{left_header:<{left_width}} | {right_header:>{right_width}}")
    print("-" * total_width)

    for name, delta in rows:
        # show delta like -2.96 with 2 decimals
        sdelta = f"{delta:+.5f}"
        # paper uses negative numbers without '+', so remove '+' if present
        if sdelta.startswith("+"):
            sdelta = sdelta[1:]
        print(f"{name:<{left_width}} | {sdelta:>{right_width}}")

    print("-" * total_width)

def main():
    if not os.path.isdir(BASE):
        raise SystemExit(f"ERROR: {BASE} not found. Run your GLUE jobs first.")

    full = macro_score("abl_full")

    # Build paper rows in the same order as the paper
    order = ["abl_only_ce", "abl_only_cos", "abl_only_mlm", "abl_rand_init"]

    rows = []
    for key in order:
        ms = macro_score(key)
        delta = ms - full
        rows.append((PAPER_NAMES[key], delta))

    print_paper_table(rows)

if __name__ == "__main__":
    main()
