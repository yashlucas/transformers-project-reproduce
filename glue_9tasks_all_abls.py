import os
import subprocess
import sys

ABLATIONS = ["abl_only_cos", "abl_only_mlm", "abl_rand_init"]
TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

RUN_GLUE = os.path.join("examples", "pytorch", "text-classification", "run_glue.py")

OUT_BASE = os.path.join("examples", "pytorch", "text-classification", "tmp", "table4_runs")

def run(cmd):
    print("\n" + "=" * 110)
    print("RUN:", " ".join(cmd))
    print("=" * 110)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    if not os.path.exists(RUN_GLUE):
        raise SystemExit(f"ERROR: run_glue.py not found at: {RUN_GLUE}")

    py = sys.executable
    os.makedirs(OUT_BASE, exist_ok=True)

    common = [
        py, RUN_GLUE,
        "--do_train",
        "--do_eval",
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "64",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "3",
        "--logging_steps", "50",
        "--report_to", "none",
        "--save_steps", "0",
    ]

    for abl in ABLATIONS:
        if not os.path.isdir(abl):
            print(f"WARNING: missing folder {abl}, skipping.")
            continue

        for task in TASKS:
            outdir = os.path.join(OUT_BASE, abl, task)
            os.makedirs(outdir, exist_ok=True)

            cmd = common + [
                "--model_name_or_path", abl,
                "--task_name", task,
                "--output_dir", outdir,
            ]

            run(cmd)

    print("\n✅ Done. Results saved under:")
    print(OUT_BASE)

if __name__ == "__main__":
    main()
