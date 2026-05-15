"""Batch retro-eval all mono benchmarks via vLLM for transformers-version-stable baseline.

Mirror of retro_eval_vllm.py (which handles WMT22 translation) but for mono
loglikelihood tasks. Use this to:
1. Get a clean post-transformers-4.57.6 baseline on base + all key ckpts
2. Verify the vLLM/transformers loglikelihood agreement
3. Replace slow eval_with_piece.py runs (5-10x faster)
"""
import argparse
import json
import os
import subprocess

SUMMER = "/home/tfbao/Shiyu/Summer"
PYTHON = "/home/tfbao/.venv/bin/python"

CKPTS = {
    "base":   "/home/tfbao/new/Qwen3-0.6B-Base",
    "v8_p1":  f"{SUMMER}/output/phase1_ckpt_v8",
    "v10_p2": f"{SUMMER}/output/phase2_ckpt_v10_aurora_anneal/checkpoint-2000",
    "v11_p2": f"{SUMMER}/output/phase2_ckpt_v11_aurora_anneal/checkpoint-2000",
    "v12_p1": f"{SUMMER}/output/phase1_ckpt_v12/checkpoint-5000",
    "v12_p2": f"{SUMMER}/output/phase2_ckpt_v12_aurora_anneal/checkpoint-2000",
    "v15_p1": f"{SUMMER}/output/phase1_ckpt_v15",
    "v16_p2": f"{SUMMER}/output/phase2_ckpt_v16_aurora_anneal/checkpoint-2000",
}

# (task, num_fewshot) matching ReTok paper Table 2 / our prior eval setup
TASKS = [
    ("lambada_openai", 0),
    ("piqa", 5),
    ("arc_challenge", 25),
    ("hellaswag", 10),
    ("mmlu", 5),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS.keys()))
    p.add_argument("--tasks", nargs="+", default=[t for t, _ in TASKS])
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke-test limit (eg 100)")
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args()

    task_shots = dict(TASKS)

    for tag in args.ckpts:
        if tag not in CKPTS:
            print(f"[SKIP] {tag}: not in CKPTS map")
            continue
        model_path = CKPTS[tag]
        if not os.path.exists(model_path):
            print(f"[SKIP] {tag}: missing path {model_path}")
            continue

        for task in args.tasks:
            if task not in task_shots:
                print(f"[SKIP] task={task} not in TASKS")
                continue
            shots = task_shots[task]
            out_dir = f"{SUMMER}/eval_results/full/{tag}_vllm/{task}"
            out_json = f"{out_dir}/result.json"
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n========== {tag} / {task} ({shots}-shot) ==========")
            cmd = [
                PYTHON, "-u", f"{SUMMER}/eval_with_piece_vllm.py",
                "--model_path", model_path,
                "--task", task,
                "--num_fewshot", str(shots),
                "--max_model_len", str(args.max_model_len),
                "--output_path", out_json,
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            log_path = f"{out_dir}/run.log"
            with open(log_path, "w") as logf:
                ret = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            # vLLM exits with -6 (SIGABRT) on teardown even after writing results.
            # Trust the result.json file rather than the exit code.
            if os.path.exists(out_json):
                try:
                    d = json.load(open(out_json))
                    res = d.get("results", {})
                    for sub, m in res.items():
                        for k in ("acc_norm,none", "acc,none"):
                            if k in m:
                                print(f"  {sub}: {k}={m[k]:.4f}"
                                      + (f"   [exit={ret}]" if ret != 0 else ""))
                                break
                except Exception as e:
                    print(f"  parse fail: {e}")
            else:
                print(f"[FAIL] {tag}/{task} no result.json — see {log_path}")


if __name__ == "__main__":
    main()
