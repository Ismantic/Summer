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

# Expanded mono eval suite covering both ReTok and YuLan-Mini conventions.
# Default (no --tasks flag): runs all 8.
# English mono: piqa, arc_challenge, hellaswag, mmlu, lambada_openai
# Chinese mono: cmmlu (haonan-li/cmmlu, 67 subjects), ceval-valid (52 subjects)
# Generative:   gsm8k (math, 5-shot, generate_until)
TASKS = [
    ("lambada_openai", 0),
    ("piqa", 5),
    ("arc_challenge", 25),
    ("hellaswag", 10),
    ("mmlu", 5),
    # NOTE: cmmlu (haonan-li/cmmlu, lmlmcat, XiaHan19) all use cmmlu.py script
    # which `datasets >= 4.0` deprecated. Temporarily skipped — see TODO file.
    # ("cmmlu", 5),
    ("ceval-valid", 5),
    ("gsm8k", 5),
]


def run_ckpt(tag: str, model_path: str, task_spec: str, max_model_len: int,
             limit: int | None, gpu_id: int) -> int:
    """Run one ckpt's full task batch on a specific GPU. Returns subprocess rc."""
    out_dir = f"{SUMMER}/eval_results/full/{tag}_vllm"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        PYTHON, "-u", f"{SUMMER}/eval_with_piece_vllm.py",
        "--model_path", model_path,
        "--tasks", task_spec,
        "--output_dir", out_dir,
        "--max_model_len", str(max_model_len),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force HF offline at subprocess level too
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    log_path = f"{out_dir}/batch_run.log"
    print(f"  [GPU {gpu_id}] launching {tag} → {log_path}")
    with open(log_path, "w") as logf:
        ret = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS.keys()))
    p.add_argument("--tasks", nargs="+", default=[t for t, _ in TASKS])
    p.add_argument("--limit", type=int, default=None, help="Smoke-test limit (eg 100)")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU ids to use. Pass e.g. --gpus 0 1 to parallelize "
                        "ckpts across two GPUs (~2x speedup on multi-ckpt runs).")
    args = p.parse_args()

    task_shots = dict(TASKS)
    selected = [(t, task_shots[t]) for t in args.tasks if t in task_shots]
    task_spec = ",".join(f"{t}:{s}" for t, s in selected)
    print(f"Task spec: {task_spec}")
    print(f"GPUs: {args.gpus}")
    print(f"Ckpts: {args.ckpts}")

    # Filter valid ckpts up-front
    valid_ckpts = []
    for tag in args.ckpts:
        if tag not in CKPTS:
            print(f"[SKIP] {tag}: not in CKPTS map")
            continue
        if not os.path.exists(CKPTS[tag]):
            print(f"[SKIP] {tag}: missing path {CKPTS[tag]}")
            continue
        valid_ckpts.append(tag)

    # Round-robin assign ckpts to GPUs, then launch each GPU's queue in parallel
    n_gpu = len(args.gpus)
    queues: dict[int, list[str]] = {g: [] for g in args.gpus}
    for i, tag in enumerate(valid_ckpts):
        queues[args.gpus[i % n_gpu]].append(tag)

    print(f"\nQueue per GPU:")
    for g, q in queues.items():
        print(f"  GPU {g}: {q}")

    import concurrent.futures
    def worker(gpu_id: int):
        for tag in queues[gpu_id]:
            print(f"\n========== [GPU {gpu_id}] {tag} ==========")
            ret = run_ckpt(tag, CKPTS[tag], task_spec, args.max_model_len, args.limit, gpu_id)
            # Print summary
            out_dir = f"{SUMMER}/eval_results/full/{tag}_vllm"
            for task, shots in selected:
                out_json = f"{out_dir}/{task}/result.json"
                if os.path.exists(out_json):
                    try:
                        d = json.load(open(out_json))
                        for sub, m in d.get("results", {}).items():
                            for k in ("acc_norm,none", "acc,none", "exact_match,strict-match"):
                                if k in m:
                                    print(f"  [GPU {gpu_id}] {tag}/{task}/{sub}: {k}={m[k]:.4f}")
                                    break
                            break
                    except Exception as e:
                        print(f"  {tag}/{task} parse fail: {e}")
                else:
                    print(f"  [FAIL] {tag}/{task} no result.json — see {out_dir}/batch_run.log")
            if ret != 0:
                print(f"  [{tag}] subprocess rc={ret} (vLLM -6 on teardown is harmless)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpu) as ex:
        list(ex.map(worker, args.gpus))


if __name__ == "__main__":
    main()
