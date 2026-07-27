"""Batch-eval all key ckpts with vLLM for consistent BLEU+COMET baseline.

vLLM is fast (~45s per ckpt for 1000-sample BLEU + ~50s for COMET) and
generation is decoupled from transformers.generate() instability.

Use this to establish a fair baseline after the May 14 transformers upgrade
made all pre-existing BLEU numbers incomparable. COMET is stable across engines.
"""
import argparse
import os
import subprocess

SUMMER = "~/Shiyu/Summer"
PYTHON = "~/.venv/bin/python"

CKPTS = {
    "base":   "~/new/Qwen3-0.6B-Base",
    "v8_p1":  f"{SUMMER}/output/phase1_ckpt_v8",
    "v10_p2": f"{SUMMER}/output/phase2_ckpt_v10_aurora_anneal/checkpoint-2000",
    "v11_p2": f"{SUMMER}/output/phase2_ckpt_v11_aurora_anneal/checkpoint-2000",
    "v12_p1": f"{SUMMER}/output/phase1_ckpt_v12/checkpoint-5000",
    "v12_p2": f"{SUMMER}/output/phase2_ckpt_v12_aurora_anneal/checkpoint-2000",
    "v15_p1": f"{SUMMER}/output/phase1_ckpt_v15",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS.keys()))
    p.add_argument("--max_samples", type=int, default=1000)
    args = p.parse_args()

    for tag in args.ckpts:
        if tag not in CKPTS:
            print(f"[SKIP] {tag}: not in map")
            continue
        model_path = CKPTS[tag]
        if not os.path.exists(model_path):
            print(f"[SKIP] {tag}: missing path {model_path}")
            continue
        out_dir = f"{SUMMER}/eval_results/full/{tag}_vllm"
        out_json = f"{out_dir}/wmt22.json"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n========== {tag} ==========")
        print(f"  model: {model_path}")
        cmd = [
            PYTHON, "-u", f"{SUMMER}/evals/eval_pretrain_translate_vllm.py",
            "--model_path", model_path,
            "--testset", "wmt22", "--exemplar_set", "wmt21",
            "--direction", "both",
            "--num_fewshot", "5",
            "--max_samples", str(args.max_samples),
            "--save_all_samples",
            "--compute_comet",
            "--comet_model_path", "/mnt/data/Summer-data/comet-wmt22-da",
            "--output_path", out_json,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        with open(f"{out_dir}/wmt22.log", "w") as logf:
            ret = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        if ret != 0:
            print(f"[FAIL] {tag} returned {ret} — see {out_dir}/wmt22.log")
        else:
            # Quick summary
            import json
            try:
                d = json.load(open(out_json))['results']
                for direction in ['zh-en', 'en-zh']:
                    r = d[direction]
                    print(f"  {direction}: BLEU {r['bleu']:.2f}  COMET {r.get('comet','-'):.4f}")
            except Exception as e:
                print(f"  parse fail: {e}")


if __name__ == "__main__":
    main()
