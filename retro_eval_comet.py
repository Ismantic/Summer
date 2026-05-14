"""Retroactively run BLEU + COMET on existing ckpts.

After v13 is done, run this once on key ckpts to get COMET numbers in addition
to BLEU. Output goes to eval_results/full/<tag>/wmt22.json (overwrites with
augmented content).
"""
import argparse
import os
import subprocess
import sys

SUMMER = "/home/tfbao/Shiyu/Summer"
PYTHON = "/home/tfbao/.venv/bin/python"

# Map: ckpt tag -> model_path
CKPTS = {
    "base":           "/home/tfbao/new/Qwen3-0.6B-Base-new-tok",
    "v8_p1":          f"{SUMMER}/output/phase1_ckpt_v8",
    "v10_p2":         f"{SUMMER}/output/phase2_ckpt_v10_aurora_anneal/checkpoint-2000",
    "v11_p2":         f"{SUMMER}/output/phase2_ckpt_v11_aurora_anneal/checkpoint-2000",
    "v12_p1":         f"{SUMMER}/output/phase1_ckpt_v12/checkpoint-5000",
    "v12_p2":         f"{SUMMER}/output/phase2_ckpt_v12_aurora_anneal/checkpoint-2000",
    "v13_p1":         f"{SUMMER}/output/phase1_ckpt_v13/checkpoint-8000",
    "v13_p2":         f"{SUMMER}/output/phase2_ckpt_v13_aurora_anneal/checkpoint-2000",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS.keys()),
                   help="Tags to evaluate; default all")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    for tag in args.ckpts:
        if tag not in CKPTS:
            print(f"[SKIP] {tag}: not in CKPTS map")
            continue
        model_path = CKPTS[tag]
        if not os.path.exists(model_path):
            print(f"[SKIP] {tag}: model path missing ({model_path})")
            continue
        out_dir = f"{SUMMER}/eval_results/full/{tag}_comet"
        out_json = f"{out_dir}/wmt22.json"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n========== {tag} ==========")
        print(f"  model: {model_path}")
        print(f"  out:   {out_json}")
        cmd = [
            PYTHON, "-u", f"{SUMMER}/eval_pretrain_translate.py",
            "--model_path", model_path,
            "--testset", "wmt22", "--exemplar_set", "wmt21",
            "--direction", "both",
            "--num_fewshot", "5",
            "--max_samples", str(args.max_samples),
            "--batch_size", str(args.batch_size),
            "--save_all_samples",
            "--compute_comet",
            "--comet_model_path", "/mnt/data/Summer-data/comet-wmt22-da",
            "--output_path", out_json,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        ret = subprocess.call(cmd, env=env)
        if ret != 0:
            print(f"[FAIL] {tag} returned {ret}")


if __name__ == "__main__":
    main()
