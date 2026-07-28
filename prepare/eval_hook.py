"""训练途中的翻译评测 —— 给 `src/train.py --inline_eval_cmd` 用。

## 为什么不看 loss 就够了

从零预训练的 loss 从头到尾都在降,**降到哪儿算能翻译,曲线上看不出来**。
1.9 和 1.8 之间可能什么都没变,也可能正好跨过了 few-shot 涌现的门槛。
而这个模型只对翻译负责,那就直接每隔一段测一次 WMT22 的 5-shot BLEU,
把它当成主曲线。

## 开销

200 条 × 双向 + vLLM 起停 ≈ 4 分钟。每 2000 步(约 1.8 小时)测一次,
占总时间约 4%。不算 COMET —— COMET 要再加载一个 2.3GB 的模型,而中途看
趋势用 BLEU 够了;收尾的正式评测再用 `make -C prepare trans` 上 COMET。

## 后端

**全程 vLLM,不换。** 两个后端的数字不能混着比(实测 lambada 上差 2.2 个点),
中途曲线要是混了后端,趋势就是假的。

## 用法

    python src/train.py ... \
        --save_steps 2000 \
        --inline_eval_cmd "python prepare/eval_hook.py --ckpt {ckpt} --step {step} \
                           --curve output/summer05b_s0/bleu_curve.jsonl"

`{ckpt}` 和 `{step}` 由 train.py 替换。模型和优化器在这条命令跑之前已经被
挪到 CPU,显存是空的 —— 所以 vLLM 能起得来。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="要评的 checkpoint 目录")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--curve", required=True, help="追加 BLEU 的 jsonl")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--testset", default="wmt22")
    p.add_argument("--exemplar_set", default="wmt21")
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()

    tmp_json = os.path.join(a.ckpt, "trans_eval.json")
    cmd = [a.python, os.path.join(HERE, "translate.py"),
           "--model_path", a.ckpt,
           "--testset", a.testset, "--exemplar_set", a.exemplar_set,
           "--direction", "both",
           "--num_fewshot", str(a.num_fewshot),
           "--max_samples", str(a.max_samples),
           "--gpu_mem_util", str(a.gpu_mem_util),
           "--output_path", tmp_json]

    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0 or not os.path.exists(tmp_json):
        # **失败要留痕。** 只打到 stdout 的话,几天后回头看曲线只会看到
        # 「这一段没有点」,分不清是没测还是测了没变。
        rec = {"step": a.step, "error": f"translate.py 退出码 {r.returncode}",
               "elapsed_s": round(time.time() - t0, 1)}
        _append(a.curve, rec)
        print(f"[eval_hook] step {a.step} 评测失败(退出码 {r.returncode})", flush=True)
        return 1

    out = json.load(open(tmp_json))
    rec = {"step": a.step, "n_samples": a.max_samples,
           "num_fewshot": a.num_fewshot, "testset": a.testset,
           "backend": "vllm", "elapsed_s": round(time.time() - t0, 1)}
    for d, res in out["results"].items():
        rec[f"bleu_{d}"] = round(res["bleu"], 2)
    _append(a.curve, rec)

    line = "  ".join(f"{k} {v}" for k, v in rec.items() if k.startswith("bleu_"))
    print(f"[eval_hook] step {a.step} | {line} | {rec['elapsed_s']:.0f}s", flush=True)
    return 0


def _append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
