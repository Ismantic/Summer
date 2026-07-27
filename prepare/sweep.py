"""批量评测多个 checkpoint,并把结果汇成对照表。

    # 跑:每个 ckpt 依次做翻译 + mono 六任务
    python prepare/sweep.py run --ckpt base=/path/to/base v18=output/phase2_ckpt_v18_tie

    # 只汇总已有结果(不跑评测)
    python prepare/sweep.py table --tags base v18_p1 v18_p2 v18_p2_tie

产物落在:

    eval_results/full/<tag>/<task>/result.json          mono 六任务
    eval_results/translate_<testset>/<tag>_*.json       BLEU + COMET

## 没有多卡代码路径

旧版本的 `retro_eval_*.py` 有 `--gpus 0 1` 把 ckpt 分到两张卡上 —— 那是
2× A6000 时代的东西。本机只有一张 4090,那套分片是**跑不到的死代码**,
一并删掉(与 BERTc「没有多卡代码路径」的做法一致)。

## 数字之间差多少才算数

**vLLM 的贪心解码不可复现。** 同一 ckpt、同一条命令跑 6 次实测:

    zh-en BLEU  range 0.1033      en-zh BLEU  range 0.1337
    zh-en COMET range 0.0010      en-zh COMET range 0.0005

所以表里 **0.1 量级的 BLEU 差是噪声,不能当结论**;COMET 稳两个数量级,
用它比更可靠。详见 docs/WHY.md。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "eval_results" / "full"

MONO_TASKS = "lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5"
# 每个任务优先看哪个指标
PREFER = {
    "lambada_openai": ["acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "ceval-valid": ["acc,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
}


# ---------------------------------------------------------------- 读结果

def read_mono(tag: str, task: str):
    """`<tag>_vllm/` 优先 —— 后端写在目录名里是「不能混后端」那条约定的产物。

    历史上还有 `<tag>_tf/`(transformers 后端)。**两种后端的数字不能放进
    同一张表比** —— under-trained ckpt 上 arc_challenge 能差 ±10%,曾经
    因此得出过错误结论。这里只认 vLLM,找不到就留空,不去凑 _tf 的数。
    """
    for d in (FULL / f"{tag}_vllm", FULL / tag):
        f = d / task / "result.json"
        if f.is_file():
            res = json.loads(f.read_text()).get("results", {}).get(task, {})
            for m in PREFER.get(task, ["acc,none"]):
                if m in res:
                    return res[m]
    return None


def read_translate(tag: str, testset: str = "wmt22"):
    """返回 {方向: (bleu, comet)}。文件名有几种历史写法,都试一遍。"""
    d = ROOT / "eval_results" / f"translate_{testset}"
    for pat in (f"{tag}_vllm_1000_comet.json", f"{tag}_*comet.json",
                f"{tag}_vllm*.json", f"{tag}.json"):
        hits = sorted(glob.glob(str(d / pat)))
        if hits:
            res = json.loads(Path(hits[-1]).read_text()).get("results", {})
            return {k: (v.get("bleu"), v.get("comet")) for k, v in res.items()}
    return {}


# ---------------------------------------------------------------- 跑评测

def run_one(tag: str, model_path: str, py: str, testset: str,
            comet_path: str | None, skip_mono: bool, skip_trans: bool) -> bool:
    ok = True
    if not skip_trans:
        out = ROOT / "eval_results" / f"translate_{testset}" / f"{tag}_vllm_1000_comet.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [py, str(ROOT / "prepare" / "translate.py"),
               "--model_path", model_path, "--testset", testset,
               "--exemplar_set", "wmt21", "--direction", "both",
               "--num_fewshot", "5", "--max_samples", "1000",
               "--save_all_samples", "--output_path", str(out)]
        if comet_path:
            # --compute_comet 必须配 --save_all_samples,否则脚本只留 5 条样本,
            # WARN 一句然后跳过 COMET(不报错)
            cmd += ["--compute_comet", "--comet_model_path", comet_path]
        print(f"\n[{tag}] 翻译 …")
        ok &= subprocess.run(cmd, cwd=ROOT).returncode == 0

    if not skip_mono:
        out = FULL / tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{tag}] mono 六任务 …")
        ok &= subprocess.run(
            [py, str(ROOT / "prepare" / "benchmark.py"),
             "--model_path", model_path, "--tasks", MONO_TASKS,
             "--output_dir", str(out)], cwd=ROOT).returncode == 0
    return ok


# ---------------------------------------------------------------- 出表

def print_table(tags: list[str], testset: str) -> None:
    tasks = list(PREFER)
    w = max(len(t) for t in tags) + 2

    print(f"\n=== mono({testset} 之外的六个任务)===")
    print(f"{'tag':<{w}}" + "".join(f"{t[:12]:>14}" for t in tasks))
    base_row = None
    for tag in tags:
        vals = [read_mono(tag, t) for t in tasks]
        if base_row is None:
            base_row = vals
        cells = "".join(f"{v:>14.4f}" if v is not None else f"{'—':>14}"
                        for v in vals)
        print(f"{tag:<{w}}{cells}")

    print(f"\n=== 翻译({testset},1000 样本 5-shot)===")
    print(f"{'tag':<{w}}{'zh-en BLEU':>12}{'zh-en COMET':>13}"
          f"{'en-zh BLEU':>12}{'en-zh COMET':>13}")
    for tag in tags:
        r = read_translate(tag, testset)
        ze, ez = r.get("zh-en", (None, None)), r.get("en-zh", (None, None))

        def f(v, n=4):
            return f"{v:.{n}f}" if v is not None else "—"
        print(f"{tag:<{w}}{f(ze[0], 2):>12}{f(ze[1]):>13}"
              f"{f(ez[0], 2):>12}{f(ez[1]):>13}")

    print("\n注:vLLM 贪心解码不可复现,BLEU 跑间 range 约 0.1 "
          "—— 这个量级的差是噪声,不是结论。COMET 稳两个数量级。")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑评测")
    r.add_argument("--ckpt", nargs="+", required=True,
                   metavar="TAG=PATH", help="如 v18=output/phase2_ckpt_v18_tie")
    r.add_argument("--python", default=sys.executable,
                   help="跑评测的解释器(要 vllm,通常在另一个 venv)")
    r.add_argument("--testset", default="wmt22")
    r.add_argument("--comet_model_path", default=None)
    r.add_argument("--skip_mono", action="store_true")
    r.add_argument("--skip_trans", action="store_true")

    t = sub.add_parser("table", help="只汇总已有结果")
    t.add_argument("--tags", nargs="+", required=True)
    t.add_argument("--testset", default="wmt22")

    a = p.parse_args()

    if a.cmd == "table":
        print_table(a.tags, a.testset)
        return 0

    pairs = []
    for spec in a.ckpt:
        if "=" not in spec:
            p.error(f"--ckpt 要写成 TAG=PATH,收到 {spec}")
        tag, path = spec.split("=", 1)
        pairs.append((tag, path))

    comet = a.comet_model_path
    if comet is None and not a.skip_trans:
        sys.path.insert(0, str(ROOT))
        from data import source
        d = source.get("comet").dir()
        comet = str(d) if d.exists() else None
        if comet is None:
            print("COMET 模型不在,只出 BLEU(要的话 `make -C data download-comet`)")

    ok = True
    for tag, path in pairs:
        ok &= run_one(tag, path, a.python, a.testset, comet,
                      a.skip_mono, a.skip_trans)
    print_table([t for t, _ in pairs], a.testset)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
