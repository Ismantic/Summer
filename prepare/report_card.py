"""把 `eval_results/nanochat/<tag>/` 下的结果排成一张对照表。

    make -C prepare bench-card
    python prepare/report_card.py --only summer05b_s0

## 为什么要有这个

`make bench-nanochat` 每个任务写一个 json,任务多了就没法一眼看全,更没法和
外部靶子比。这里把它们排成一行,并把 **nanochat d20 的公开成绩摆在旁边** ——
它是 560M 参数 / 约 11B token,我们是 524M / 13B,同一档,是目前最贴的外部
参照。

## 不实现 CORE

nanochat 的 base 模型报 CORE(DCLM 那套 22 个任务的中心化准确率)。这里**不
实现它**:那 22 个任务里有若干个 lm_eval 没有等价配置(bigbench 的几个子任务、
jeopardy、coqa 的具体切分),凑一个「近似 CORE」出来会得到一个**看着能比、
其实不可比**的数 —— 那比没有更糟。

所以对照只在两边都真跑了同名任务的地方做:ARC-E / ARC-C / MMLU / GSM8K /
HumanEval。nanochat 那几个是 SFT 之后的数,我们的底座还没 midtrain/SFT,
所以现在这张表看的是**差距的起点**,不是终局。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval_results", "nanochat")

# nanochat d20 的公开成绩(speedrun 档,SFT 之后;base 只公开了 CORE 0.2219)。
# 出处:https://github.com/karpathy/nanochat/discussions/1
# **注意:口径未必一致。** nanochat 用自己的 harness,这几个数是 acc 还是
# acc_norm 没有公开说明。实测 Qwen3-0.6B-Base 上两者能差 0.07(arc_easy
# acc 0.6549 / acc_norm 0.5816,arc_challenge 反过来 0.3345 / 0.3823)——
# 也就是说光靠一个数字对不上号。所以这一列只当**量级参考**,不做精确比较。
NANOCHAT_D20 = {
    "arc_easy": 0.3876,
    "arc_challenge": 0.2807,
    "mmlu": 0.3151,
    "gsm8k": 0.0455,
    "humaneval": 0.0854,
}

# 任务 → (显示名, 该任务里用哪个指标, 随机基线)
#
# **随机基线必须摆出来。** 四选一的 MMLU 随机就是 0.25,报一个 0.26 而不说
# 基线,读者会以为模型「学到了一点」——实际上那是噪声。
TASKS = [
    ("arc_easy",       "ARC-Easy",   "acc,none",            0.25),
    ("arc_easy",       "  ↳ acc_norm", "acc_norm,none",      0.25),
    ("arc_challenge",  "ARC-Chall",  "acc,none",            0.25),
    ("arc_challenge",  "  ↳ acc_norm", "acc_norm,none",      0.25),
    ("mmlu",           "MMLU",       "acc,none",            0.25),
    ("gsm8k",          "GSM8K",      "exact_match,strict-match", 0.0),
    ("ceval-valid",    "C-Eval(中)", "acc,none",            0.25),
]


def load(tag: str) -> dict:
    """→ {task: value}。取不到就不放进去 —— **不要填 0**,那会被当成「测了但很差」。"""
    out = {}
    d = os.path.join(RESULTS, tag)
    for path in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
        try:
            blob = json.load(open(path))
        except Exception:                                      # noqa: BLE001
            continue
        res = blob.get("results") or {}
        for task, val in res.items():
            if isinstance(val, dict):
                out.setdefault(task, val)
    return out


def pick(val: dict, metric: str):
    if metric in val:
        return val[metric]
    # lm_eval 的键会带后缀变体,退一步按前缀找
    base = metric.split(",")[0]
    for k, v in val.items():
        if k.startswith(base) and isinstance(v, (int, float)):
            return v
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="只看某个 tag")
    a = p.parse_args()

    if not os.path.isdir(RESULTS):
        print(f"没有 {RESULTS} —— 先跑 `make -C prepare bench-nanochat "
              f"CKPT=<ckpt> TAG=<名字>`")
        return 1
    tags = sorted(t for t in os.listdir(RESULTS)
                  if os.path.isdir(os.path.join(RESULTS, t))
                  and (a.only is None or t == a.only))
    if not tags:
        print(f"{RESULTS} 下没有结果目录")
        return 1

    data = {t: load(t) for t in tags}
    w = max(11, max(len(t) for t in tags) + 1)

    print("nanochat 口径的报告卡")
    print(f"  外部靶子 nanochat d20:560M 参数 / 约 11B token(我们 524M / 13B)")
    print(f"  它那几个数是 **SFT 之后**的;底座还没 midtrain/SFT 的话差距是起点不是终局\n")
    hdr = f"{'任务':<12}{'随机':>7}{'d20':>8}" + "".join(f"{t:>{w}}" for t in tags)
    print(hdr)
    print("-" * len(hdr))
    for task, name, metric, rnd in TASKS:
        nc = None if name.startswith("  ↳") else NANOCHAT_D20.get(task)
        row = f"{name:<12}{rnd:>7.2f}{(f'{nc:.4f}' if nc else '—'):>8}"
        for t in tags:
            v = pick(data[t].get(task, {}), metric)
            row += f"{(f'{v:.4f}' if isinstance(v, (int, float)) else '—'):>{w}}"
        print(row)
    print()
    for t in tags:
        got = [k for k, _, _, _ in TASKS if k in data[t]]
        miss = [k for k, _, _, _ in TASKS if k not in data[t]]
        print(f"  {t}: 有 {len(got)} 项" + (f",缺 {miss}" if miss else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
