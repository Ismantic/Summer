"""按 nanochat 的协议做选择题评测:渲染成 MC prompt,让模型输出**字母**。

    make -C prepare mc-eval CKPT=<ckpt> TAG=<名字>
    python prepare/mc_eval.py --model_path output/summer05b_sft_v2 --tasks arc_easy,mmlu

## 为什么要再写一套评测

`benchmark.py` 走 lm_eval,它的 `acc` / `acc_norm` 是**比各选项正文的对数似然**:
不给模型看字母,分别算「问题+选项A正文」「问题+选项B正文」的似然,取最大。
nanochat 完全不是这么做的(`tasks/arc.py` + `scripts/chat_eval.py`):

```
Multiple Choice question: {问题}
- {选项正文}={字母}
- ...

Respond only with the letter of the correct answer.
```

然后取 assistant 位置的 logits,**只在候选字母的 token 上 argmax**,比字母。

所以之前纠结 nanochat 那几个数是 `acc` 还是 `acc_norm` 是问错了 —— 两个都不是。
不实现这套协议,就没法回答「英文指标赶超了没有」。

**两套协议量的不是一回事,这才是要它的真正理由:**

| | 量什么 |
|---|---|
| 似然打分(lm_eval) | 底座知识。SFT 前后都能测,但看不见 SFT 加了什么 |
| 字母 MC(这里) | **能不能听懂指令**。这正是 midtrain + SFT 要加的能力 |

整条 chat 线做完之后,原先没有任何一个指标在量它要加的那个能力。

## 读数时必须知道的两件事

1. **nanochat 的 SFT 数据里就有 ARC 的 train split**(`chat_sft.py:88-89`,
   ARC-Easy 2.3K + ARC-C 1.1K)。它的 ARC 成绩有相当一部分是**同分布的格式
   训练** —— 它学过「MC 题面 → 输出字母」这个格式本身。我们的混比里 ARC 是零。
   所以这里的差距要分开看:格式会不会 / 知识有没有。
2. **底座模型(没 midtrain/SFT)在这个协议下天然吃亏**,它没见过对话格式。
   照样测、照样报,但不能拿它和 SFT 过的模型比高低。

## 实现上的两个坑

- **字母必须是单 token,且前面不能有空格。** `" A"` 和 `"A"` 在 BBPE / piece
  词表里是不同 id。nanochat 特意把字母放在选项**之后**(`={字母}`)就是为了
  让 prompt 里的字母和 assistant 要输出的字母是同一个 token。这里 assert 住。
- **右侧 padding 是安全的,左侧不是。** `src/model.py` 只做因果注意力、不接受
  attention_mask;取 logits 的位置是每条序列**真实的最后一个位置**,它看不见
  后面的 pad。左填充会让位置编码整体错位。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# **评测不该依赖网络。** 四个数据集都在本地 HF 缓存里,但 load_dataset 每次仍会
# 去请求元数据;这台机器过代理,代理一抖就整个卡住 —— 实测卡了 15 分钟,GPU
# 利用率 0%、进程 sleeping,而且脚本在任务结束前不打任何进度,**看起来和正常跑
# 没区别**。离线模式下缓存缺失会立刻报错,那比静默挂住好得多。
# 要下新数据集时用 SUMMER_MC_ONLINE=1 关掉。
if not os.environ.get("SUMMER_MC_ONLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# **渲染函数在 `prepare/tasks/common.py`,不在这里。** 训练和评测必须用同一份 ——
# 各写一份的话,改了训练侧忘了改评测侧,量到的就是「换了提示格式认不认得」,
# 而不是「会不会做题」,**而且两边都不会报错**。
#
#   nanochat  `Multiple Choice question: …\n- {选项}={字母}\n\nRespond only with…`
#   ours      `{问题}\n{A. 选项}`  —— v2~v5 的 midtrain 是用这个训的
#
# v6 起训练侧统一改用 nanochat 的渲染(对齐基准),所以 **v6 之后的数要用
# `--render nanochat` 读**;v2~v5 的数要用 `--render ours`。两者不可比。
from prepare.tasks.common import render_mc, render_mc_ours       # noqa: E402

RENDERERS = {"nanochat": render_mc, "ours": render_mc_ours}


# task → (加载函数, 说明)。每个返回 [(question, letters, choices, answer_letter)]
def _arc(subset: str):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    out = []
    for row in ds:
        letters = row["choices"]["label"]
        if row["answerKey"] not in letters:      # 少数行的 answerKey 是数字
            continue
        out.append((row["question"], letters, row["choices"]["text"], row["answerKey"]))
    return out


def _mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    L = ["A", "B", "C", "D"]
    return [(r["question"], L, r["choices"], L[r["answer"]]) for r in ds]


def _ceval():
    """C-Eval 的 validation split(test 没有公开答案)。中文侧的对照。"""
    from datasets import load_dataset
    subjects = ["computer_network", "operating_system", "high_school_chinese",
                "high_school_history", "middle_school_geography", "law",
                "physician", "teacher_qualification", "accountant", "logic"]
    L = ["A", "B", "C", "D"]
    out = []
    for s in subjects:
        for r in load_dataset("ceval/ceval-exam", s, split="val"):
            out.append((r["question"], L, [r["A"], r["B"], r["C"], r["D"]], r["answer"]))
    return out


TASKS = {
    "arc_easy":      lambda: _arc("ARC-Easy"),
    "arc_challenge": lambda: _arc("ARC-Challenge"),
    "mmlu":          _mmlu,
    "ceval":         _ceval,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tasks", default="arc_easy,arc_challenge,mmlu,ceval")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_problems", type=int, default=None, help="调试用,正式跑别设")
    p.add_argument("--render", default="nanochat", choices=list(RENDERERS),
                   help="提示格式。nanochat=可比它的数;ours=我们 midtrain 训的那套")
    p.add_argument("--output_path", default=None)
    a = p.parse_args()

    from prepare.tokenizer import PieceTokenizerWrapper
    from src.model import Qwen3ForCausalLM

    tok = PieceTokenizerWrapper(a.model_path)
    model = Qwen3ForCausalLM.from_pretrained(a.model_path).cuda().eval()
    pad = tok.pad_token_id

    # 字母 → token id,并 assert 单 token(见文档字符串「实现上的两个坑」)
    letter_id: dict[str, int] = {}

    def lid(letter: str) -> int:
        if letter not in letter_id:
            enc = tok.encode(letter, add_special_tokens=False)
            assert len(enc) == 1, f"字母 {letter!r} 不是单 token:{enc}"
            letter_id[letter] = enc[0]
        return letter_id[letter]

    results = {}
    for name in a.tasks.split(","):
        name = name.strip()
        if name not in TASKS:
            print(f"  跳过未知任务 {name}")
            continue
        print(f"  {name:14s} 读数据…", flush=True)   # 卡住时能看出卡在哪
        rows = TASKS[name]()
        if a.max_problems:
            rows = rows[:a.max_problems]

        # 先全部编码,再按长度分批 —— 同批长度接近,padding 少
        encoded = []
        for question, letters, choices, answer in rows:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": RENDERERS[a.render](question, letters, choices)}],
                tokenize=True, add_generation_prompt=True)
            encoded.append((ids, letters, answer))
        encoded.sort(key=lambda x: len(x[0]))

        # **把两种失败分开。** 准确率贴着随机线有两种可能,含义完全不同:
        #   格式失败 —— 模型压根不想输出字母(约束到字母上取 argmax 等于瞎猜)
        #   知识失败 —— 它确实输出字母,只是选错了
        # 判据:不加约束、在**全词表**上取 argmax,看首选是不是候选字母之一。
        # 只看被约束后的准确率永远分不出这两者。
        passed = free_is_letter = 0
        for i in range(0, len(encoded), a.batch_size):
            chunk = encoded[i:i + a.batch_size]
            width = max(len(ids) for ids, _, _ in chunk)
            # **右侧 padding**:取 logits 的位置是各自真实的最后一个 token
            batch = torch.tensor([ids + [pad] * (width - len(ids))
                                  for ids, _, _ in chunk], device="cuda")
            with torch.no_grad():
                logits = model(batch)
            for j, (ids, letters, answer) in enumerate(chunk):
                cand = [lid(x) for x in letters]
                row = logits[j, len(ids) - 1].float()
                pick = letters[row[cand].argmax().item()]
                passed += int(pick == answer)
                free_is_letter += int(row.argmax().item() in cand)

        acc, fmt = passed / len(encoded), free_is_letter / len(encoded)
        results[name] = {"acc": acc, "free_argmax_is_letter": fmt, "n": len(encoded)}
        print(f"  {name:14s} [{a.render}] acc {acc:.4f}   ({passed}/{len(encoded)})"
              f"   自由首选是字母 {fmt:.4f}")

    if a.output_path:
        os.makedirs(os.path.dirname(a.output_path), exist_ok=True)
        with open(a.output_path, "w") as fh:
            json.dump({"model_path": a.model_path, "protocol": "letter-mc",
                       "render": a.render,
                       "results": results}, fh, indent=1, ensure_ascii=False)
        print(f"  → {a.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
