"""自由生成的停止率 —— chat 线的**主判据**。

    python prepare/stoprate.py <ckpt> [n=200] [budget=600]

## 为什么它是主判据而不是 loss

v4 那一版(midtrain 打包改连续流)在中文停止率上 +40 点、格式跟随 +32 点,
而 loss 侧完全看不出改进(数据难度解释了全部差值,同一数据上各赢主场)。
**val loss 会把那一版判成平手** —— 详见 `docs/POSTTRAIN.md`。

## 为什么 n=200

之前用 n=20。p≈0.65 时标准误 **10.7 个点**,一条 prompt 占 5 个点 —— v2 的 65%
和 v3 的 70% 差不到一个标准误,那不是结论。n=200 把标准误压到 3.4。
**脚本自己把标准误打出来**,省得下次又拿噪声当结果。

比两个版本时还要再退一步:**两个 ±3.3 相减,差值的标准误是 4.6** ——
v4 的 67.0% 和 v5 的 71.5% 差 4.5 点,仍然不算数。

## prompt 从哪来:可复现的构造,不手写

原来那 20 条是手写的、只存在临时目录里,机器一重启就没了(2026-08-13 真的
发生了)。**手写 prompt 集没法复现也没法审。** 改成从两个已在本地缓存的评测集
构造:ARC-Easy test 的问句(en)+ C-Eval val 的问句(zh),**去掉选项只留问句**,
于是变成开放问答。确定、离线、中英各半。

**这批数字不能和历史上的 65% / 70% / nanochat 的 95% 直接比** —— prompt 集换了。
要横向比就得用同一批 prompt 重测每个 ckpt。

## 预算为什么是 600

SFT 答案中位 338 token、平均 374,只有 20.6% ≤200。预算 200 时天花板就是
20.6%,测出来的 10% 是「被截断」不是「不会停」。
**量停止率之前先看训练答案的长度分布。**
"""
from __future__ import annotations

import os
import sys

# 和 mc_eval.py 同样的理由:评测不该依赖网络,代理一抖就静默卡住(实测卡过 15 分钟)
if not os.environ.get("SUMMER_MC_ONLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_prompts(n: int) -> list[tuple[str, str]]:
    """→ [(lang, prompt)]。中英各 n/2,顺序固定。"""
    from datasets import load_dataset
    half = n // 2
    out: list[tuple[str, str]] = []
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    out += [("en", r["question"]) for r in list(ds)[:half]]
    zh: list[tuple[str, str]] = []
    for s in ["high_school_chinese", "high_school_history", "law", "logic",
              "teacher_qualification", "physician", "accountant",
              "middle_school_geography", "operating_system", "computer_network"]:
        if len(zh) >= half:
            break
        for r in load_dataset("ceval/ceval-exam", s, split="val"):
            zh.append(("zh", r["question"]))
            if len(zh) >= half:
                break
    return out + zh[:half]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 600

    from prepare.tokenizer import PieceTokenizerWrapper
    tok = PieceTokenizerWrapper(ckpt)
    stop = tok.stop_token_ids

    print(f"  构造 prompt(n={n})…", flush=True)
    prompts = build_prompts(n)
    ids = [tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=True, add_generation_prompt=True)
           for _, p in prompts]

    from vllm import LLM, SamplingParams
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=0.85,
              skip_tokenizer_init=True, trust_remote_code=True, enforce_eager=False)
    # 贪心。**stop_token_ids 给全集**(<end> + <eos>),理由见 tokenizer.py
    outs = llm.generate(
        [{"prompt_token_ids": x} for x in ids],
        SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=stop))

    stopped = {"en": 0, "zh": 0}
    lens: dict[str, list[int]] = {"en": [], "zh": []}
    disagree = 0
    for (lang, _), o in zip(prompts, outs):
        out = o.outputs[0]
        gen = out.token_ids
        # **权威依据是 finish_reason**:"stop" = 撞到停止符,"length" = 烧完预算
        nat = out.finish_reason == "stop"
        # 启发式对照:停止符可能被算进 token_ids(取决于 vLLM 版本)。两者不一致
        # 时计数报出来 —— 这个指标已经因为判据错过一次,不再信任任何单一信号。
        heur = len(gen) < budget or (gen and gen[-1] in stop)
        disagree += int(nat != heur)
        stopped[lang] += int(nat)
        lens[lang].append(len(gen))

    tot = sum(stopped.values())
    p = tot / len(prompts)
    se = (p * (1 - p) / len(prompts)) ** 0.5
    avg = sum(sum(v) for v in lens.values()) / len(prompts)
    tag = os.path.basename(ckpt.rstrip("/"))
    for lang in ("en", "zh"):
        k = len(lens[lang])
        if k:
            print(f"  {tag}  [{lang}] 自然停止 {stopped[lang]}/{k} = "
                  f"{stopped[lang] / k:.1%}   平均生成 {sum(lens[lang]) / k:.0f} token")
    print(f"  {tag}  自然停止 {tot}/{len(prompts)} = {p:.1%} "
          f"(标准误 ±{se * 100:.1f} 点)   平均生成 {avg:.0f} token   预算 {budget}")
    if disagree:
        print(f"  !! finish_reason 和长度启发式在 {disagree} 条上不一致 —— "
              f"读数前先查清楚,别直接用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
