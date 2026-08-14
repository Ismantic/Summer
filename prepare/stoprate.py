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

## prompt 从哪来:**留出的真实指令**(2026-08-14 换过一次,原因见下)

    en   SmolTalk 第 100,000 条之后的首轮用户提问(训练只用了前 10 万)
    zh   Firefly 第 35,000 条之后、CJK 占比 > 0.3 的提问(训练只用了前 3.5 万)

两边都是**真实的开放式指令**,而且都在训练用量之外 —— 留出的。

### 上一版是坏的:从选择题题面去掉选项

原先是拿 ARC-Easy test 和 C-Eval val 的**题面去掉选项**当开放问答。
构造可复现(手写的那 20 条随机器重启丢过一次),但**只验了「能不能离线跑通」,
没验「去掉选项之后还成不成题」**:

    en    2/100 含「指向被删选项」的标志
    zh  **100/100**  —— 「下列各句中,没有语病的一句是____」「选填哪项最恰当____」

**全部 100 条中文 prompt 都无法回答。** 模型答不出来就退化成复读,于是
2026-08-14 之前报的每一个中文停止率(v3 29% / v4 69% / v5 73% / v6 37% /
v7 57%,以及三版预演)都是在无效 prompt 上量的,**绝对值没有意义**。
跨版本的方向可能还成立(同一批 prompt),但「中英不对称」那个结论至少一部分
是这个缺陷造出来的。

英文侧碰巧成立(ARC 的「Which piece of safety equipment is used to…」有真实
答案),所以英文的数受影响小 —— 但为了两侧同源,一起换掉了。

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


#: 训练用量 —— prompt 从这之后取,保证是留出的。改混比时这两个数要跟着改。
SMOLTALK_TRAINED = 100000
FIREFLY_TRAINED = 35000


def _cjk(t: str) -> float:
    return sum(1 for c in t if "\u4e00" <= c <= "\u9fff") / max(len(t), 1)


def build_prompts(n: int) -> list[tuple[str, str]]:
    """→ [(lang, prompt)]。中英各 n/2,顺序固定,**全部取自训练用量之外**。"""
    import itertools
    from prepare.tasks.smoltalk import SmolTalk
    from prepare.tasks.zh_instruct import FireflyZH
    half = n // 2
    out: list[tuple[str, str]] = []
    for turns in itertools.islice(iter(SmolTalk()), SMOLTALK_TRAINED, None):
        q = turns[0][1].strip()
        if 20 <= len(q) <= 400:          # 太短没信息,太长是贴代码
            out.append(("en", q))
        if len(out) >= half:
            break
    zh: list[tuple[str, str]] = []
    for turns in itertools.islice(iter(FireflyZH()), FIREFLY_TRAINED, None):
        q = turns[0][1].strip()
        if 10 <= len(q) <= 400 and _cjk(q) > 0.3:   # Firefly 里混了英文条目
            zh.append(("zh", q))
        if len(zh) >= half:
            break
    return out + zh


# ---------------------------------------------------------------- 长回答判据
#
# **停止率单独看是有缺陷的**:一个把所有问题都用 36 个 token 打发掉的模型会拿到
# 很高的停止率,却毫无用处 —— 而 v7 的中文正是这样(成功样本长度中位 36,
# 英文是 276)。所以要配一个「答得够长、没崩、还收了尾」的判据一起看。
#
#     长回答成功 = 生成 ≥ min_len token  且  复读率 < 0.2  且  自然停止
#
# 三个条件缺一不可:只要长度会奖励复读到烧满预算的;只要复读率会奖励一句话答完的;
# 只要停止会奖励 36 token 打发一切的。
def rep_ratio(text: str, n: int = 24) -> float:
    """粗略复读率:长度 n 的子串里重复出现的占比。1.0 = 完全复读。"""
    if len(text) < n * 2:
        return 0.0
    g = [text[i:i + n] for i in range(0, len(text) - n, n)]
    return 1.0 - len(set(g)) / max(len(g), 1)


def longform_ok(text: str, n_tok: int, stopped: bool,
                min_len: int = 150, max_rep: float = 0.2) -> bool:
    return stopped and n_tok >= min_len and rep_ratio(text) < max_rep


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
    longok = {"en": 0, "zh": 0}
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
        txt = tok.decode(list(gen), skip_special_tokens=True)
        longok[lang] += int(longform_ok(txt, len(gen), nat))

    tot = sum(stopped.values())
    p = tot / len(prompts)
    se = (p * (1 - p) / len(prompts)) ** 0.5
    avg = sum(sum(v) for v in lens.values()) / len(prompts)
    tag = os.path.basename(ckpt.rstrip("/"))
    for lang in ("en", "zh"):
        k = len(lens[lang])
        if k:
            print(f"  {tag}  [{lang}] 自然停止 {stopped[lang]}/{k} = "
                  f"{stopped[lang] / k:.1%}   **长回答成功 {longok[lang]}/{k} = "
                  f"{longok[lang] / k:.1%}**   平均生成 {sum(lens[lang]) / k:.0f} token")
    print(f"  {tag}  长回答成功合计 {sum(longok.values())}/{len(prompts)} = "
          f"{sum(longok.values()) / len(prompts):.1%}"
          f"   (≥150 token、复读 <0.2、且自然停止)")
    print(f"  {tag}  自然停止 {tot}/{len(prompts)} = {p:.1%} "
          f"(标准误 ±{se * 100:.1f} 点)   平均生成 {avg:.0f} token   预算 {budget}")
    if disagree:
        print(f"  !! finish_reason 和长度启发式在 {disagree} 条上不一致 —— "
              f"读数前先查清楚,别直接用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
