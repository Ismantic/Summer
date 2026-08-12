"""对话数据的预编码 —— midtrain 与 SFT 两个阶段共用。

    python -m prepare.chat --mix midtrain --output output/midtrain_1024
    python -m prepare.chat --mix sft      --output output/sft_1024

产出 `<out>.pt`(int32 `[N, seq_len]`,和预训练同格式)加 `<out>.mask.pt`
(uint8 同形状,1 = 这个位置要算 loss)。`src/train.py --loss_mask <path>` 读它。

## 为什么需要这一层(而不是直接 SFT)

预训练语料全是纯文本,`<user>` / `<assistant>` / `<system>` 三个 token 从没作为
**输入**出现过。实测 Summer-0.5B-S1 的权重里,它们和 `<pad>` 的余弦相似度
**全是 +1.0000** —— 在模型眼里是同一个向量(普通 token 之间平均 0.285)。
它们的范数从 0.64 长到 1.66,只是因为 embedding 与 lm_head 绑权重、每步都在
把「不该预测的 token」的 logit 往下压,四个受到完全相同的推力。

所以直接上 SFT,等于让模型在一个 epoch 里同时学任务和学这三个分隔符。
nanochat 把这件事拆成 midtrain(先用对话格式的数据把模板喂熟,整段算 loss)
和 SFT(只在助手那部分算 loss),这里照它分。

## 打包,不 padding

**自写的模型只做 is_causal 的因果注意力,不接受 attention_mask**
(见 src/model.py 与 src/train.py:202)。padding 进来会被当成真 token 参与
注意力,算出来的东西是错的 —— 而且不报错。

所以这里把多条对话首尾相接打包成定长 seq_len,完全没有 padding。代价是
一条序列里可能横跨两段对话,因果注意力会让后半段"看到"前半段。nanochat
接受这个代价,我们也接受:边界处的噪声远小于 padding 算错的代价,而且
每条序列平均只有一到两个边界。

## 掩码

midtrain:整段算 loss(mask 全 1)。目的是让模型见过这个格式,不是精调答案。
SFT:只在助手回复 + 其后的 eos 上算 loss。用户那段不算 —— 否则模型会去学
「怎么生成问题」,那不是我们要的能力,而且会稀释真正的监督信号。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import source as _source                              # noqa: E402
from prepare.tokenizer import (PieceTokenizerWrapper,           # noqa: E402
                               resolve_assets)

# ------------------------------------------------------------------ 配比
#
# 参照 nanochat 的 midtrain:对话为主,掺一点多选题和数学。
#
# 多选题那份不是为了"学知识",是为了**让模型见过「四个选项选一个」这种题型**
# —— 否则 MMLU / ARC / C-Eval 评测时连输出格式都不会,分数低到无法反映真实能力。
#
# 中英对半:底座是中英 50:50 训出来的,只喂英文会浪费掉一半能力。
# **配比受制于中文侧的供给。** 实测各源本地可出的 token:
#
#   SmolTalk(en)      90.2M(本地 1/4 分片,全量约 360M)
#   COIG_CQIA(zh)      6.5M  ← **这个数是错的,见下**
#   MMLU_AuxTrain      26.6M
#   GSM8K_Train         1.3M
#
# 中文对话料比英文少一个量级。硬凑 50:50 就得把中文重复十几遍,那会在中文上
# 过拟合 —— 而这种偏斜在 loss 曲线上看不出来。所以这一版**按中文侧的实际供给
# 定总量**(约 35M),而不是先定总量再配比。
#
# 中英不是严格 50:50:多选题(MMLU)和数学(GSM8K)本身是英文的,它们进来是为了
# 教**题型**,不是教英文。纯对话那部分 SmolTalk : COIG_CQIA = 1 : 1。
#
# ## 那个 6.5M 是幸存者偏差,真实是约 50M
#
# 估算时我用「encode_turns 返回的 token 总数 ÷ 条数」当平均长度,但这个函数
# **超长的会返回空**,于是分母算了全部、分子只算了留下来的短样本 —— 越丢越显得
# 平均短,估出 145 token/条。
#
# 实测 COIG-CQIA 的真实分布:**中位数 1126 token**,p90 2093,p99 2459。
# seq_len 1024 下只有 50% 能整条装下,另一半按「单轮超长」被整条丢掉 ——
# 而切段策略对它无效:它是单轮长回答(考试题解析、知乎长答),切段只在多轮之间
# 切,单轮切不了。
#
# 所以中文侧的实际供给是约 50M 而不是 6.5M,上面这个配比的前提不成立。
# **当前这份 midtrain 数据仍然可用**(34.8M、配比精确、掩码正确,中文那 6.65M
# 是真实完整样本,只是偏短的那一半),先拿它跑通链路;要不要重编等验证完再定。
# 重编的话有两条路:seq_len 提到 2048(丢弃率降到个位数),或允许在助手回复内部
# 截断(会让模型学到「说到一半就停」,不推荐)。
MIDTRAIN_WEIGHTS = {
    "SmolTalk": 0.19,          # en 对话  ≈ 6.5M,与中文侧对齐
    "COIG_CQIA": 0.19,         # zh 对话  ≈ 6.5M,全量
    "MMLU_AuxTrain": 0.58,     # 多选题型 ≈ 20M
    "GSM8K_Train": 0.04,       # 数学/分步推理 ≈ 1.3M,全量
}

# SFT 只用对话,不掺题型 —— 题型该在 midtrain 学完。中英仍对半。
SFT_WEIGHTS = {
    "SmolTalk": 0.5,
    "COIG_CQIA": 0.5,
}

MIXES = {"midtrain": MIDTRAIN_WEIGHTS, "sft": SFT_WEIGHTS}


# ------------------------------------------------------------------ 读取
def _files(name: str) -> list[str]:
    src = _source.get(name)
    pats = src.allow_patterns or [src.part_glob]
    out: list[str] = []
    for p in pats:
        out.extend(sorted(glob.glob(str(src.dir() / p))))
    if not out:
        raise FileNotFoundError(
            f"{name} 没有本地文件({src.dir()} / {pats})。"
            f"先 `make -C data probe {name}` 或 `python data/download.py {name}`。")
    return out


def iter_turns(name: str):
    """→ [(role, content)]。**结构化的轮次**,不是拼好的纯文本 ——
    掩码要知道边界在哪,所以不能复用 encode_corpus 里那套渲染成文本的读法。"""
    src = _source.get(name)
    fmt, field = src.fmt, src.text_field
    files = _files(name)

    if fmt == "jsonl_instruct":                       # COIG-CQIA
        for path in files:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:                          # noqa: BLE001
                        continue
                    q = (r.get("instruction") or "").strip()
                    extra = (r.get("input") or "").strip()
                    a = (r.get("output") or "").strip()
                    if not q or not a:
                        continue
                    yield [("user", f"{q}\n{extra}" if extra else q),
                           ("assistant", a)]
        return

    import pyarrow.parquet as pq

    for path in files:
        pf = pq.ParquetFile(path)
        cols = [field] if fmt != "parquet_qa" else ["question", "answer"]
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            if fmt == "parquet_qa":                   # GSM8K
                for q, a in zip(batch.column(0).to_pylist(),
                                batch.column(1).to_pylist()):
                    if q and a:
                        yield [("user", q), ("assistant", a)]
                continue
            for row in batch.column(0).to_pylist():
                if row is None:
                    continue
                if fmt == "parquet_chat":             # SmolTalk
                    t = [(m.get("role", ""), m.get("content", "")) for m in row]
                elif fmt == "parquet_chat_sharegpt":
                    t = [({"human": "user", "gpt": "assistant"}.get(
                              m.get("from", ""), m.get("from", "")),
                          m.get("value", "")) for m in row]
                elif fmt == "parquet_mc":             # MMLU auxiliary_train
                    ch = row.get("choices") or []
                    if not ch or row.get("answer") is None:
                        continue
                    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ch))
                    t = [("user", f"{row.get('question', '')}\n{opts}"),
                         ("assistant", chr(65 + int(row["answer"])))]
                else:
                    raise ValueError(f"{name}: 不认识的 fmt={fmt}")
                t = [(r, c) for r, c in t if r in ("system", "user", "assistant") and c]
                if len(t) >= 2:
                    yield t


# ------------------------------------------------------------------ 编码
def encode_turns(tok, turns, seq_len):
    """→ [(ids, mask), ...]。**一条对话可能切成多段**,不是丢掉。

    mask=1 的位置要算 loss(只在助手回复及其 eos 上)。

    与 tokenizer.apply_chat_template 产出的序列**必须逐 token 相同** ——
    推理时走的是那个函数,训练和推理的格式对不上,学到的东西就用不上。
    这里自己拼是因为要同时产出掩码,拼法逐行对齐那边的实现。

    ## 为什么切段而不是丢弃

    seq_len 1024 下 SmolTalk 有 **40.9%** 的对话装不下(COIG-CQIA 22.5%) ——
    它是多轮长回复。整条丢掉等于扔掉四成数据,而且扔掉的恰好是信息最多的
    那些长对话,剩下的全是短问答,模型会学得又短又浅。

    切段的粒度是**轮次边界**:攒满就切,下一段重新以 <bos> 开头。绝不截在一轮
    中间 —— 截在助手回复中途的话,模型会学到「说到一半就停」。
    单轮本身就超长的(极少),那一轮整个丢掉。
    """
    segs = []
    ids: list[int] = []
    mask: list[int] = []

    def start_seg(system_content=None):
        nonlocal ids, mask
        ids, mask = [tok.bos_token_id], [0]
        if system_content:
            s = tok.encode(system_content, add_special_tokens=False)
            ids.extend(s); mask.extend([0] * len(s))
            ids.append(tok.system_token_id); mask.append(0)

    sys_content = turns[0][1] if turns and turns[0][0] == "system" else None
    body = turns[1:] if sys_content else turns
    start_seg(sys_content)

    # 一轮 = 用户 + 助手。按轮成对处理,这样切点一定落在轮次边界上。
    i = 0
    while i < len(body):
        role, content = body[i]
        turn_ids: list[int] = []
        turn_mask: list[int] = []
        if role == "user":
            turn_ids.append(tok.user_token_id); turn_mask.append(0)
            b = tok.encode(content, add_special_tokens=False)
            turn_ids.extend(b); turn_mask.extend([0] * len(b))
            # 紧跟的助手回复一起算进这一轮
            if i + 1 < len(body) and body[i + 1][0] == "assistant":
                b2 = tok.encode(body[i + 1][1], add_special_tokens=False)
                turn_ids.append(tok.assistant_token_id); turn_mask.append(0)
                turn_ids.extend(b2); turn_mask.extend([1] * len(b2))
                turn_ids.append(tok.eos_token_id); turn_mask.append(1)
                i += 2
            else:
                i += 1
        else:
            i += 1
            continue

        if len(turn_ids) + 1 > seq_len:
            continue                      # 单轮就超长,这一轮丢掉
        if len(ids) + len(turn_ids) > seq_len:
            if any(mask):                 # 上一段有监督内容才留
                segs.append((ids, mask))
            start_seg(sys_content)
        ids.extend(turn_ids); mask.extend(turn_mask)

    if any(mask):
        segs.append((ids, mask))
    return segs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mix", choices=list(MIXES), required=True)
    p.add_argument("--output", required=True, help="不带扩展名,会写 .pt 和 .mask.pt")
    p.add_argument("--total_tokens", type=int, default=400_000_000)
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer_model", default=None)
    p.add_argument("--cn_dict", default=None)
    a = p.parse_args()

    # 词表在 PieceTokenizer 仓库的 save/ 下,本仓库不留副本 ——
    # resolve_assets() 反查(见 CLAUDE.md)。Wrapper 要的是**目录**。
    piece = a.tokenizer_model or resolve_assets()[0]
    tok = PieceTokenizerWrapper(os.path.dirname(os.path.abspath(str(piece))))

    weights = MIXES[a.mix]
    rng = random.Random(a.seed)
    seq_len = a.seq_length
    budget = {k: int(a.total_tokens * w) for k, w in weights.items()}
    print(f"[chat] mix={a.mix}  目标 {a.total_tokens:,} token  seq_len={seq_len}")
    for k, v in budget.items():
        print(f"  {k:16s} {v:,}")

    # 每个源单独打包 —— 混在一起打包会让某个源的对话被别的源截断,
    # 而且 shuffle 之后配比对不上账。
    all_ids: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    actual = {}
    for name, want in budget.items():
        buf_i: list[int] = []
        buf_m: list[int] = []
        got = dropped = split = 0
        for turns in iter_turns(name):
            segs = encode_turns(tok, turns, seq_len)
            if not segs:
                dropped += 1
                continue
            if len(segs) > 1:
                split += 1
            for ids, mask in segs:
                if a.mix == "midtrain":
                    mask = [1] * len(ids)  # midtrain 整段算 loss
                buf_i.extend(ids); buf_m.extend(mask)
                got += len(ids)
            while len(buf_i) >= seq_len:
                all_ids.append(np.asarray(buf_i[:seq_len], dtype=np.int32))
                all_mask.append(np.asarray(buf_m[:seq_len], dtype=np.uint8))
                del buf_i[:seq_len], buf_m[:seq_len]
            if got >= want:
                break
        actual[name] = got
        print(f"  {name:16s} 实得 {got:,} token"
              f"(切成多段 {split:,} 条,整条丢弃 {dropped:,} 条)")

    if not all_ids:
        print("!! 一条都没编出来"); return 1

    idx = list(range(len(all_ids)))
    rng.shuffle(idx)
    arr = np.stack([all_ids[i] for i in idx])
    msk = np.stack([all_mask[i] for i in idx])

    import torch
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    torch.save(torch.from_numpy(arr), a.output + ".pt")
    torch.save(torch.from_numpy(msk), a.output + ".mask.pt")
    ratio = msk.sum() / msk.size
    meta = {"mix": a.mix, "seq_length": seq_len, "chunks": int(arr.shape[0]),
            "total_tokens": int(arr.size), "supervised_ratio": float(ratio),
            "planned_weights": weights, "actual_tokens": actual}
    with open(a.output + ".mix.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n写出 {arr.shape[0]:,} 条 × {seq_len} = {arr.size:,} token")
    print(f"  被监督的比例 {ratio * 100:.1f}%")
    print(f"  {a.output}.pt / .mask.pt / .mix.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
