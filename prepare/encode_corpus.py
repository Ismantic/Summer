"""v18 pretokenizer — v15(=v8)P1 mix(去 CnnDailyMail/PeopleDaily,a6000 路径)
                   + v16(=v12 anneal)P2 mix。

用 NEW piece.model (81903) 编码,seqlen 1024。基于 pretokenize_v12.py 框架
(spawn pool + numpy 预分配 + iter_batches 流式)。

  --mix main    → ReTok P1  8 源,EN 0.663 / CN 0.336,1B token
  --mix anneal  → ReTok P2  6 源,EN 0.60  / CN 0.40, 200M token
  --mix scratch → Summer-0.5B 从零预训练 10 源,EN 0.50 / CN 0.50,12B token
                  (配合 --shard_output —— 12B 拼成一份要 48GB 内存)
  --mix anneal_mt → Summer-0.5B 退火段:最高质量单语 0.70 + 中英平行语料 0.30,
                  1.2B token(= WSD 退火窗口 4514 步 × 262,144)
"""
import argparse
import glob
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch


# 源的落地路径、格式、字段全部来自 data/source.py —— 这个文件里不出现任何
# 本机绝对路径。改数据源去改注册表,不要改这里。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import source as _source                             # noqa: E402


# P1: v8 mix 去 CnnDailyMail(0.02)+PeopleDaily(0.03)归一化,
# 再把 Wikipedia_CN 紧池子分一半给 CCI3-HQ。
MAIN_WEIGHTS = {
    "FineWebEdu":   0.316,
    "Wikipedia_EN": 0.189,
    "Gutenberg":    0.105,
    "CC_EN":        0.053,   # v18 代码里叫 C4_EN —— 那是 v17 的遗留名,数据不是 C4
    # EN total = 0.663
    "SkyPile":      0.126,
    "Wikipedia_CN": 0.063,   # 池仅 150M tok,降权
    "CCI3-HQ":      0.126,   # 接 Wiki_CN 让出的权重
    "CC_CN":        0.021,   # 同上,v18 里叫 C4_CN
    # CN total = 0.336
}


# Summer-0.5B 从零预训练的配比。**与 ReTok 的两个 mix 目标不同,不要混用。**
#
# ReTok 是把一个已经会中英文的模型换词表后修回来,配比继承自 Qwen 的语料分布
# (EN 0.663 / CN 0.336)。从零训练没有可继承的东西,配比要直接服务于目标 ——
# 这个模型只对中英翻译负责,所以:
#
#   1. **中英对半。** 66:34 会让中文侧欠训,而翻译两个方向都要考。
#   2. **质量优先于多样性。** 12B token 对 0.52B 参数只有 23 token/参数,
#      每个 token 都得算数;原始网页(CC_EN/CC_CN/SkyPile)保留但降权,
#      教育向和合成教科书(FineWebEdu/Cosmopedia/CN_FineWeb_Edu)顶上去。
#   3. **Gutenberg 压到 0.03。** 公版书是 19 世纪英语,对现代新闻翻译帮助小。
#
# **Gutenberg 实际只喂得出 196M,不是配额的 360M。** 卡它的不是池子大小,是
# `--max_line_chars 100000` —— 每篇截到 10 万字符,而一本书平均 50 万字符,
# 9930 本截完就只剩这么多。按「池子字节 × token/字节」估会得到 1.0B,差 5 倍,
# 因为那个估法看不见截断。少的 164M 占总量 1.4%,中英比例因此是 49.3 / 50.7
# 而不是 50 / 50 —— 偏 0.7 个点,在容差内,所以 2026-07-28 那批数据照用。
#
# 每个源吃掉多少 vs 池子实测有多大(2026-07-28 量的,见 data/source.py):
#
#   FineWebEdu     0.180 → 2.16B / 13.4B      CCI3-HQ         0.140 → 1.68B / 5.2B
#   Cosmopedia     0.120 → 1.44B / 10.1B      SkyPile         0.130 → 1.56B / 12.5B
#   Wikipedia_EN   0.100 → 1.20B / 17.3B      CN_FineWeb_Edu  0.120 → 1.44B / 3.6B
#   CC_EN          0.070 → 0.84B / 10.2B      CC_CN           0.070 → 0.84B / 11.1B
#   Gutenberg      0.030 → 0.36B /  1.0B      Wikipedia_CN    0.040 → 0.48B / 3.5B
#
# CCI3-HQ 和 CN_FineWeb_Edu 的 n_parts 是为这份配比扩过的 —— 原来的池子比
# 消耗量还小,**少喂了不报错**,只会让中英比例悄悄偏掉。所以下面 main() 里
# 有一道实际产出对账。
SCRATCH_WEIGHTS = {
    # ---------------- EN 0.50 ----------------
    "FineWebEdu":     0.180,
    "Cosmopedia":     0.120,
    "Wikipedia_EN":   0.100,
    "CC_EN":          0.070,
    "Gutenberg":      0.030,
    # ---------------- CN 0.50 ----------------
    "CCI3-HQ":        0.140,
    "SkyPile":        0.130,
    "CN_FineWeb_Edu": 0.120,
    "CC_CN":          0.070,
    "Wikipedia_CN":   0.040,
}


# Summer-0.5B 的退火配方:最高质量的单语 + 30% 中英平行语料。
#
# 起因是 2026-07-30 在 S0 的 step 8000(2.5B token)看实际输出:中文已经流利,
# 但 5-shot 的例子对模型没有任何约束力 —— 让它把中文译成英文,它输出一段无关
# 的中文。**这是质的缺口,不是量的缺口**,剩下 9B 单语 token 不会改变它。
#
# 平行语料占 30%(TowerBase 那类配方的量级)。单语那 70% 只留质量最高的六个源
# —— 退火段的作用是把能力「定型」,不是继续铺广度。
#
# 中 / 英 / 双语 = 0.35 / 0.35 / 0.30。平行数据本身两个方向各半,所以实际的
# 语言曝光大致还是对半。
ANNEAL_MT_WEIGHTS = {
    # ---------------- EN 0.35 ----------------
    "FineWebEdu":     0.14,
    "Cosmopedia":     0.14,
    "Wikipedia_EN":   0.07,
    # ---------------- CN 0.35 ----------------
    "CN_FineWeb_Edu": 0.15,
    "CCI3-HQ":        0.12,
    "Wikipedia_CN":   0.08,
    # ---------------- 平行 0.30 ----------------
    "WMT19_ZHEN":     0.26,
    "OPUS100_ZHEN":   0.04,   # 池子只有约 100 万句对,再高就喂不满
}


# P2 anneal: v12 anneal mix 原样 (60/40 HQ-concentrated)
ANNEAL_WEIGHTS = {
    "FineWebEdu":     0.20,
    "Cosmopedia":     0.25,
    "Wikipedia_EN":   0.15,
    # EN total = 0.60
    "CN_FineWeb_Edu": 0.25,
    "Wikipedia_CN":   0.10,
    "CCI3-HQ":        0.05,
    # CN total = 0.40
}


# ---------------------------------------------------------------- 平行语料
#
# ## 为什么不用评测那个模板
#
# `prepare/translate.py` 的 few-shot 提示是:
#
#     Translate Chinese to English.\n\n
#     Chinese: {源}\nEnglish: {译}\n\n   ×5
#     Chinese: {待译}\nEnglish:
#
# 照这个字面写法喂进预训练,5-shot BLEU 就**不再是泛化的度量,而是
# in-distribution 测试** —— 数字会好看,但意义变了。这跟「为了让数字好看去改
# 评测代码」是同一类问题,只是绕了一圈。
#
# 所以下面这些模板一个都不等于它:标签用中文/小写/Source-Target,表头换措辞
# 或者干脆没有。模型要学到的是「连续的双语对照」这件事,不是那一句英文指令。
#
# ## 一篇文档装多对
#
# few-shot 能力来自「看见前面几对之后接着照做」。一篇只放一对的话,模型学到
# 的是句子级的翻译映射,学不到「照着上文的格式继续」。所以每篇塞十几对、
# 同一个模板贯穿,让 1024 的窗口里能看到完整的模式重复。

PAIR_HEADERS = {
    "zh-en": ["以下是中英对照。\n\n", "把下面的中文译成英文。\n\n",
              "Chinese–English bitext:\n\n", ""],
    "en-zh": ["以下是英中对照。\n\n", "把下面的英文译成中文。\n\n",
              "English–Chinese bitext:\n\n", ""],
}
PAIR_LABELS = {
    "zh-en": [("中文", "英文"), ("原文", "译文"), ("zh", "en"),
              ("Source (Chinese)", "Target (English)")],
    "en-zh": [("英文", "中文"), ("原文", "译文"), ("en", "zh"),
              ("Source (English)", "Target (Chinese)")],
}
PAIR_SEPS = [": ", "：", "\t"]


def _format_pair_block(pairs, rng):
    """把若干 (zh, en) 句对格式化成一篇双语文档。"""
    direction = rng.choice(["zh-en", "en-zh"])
    header = rng.choice(PAIR_HEADERS[direction])
    s_lab, t_lab = rng.choice(PAIR_LABELS[direction])
    sep = rng.choice(PAIR_SEPS)
    out = [header]
    for zh, en in pairs:
        s, t = (zh, en) if direction == "zh-en" else (en, zh)
        if sep == "\t":                       # 经典 bitext:一行一对,制表符分隔
            out.append(f"{s}\t{t}\n")
        else:
            out.append(f"{s_lab}{sep}{s}\n{t_lab}{sep}{t}\n\n")
    return "".join(out)


def iter_pairs(files, seed=0, pairs_per_doc=14):
    """读 `translation: struct<en, zh>` 的 parquet,吐格式化好的双语文档。"""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    buf = []
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4000, columns=["translation"]):
            for d in batch.column(0).to_pylist():
                if not d:
                    continue
                zh = (d.get("zh") or "").strip()
                en = (d.get("en") or "").strip()
                # 空的、以及长得不像一句话的都丢掉 —— 平行语料里的超长条目
                # 多半是对齐错位,一条脏数据会带着十几对一起进同一篇文档。
                if not zh or not en or len(zh) > 1000 or len(en) > 1000:
                    continue
                buf.append((zh, en))
                if len(buf) >= pairs_per_doc:
                    yield _format_pair_block(buf, rng)
                    buf = []
    if buf:
        yield _format_pair_block(buf, rng)


def iter_text(fmt, files, field, seed=0):
    if fmt == "parquet_pair":
        yield from iter_pairs(files, seed=seed)
        return
    yield from _iter_text_plain(fmt, files, field)


def _iter_text_plain(fmt, files, field):
    for path in files:
        if fmt == "parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4000, columns=[field]):
                for txt in batch.column(0).to_pylist():
                    if txt:
                        yield txt
            continue
        opener = gzip.open if fmt == "jsonl_gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = obj.get(field)
                if txt:
                    yield txt


def worker_process_shard(args):
    (src_name, fmt, field, files, target_tokens, seq_len, max_line_chars,
     tok_model, cn_dict, out_path, shard_idx) = args
    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    # 位置参数,不用关键字 —— 上游 commit 20d55e0 把这个参数从 `cn_dict` 改名成
    # `dict`,写死关键字名会在升级 PieceTokenizer 之后突然 TypeError。
    # 这条路径就这么坏过一次:v18 之后没再跑过预编码,直到 2026-07-27 才发现。
    tok.load(tok_model, cn_dict)
    eos = tok.piece_to_id("</s>")

    target_chunks = max(1, target_tokens // seq_len)
    out = np.empty((target_chunks, seq_len), dtype=np.int32)
    n_filled = 0
    buf = []
    n_docs, last_log = 0, 0
    t0 = time.time()
    label = f"{src_name}#{shard_idx}"
    # 每个 shard 用不同的种子,模板抽样才不会 52 份全一样;换机器重跑仍相同。
    for text in iter_text(fmt, files, field, seed=1000 + shard_idx):
        if len(text) > max_line_chars:
            text = text[:max_line_chars]
        buf.extend(tok.encode_as_ids(text))
        buf.append(eos)
        n_docs += 1
        while len(buf) >= seq_len and n_filled < target_chunks:
            out[n_filled] = buf[:seq_len]
            del buf[:seq_len]
            n_filled += 1
        if n_filled >= target_chunks:
            break
        if n_docs - last_log >= 50000:
            elapsed = time.time() - t0
            print(f"  [{label}] {n_filled:,}/{target_chunks:,} chunks "
                  f"| {n_docs:,} docs | {elapsed:.0f}s", flush=True)
            last_log = n_docs

    arr = out[:n_filled]
    np.save(out_path, arr)
    elapsed = time.time() - t0
    print(f"  [{label}] DONE {n_filled:,}/{target_chunks:,} chunks "
          f"({n_filled*seq_len:,} tok) | {n_docs:,} docs | {elapsed:.0f}s", flush=True)
    return src_name, shard_idx, out_path, n_filled


def build_shards(weights, total_tokens, n_workers, tmpdir, seq_len, max_line_chars,
                 tok_model, cn_dict, max_tokens_per_shard=0):
    # 每个 worker 预分配 `target_chunks × seq_len` 的 int32 数组,并发跑
    # n_workers 个 —— 所以单份的上限直接决定峰值内存。1B token 的活儿按
    # 12 个 worker 平分是 4GB/份 × 12 = 48GB,本机 61GB 会被压死;
    # 12B 的活儿不封顶就是 4GB/份,同样爆。max_tokens_per_shard 封住这一头。
    max_per_shard = max(1, total_tokens // n_workers)
    if max_tokens_per_shard:
        max_per_shard = min(max_per_shard, max_tokens_per_shard)
    tasks = []
    print("\n=== Shard plan ===")
    for src_name, w in weights.items():
        defn = _source.get(src_name)
        files = [str(p) for p in defn.parts()]
        if not files:
            raise RuntimeError(
                f"{src_name}: {defn.dir()}/{defn.part_glob} 下没有文件。\n"
                f"  先跑 `python data/download.py {src_name}` "
                f"(或 `make -C data download`)。")
        src_target = int(total_tokens * w)
        ideal_shards = max(1, math.ceil(src_target / max_per_shard))
        n_shards = min(ideal_shards, len(files))
        per_shard_target = src_target // n_shards
        shards = [[] for _ in range(n_shards)]
        for i, f in enumerate(files):
            shards[i % n_shards].append(f)
        for idx, fset in enumerate(shards):
            out_path = os.path.join(tmpdir, f"{src_name}_s{idx}.npy")
            tasks.append((src_name, defn.fmt, defn.text_field, fset,
                          per_shard_target, seq_len, max_line_chars,
                          tok_model, cn_dict, out_path, idx))
        print(f"  {src_name:18s} w={w:.3f} files={len(files):4d} "
              f"shards={n_shards} target_per_shard={per_shard_target/1e6:.0f}M tokens")
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix", choices=["main", "anneal", "scratch", "anneal_mt"], required=True)
    p.add_argument("--tmpdir", default="",
                   help="worker 中间 .npy 的落地目录。默认与 --output 同目录 ——"
                        "中间文件总量等于产物大小,放 /tmp 会撑爆 tmpfs。")
    p.add_argument("--shard_output", action="store_true",
                   help="分多个 .pt 写出而不是拼成一份。12B token 拼起来要 48GB 内存。")
    p.add_argument("--max_tokens_per_shard", type=int, default=250_000_000,
                   help="单个 worker 一次预分配多少 token(封住峰值内存)。"
                        "0 = 不封顶(旧行为)。")
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, required=True)
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    # 默认跟着 CPU 核数走,不写死。
    #
    # **原来写死 28,那是 2× A6000 那台机器的核数。** 本机 16 核,28 个 spawn
    # 进程各自加载分词器 + 预分配 numpy 数组,1B token 预算下能把 61GB 内存
    # 压满 —— 2026-07-27 实测把整台机器卡死过一次。
    #
    # 留一核给系统,再压 12 个上限:再多也只是抢内存,I/O 早就饱和了。
    p.add_argument("--num_workers", type=int,
                   default=min(12, max(1, (os.cpu_count() or 4) - 1)))
    args = p.parse_args()

    weights = {"main": MAIN_WEIGHTS, "anneal": ANNEAL_WEIGHTS,
               "scratch": SCRATCH_WEIGHTS, "anneal_mt": ANNEAL_MT_WEIGHTS}[args.mix]
    print(f"Mix: {args.mix} | sources: {len(weights)} | total weight {sum(weights.values()):.3f}")
    print(f"Budget: {args.total_tokens:,} tokens "
          f"({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")
    print(f"Workers: {args.num_workers}")

    # **中间文件和产物一样大,必须落在同一个文件系统上。**
    #
    # 原来直接 `tempfile.mkdtemp()`,也就是 /tmp。ReTok 的 1B token 是 4GB,
    # 塞得进去;从零预训练的 12B 是 48GB,而本机 /tmp 是 31GB 的 tmpfs ——
    # **写满了整台机器的内存都跟着遭殃**,而且是跑到一半才炸(2026-07-28 实测,
    # 第 10 个 shard 左右)。
    #
    # 默认落在 --output 旁边:每个 worker 的 .npy 之和恰好等于最终产物的大小,
    # 产物放得下,中间文件就一定放得下。这个不变式比「记得设 TMPDIR」可靠。
    tmp_root = args.tmpdir or (os.path.dirname(os.path.abspath(args.output)))
    os.makedirs(tmp_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"pretok_{args.mix}_", dir=tmp_root)
    print(f"Temp dir: {tmpdir}")

    tasks = build_shards(weights, args.total_tokens, args.num_workers, tmpdir,
                         args.seq_length, args.max_line_chars,
                         args.tokenizer_model, args.cn_dict,
                         args.max_tokens_per_shard)
    print(f"Total shards: {len(tasks)}")
    tasks.sort(key=lambda t: -t[4])

    t_start = time.time()
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        results = pool.map(worker_process_shard, tasks)
    print(f"\nAll {len(tasks)} shards done in {time.time()-t_start:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.shard_output:
        # **不 concat、不 shuffle。**
        #
        # concat 12B token 要在内存里拼出一个 48GB 的数组,而它下一秒就会被
        # 训练那边 mmap 读回去 —— 纯粹是为了拼而拼。
        #
        # shuffle 也是多余的:`src/train.py` 的 epoch_order 每个 epoch 对全体
        # chunk 做一次完整随机排列,盘上按源聚在一起完全不影响取样。
        # (ReTok 那两个 mix 保留原行为,不动已经跑通的路径。)
        stem = args.output[:-3] if args.output.endswith(".pt") else args.output
        total_chunks = 0
        for i, (name, idx, path, n) in enumerate(results):
            a = np.load(path)
            out = f"{stem}.s{i:03d}.pt"
            torch.save(torch.from_numpy(a), out)
            total_chunks += a.shape[0]
        print(f"\n写出 {len(results)} 个 shard:{stem}.s000.pt … "
              f"s{len(results)-1:03d}.pt")
    else:
        arrays = []
        for name, idx, path, n in results:
            a = np.load(path)
            assert a.shape[0] == n
            arrays.append(a)
        arr = np.concatenate(arrays, axis=0)
        del arrays

        rng = np.random.default_rng(args.seed)
        rng.shuffle(arr, axis=0)
        data = torch.from_numpy(arr)
        total_chunks = arr.shape[0]
        torch.save(data, args.output)
        print(f"Saved to {args.output}")

    print(f"Total: {total_chunks:,} chunks x {args.seq_length} = "
          f"{total_chunks * args.seq_length:,} tokens | {time.time()-t_start:.0f}s")

    # ---------------------------------------------------------------- 对账
    #
    # **池子喂不满不会报错。** worker 读完手上的文件就停,产出比目标少,
    # 最后拼出来的东西照样是一份合法的训练数据 —— 只是配比不是你写的那个。
    # 12B 的跑动里这意味着中英比例悄悄偏掉几个点,而训练日志上什么都看不出来。
    got = {}
    for name, _, _, n in results:
        got[name] = got.get(name, 0) + n * args.seq_length
    actual_total = sum(got.values())
    print(f"\n{'源':<16}{'目标':>10}{'实得':>10}{'完成度':>8}")
    short = []
    for src_name, w in weights.items():
        want = int(args.total_tokens * w)
        have = got.get(src_name, 0)
        pct = have / want if want else 1.0
        flag = "" if pct >= 0.99 else ("  ← 没喂满" if pct < 0.90 else "  ←")
        print(f"{src_name:<16}{want/1e6:>9.0f}M{have/1e6:>9.0f}M{pct*100:>7.1f}%{flag}")
        if pct < 0.90:
            short.append((src_name, pct, (want - have) / args.total_tokens))

    # 判据落在语言比例上,不是逐源完成度。
    #
    # **少喂一个源和配比错掉不是一回事。** Gutenberg 少 164M(占总量 1.4%)
    # 不影响任何结论;而 CN_FineWeb_Edu 的池子比配额小,会把中英比例整体拉偏
    # ——那才是会悄悄改变模型的东西。所以逐源只报告(上表),报错看比例。
    lang_dev = {}
    # 语言集合从注册表来,不写死 —— 退火配方里有 lang="bi" 的平行语料,
    # 硬编码 ("en","zh") 的话它整个不进对账,占了 30% 也看不见。
    for lang in sorted({_source.get(n).lang for n in weights}):
        names = [n for n in weights if _source.get(n).lang == lang]
        plan = sum(weights[n] for n in names)
        real = sum(got.get(n, 0) for n in names) / max(1, actual_total)
        lang_dev[lang] = abs(real - plan)
        print(f"{lang} 计划占比 {plan*100:.1f}% → 实际 {real*100:.1f}% "
              f"(偏 {(real-plan)*100:+.1f} 个点)")
    print(f"总量 计划 {args.total_tokens/1e9:.2f}B → 实得 {actual_total/1e9:.2f}B")

    # 让数据自我描述:配方写在代码里会漂,而 checkpoint 上看不出它吃的是哪一版。
    stem = args.output[:-3] if args.output.endswith(".pt") else args.output
    with open(f"{stem}.mix.json", "w") as f:
        json.dump({"mix": args.mix, "seq_length": args.seq_length,
                   "planned_total": args.total_tokens,
                   "actual_total": actual_total,
                   "planned_weights": weights,
                   "actual_tokens": got,
                   "actual_weights": {k: v / actual_total for k, v in got.items()}},
                  f, indent=2, ensure_ascii=False)
    print(f"实际配比写入 {stem}.mix.json")

    for _, _, path, _ in results:
        try: os.remove(path)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass

    if max(lang_dev.values()) > 0.01:
        raise SystemExit(
            f"\n**中英比例偏了 {max(lang_dev.values())*100:.1f} 个点(见上表)。**\n"
            "  数据已经写出来了,没白跑 —— 但它的配比不是设计的那个,而且\n"
            "  事后从 checkpoint 上看不出来。翻译两个方向都要考,比例是会\n"
            "  直接改变结论的东西。\n"
            "  通常是某个源的池子比配额小:调大它的 n_parts 再下"
            "(data/source.py),或者改权重。\n"
            "  没喂满的源:" + ", ".join(
                f"{n} 只有 {p*100:.0f}%(缺总量的 {g*100:.1f}%)"
                for n, p, g in short))


if __name__ == "__main__":
    main()
