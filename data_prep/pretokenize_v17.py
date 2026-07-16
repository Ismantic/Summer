"""v17 weighted multi-source pretokenizer (多进程 + 低内存版).

v7 mix 的严格复现版，但用新 81903 piece 词表，数据源改为重新下载/本地语料：
  - Wikipedia EN/CN 用本地 enwiki.jsonl / cnwiki.jsonl
  - 其余源从 ModelScope 重下（FineWebEdu / C4 / SkyPile / CCI3-HQ），Gutenberg 从 HF 下
  - PeopleDaily（v7 占 0.03）ModelScope 无纯文本语料，份额并入 CCI3-HQ（0.06→0.09）

权重总和 = 1.0，EN 65% / CN 35%，与 v7 一致。

多进程：每个源只有 1-2 个文件，无法按文件分片，故改为**文档级并行** ——
主进程流式读文档，imap 分发给 N 个 worker 并行 encode（保序），主进程按序 packing。

低内存：chunks 直接写入预分配的 numpy int32 数组（不是 Python list of int，
后者 2B token 会占 50GB+ 触发 OOM）；parquet 用 iter_batches 流式读（不 to_pylist
整个文件）。峰值内存 ~16GB。

Run:
    python data_prep/pretokenize_v17.py \
        --tokenizer_model ./piece_v2.model \
        --cn_dict /home/tfbao/Shiyu/PieceTokenizer/dict.txt \
        --output ./output/phase1_train_512_v17.pt \
        --total_tokens 2000000000 [--num_workers 14]

注意: --cn_dict 必须是【词表训练时用的那个】dict（PieceTokenizer/dict.txt，
359987 行），不是 Summer/dict.txt。两者内容不同，用错会导致分词与词表不一致。
"""
import argparse
import glob
import gzip
import json
import os
import time
import multiprocessing as mp

import numpy as np
import torch


_CORPUS = "/home/tfbao/Shiyu/Summer/corpus_v17"
_LOCAL_WIKI = "/home/tfbao/Shiyu/PieceTokenizer/data/data"

# Total weight = 1.0. EN 65% / CN 35%. No data is repeated; weight is sampling
# fraction at the chunk level.
SOURCES = [
    # ---- EN 65% ----
    {"name": "FineWebEdu", "weight": 0.30, "format": "parquet", "field": "text",
     "glob": f"{_CORPUS}/fineweb_edu/data/CC-MAIN-2013-20/train-00000-of-00014.parquet"},
    {"name": "Wikipedia_EN", "weight": 0.18, "format": "jsonl", "field": "text",
     "glob": f"{_LOCAL_WIKI}/enwiki.jsonl"},
    {"name": "Gutenberg", "weight": 0.10, "format": "parquet", "field": "text",
     "glob": f"{_CORPUS}/gutenberg/data/en-*.parquet"},
    {"name": "C4_EN", "weight": 0.05, "format": "jsonl_gz", "field": "text",
     "glob": f"{_CORPUS}/c4_en/en.noblocklist/c4-train.00001-of-01024.json.gz"},
    {"name": "CnnDailyMail", "weight": 0.02, "format": "jsonl", "field": "text",
     "glob": f"{_CORPUS}/cnn_dailymail/cnn_documents.jsonl"},
    # ---- CN 35% ----
    {"name": "SkyPile", "weight": 0.12, "format": "jsonl", "field": "text",
     "glob": f"{_CORPUS}/skypile/data/2020-40_zh_head_0000.jsonl"},
    {"name": "Wikipedia_CN", "weight": 0.12, "format": "jsonl", "field": "text",
     "glob": f"{_LOCAL_WIKI}/cnwiki.jsonl"},
    # v7 的 CCI3-HQ(0.06) + PeopleDaily(0.03) 合并
    {"name": "CCI3-HQ", "weight": 0.09, "format": "jsonl", "field": "text",
     "glob": f"{_CORPUS}/cci3_hq/data/part_*.jsonl"},
    {"name": "C4_CN", "weight": 0.02, "format": "jsonl", "field": "text",
     "glob": f"{_CORPUS}/c4_cn/data/chinese-c4-0000-of-0096.jsonl"},
]


def iter_text(source):
    """Yield text strings from one source. Skips records with missing/empty text."""
    fmt = source["format"]
    files = sorted(glob.glob(source["glob"]))
    if not files:
        raise RuntimeError(f"No files match {source['glob']}")
    field = source.get("field", "text")

    for path in files:
        if fmt == "parquet":
            # 流式按 batch 读，不把整个 parquet 的 text 列 to_pylist 进内存
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4000, columns=[field]):
                for txt in batch.column(0).to_pylist():
                    if txt:
                        yield txt
            continue
        if fmt == "txt":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
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


# --- 多进程 worker：每个进程各自 load 一份 tokenizer ---
_TOK = None
_EOS = None


def _init_worker(tok_model, cn_dict):
    global _TOK, _EOS
    import piece_tokenizer as pt
    _TOK = pt.Tokenizer()
    _TOK.load(tok_model, cn_dict=cn_dict)
    _EOS = _TOK.piece_to_id("</s>")


def _encode_doc(text):
    """encode 一篇文档，返回 token ids + 末尾 </s>。"""
    ids = _TOK.encode_as_ids(text)
    ids.append(_EOS)
    return ids


def collect_chunks(source, pool, target_tokens, seq_len, max_line_chars):
    """分批读文档 -> pool.map 并行 encode（保序、限流）-> packing 到预分配 int32 数组。

    限流关键：每次只把 BATCH 篇文档交给 pool.map（阻塞到该批 encode 完才读下一批），
    避免 imap 对超大 generator（如 42GB 的 enwiki.jsonl）无限缓冲 —— 那曾让主进程
    吃到 50GB+ 触发 earlyoom。in-flight 内存 ≈ BATCH 篇文档，受控。
    """
    target_chunks = max(1, target_tokens // seq_len)
    out = np.empty((target_chunks, seq_len), dtype=np.int32)   # 预分配，避免 list-of-int
    n_filled = 0
    buf = []
    n_docs = 0
    t0 = time.time()
    last_log = 0
    BATCH = 20000

    def flush(batch):
        nonlocal n_filled, n_docs
        for ids in pool.map(_encode_doc, batch, chunksize=128):
            buf.extend(ids)
            n_docs += 1
            while len(buf) >= seq_len and n_filled < target_chunks:
                out[n_filled] = buf[:seq_len]
                del buf[:seq_len]
                n_filled += 1

    batch = []
    for text in iter_text(source):
        if len(text) > max_line_chars:
            text = text[:max_line_chars]
        batch.append(text)
        if len(batch) < BATCH:
            continue
        flush(batch)
        batch = []
        if n_filled >= target_chunks:
            break
        if n_docs - last_log >= 100000:
            elapsed = time.time() - t0
            print(f"  [{source['name']}] {n_filled:,}/{target_chunks:,} chunks "
                  f"| {n_docs:,} docs | {elapsed:.0f}s", flush=True)
            last_log = n_docs
    if batch and n_filled < target_chunks:
        flush(batch)

    elapsed = time.time() - t0
    if n_filled < target_chunks:
        print(f"  [{source['name']}] WARNING: source exhausted, "
              f"{n_filled:,} < {target_chunks:,} target chunks", flush=True)
    print(f"  [{source['name']}] DONE {n_filled:,}/{target_chunks:,} chunks "
          f"({n_filled*seq_len:,} tok) | {n_docs:,} docs | {elapsed:.0f}s", flush=True)
    return out[:n_filled]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, default=500_000_000)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--num_workers", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    total_w = sum(s["weight"] for s in SOURCES)
    print(f"Source count: {len(SOURCES)}, total weight {total_w:.3f}")
    print(f"Budget: {args.total_tokens:,} tokens "
          f"({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")
    print(f"Workers: {args.num_workers}")

    pool = mp.Pool(args.num_workers, initializer=_init_worker,
                   initargs=(args.tokenizer_model, args.cn_dict))

    arrays = []
    t_start = time.time()
    for src in SOURCES:
        target = int(args.total_tokens * src["weight"])
        print(f"\n=== {src['name']} weight={src['weight']:.3f} target={target:,} tokens ===",
              flush=True)
        arrays.append(collect_chunks(src, pool, target,
                                     args.seq_length, args.max_line_chars))

    pool.terminate()
    pool.join()

    data_np = np.concatenate(arrays, axis=0)
    del arrays
    np.random.seed(args.seed)
    np.random.shuffle(data_np)            # in-place shuffle along axis 0

    data = torch.from_numpy(data_np)
    elapsed = time.time() - t_start
    print(f"\nTotal: {data.shape[0]:,} chunks x {args.seq_length} = "
          f"{data.numel():,} tokens | {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
