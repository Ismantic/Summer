"""v17 anneal pretokenize（基于 a6000 真数据 + v12 anneal mix）。

v16 配方的 Phase 2 anneal mix（6 源，EN:CN=60:40），seqlen 1024，~200M token。
数据全部来自 /home/tfbao/a6000（用户从 A6000 机拷过来的原始语料）。

继承 pretokenize_v17.py 的低内存多进程框架（流式 parquet + 分批 pool.map 限流 +
预分配 numpy 数组）。
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


A6000 = "/home/tfbao/a6000"

# v12 anneal weights（精选 HQ tier）
SOURCES = [
    # ---- EN 60% ----
    {"name": "FineWebEdu",     "weight": 0.20, "format": "jsonl",   "field": "text",
     "glob": f"{A6000}/FinewebEdu_json_files/sample-10bt-*.json"},
    {"name": "Cosmopedia",     "weight": 0.25, "format": "parquet", "field": "text",
     "glob": f"{A6000}/Summer-data/cosmopedia-v2/cosmopedia-v2/train-*.parquet"},
    {"name": "Wikipedia_EN",   "weight": 0.15, "format": "jsonl",   "field": "text",
     "glob": f"{A6000}/Wikipedia_en_json_files/wiki-*.json"},
    # ---- CN 40% ----
    {"name": "CN_FineWeb_Edu", "weight": 0.25, "format": "parquet", "field": "text",
     "glob": f"{A6000}/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet"},
    {"name": "Wikipedia_CN",   "weight": 0.10, "format": "jsonl",   "field": "text",
     "glob": f"{A6000}/Wikipedia_cn_json_files/wiki*.json"},
    {"name": "CCI3-HQ",        "weight": 0.05, "format": "jsonl",   "field": "text",
     "glob": f"{A6000}/Summer-data/CCI3-HQ/data/part_*.jsonl"},
]


def iter_text(source):
    fmt = source["format"]
    files = sorted(glob.glob(source["glob"]))
    if not files:
        raise RuntimeError(f"No files match {source['glob']}")
    field = source.get("field", "text")
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


_TOK = None
_EOS = None


def _init_worker(tok_model, cn_dict):
    global _TOK, _EOS
    import piece_tokenizer as pt
    _TOK = pt.Tokenizer()
    _TOK.load(tok_model, cn_dict=cn_dict)
    _EOS = _TOK.piece_to_id("</s>")


def _encode_doc(text):
    ids = _TOK.encode_as_ids(text)
    ids.append(_EOS)
    return ids


def collect_chunks(source, pool, target_tokens, seq_len, max_line_chars):
    target_chunks = max(1, target_tokens // seq_len)
    out = np.empty((target_chunks, seq_len), dtype=np.int32)
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
        print(f"  [{source['name']}] WARNING: exhausted {n_filled:,}/{target_chunks:,}", flush=True)
    print(f"  [{source['name']}] DONE {n_filled:,}/{target_chunks:,} chunks "
          f"({n_filled*seq_len:,} tok) | {n_docs:,} docs | {elapsed:.0f}s", flush=True)
    return out[:n_filled]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, default=200_000_000)
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--num_workers", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    total_w = sum(s["weight"] for s in SOURCES)
    print(f"v17 anneal: {len(SOURCES)} sources, total weight {total_w:.3f}")
    print(f"Budget: {args.total_tokens:,} tokens, seqlen {args.seq_length}, workers {args.num_workers}")

    pool = mp.Pool(args.num_workers, initializer=_init_worker,
                   initargs=(args.tokenizer_model, args.cn_dict))
    arrays = []
    t_start = time.time()
    for src in SOURCES:
        target = int(args.total_tokens * src["weight"])
        print(f"\n=== {src['name']} weight={src['weight']:.3f} target={target:,} ===", flush=True)
        arrays.append(collect_chunks(src, pool, target, args.seq_length, args.max_line_chars))
    pool.terminate(); pool.join()

    data_np = np.concatenate(arrays, axis=0)
    del arrays
    np.random.seed(args.seed)
    np.random.shuffle(data_np)
    data = torch.from_numpy(data_np)
    elapsed = time.time() - t_start
    print(f"\nTotal: {data.shape[0]:,} chunks x {args.seq_length} = {data.numel():,} tokens | {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
