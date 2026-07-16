"""v9 weighted multi-source pretokenizer.

Same machinery as pretokenize_v7.py but with v9's source mix:
- Adds OpenWebMath (20% — STEM content for MMLU/ARC-C lift)
- Shifts EN/CN to 60/40 (vs v7's 65/35) — CN side gets a little more
- Drops CnnDailyMail (news redundant with C4-CN and SkyPile)
- Targets a 4B-token pool by default (vs v8's 2B), 1.5B training tokens
  (vs v8's 1B) to push translation closer to base.
"""
import argparse
import glob
import gzip
import json
import os
import random
import time

import numpy as np
import torch

import piece_tokenizer as pt


SOURCES = [
    # ---- EN 60% ----
    {
        "name": "FineWebEdu",
        "weight": 0.20,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/FinewebEdu_json_files/sample-10bt-*.json",
    },
    {
        "name": "Wikipedia_EN",
        "weight": 0.10,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/Wikipedia_en_json_files/wiki-20231101-en-*.json",
    },
    {
        "name": "Gutenberg",
        "weight": 0.05,
        "format": "parquet",
        "field": "text",
        "glob": "/mnt/data/Summer-data/gutenberg/data/train-*.parquet",
    },
    {
        "name": "C4_EN",
        "weight": 0.02,
        "format": "jsonl_gz",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/c2-en/*.json.gz",
    },
    {
        "name": "OpenWebMath",
        "weight": 0.10,
        "format": "parquet",
        "field": "text",
        "glob": "/mnt/data/Summer-data/OpenWebMath/data/train-*.parquet",
    },
    {
        "name": "MathPile",
        "weight": 0.13,
        "format": "jsonl_gz",
        "field": "text",
        "glob": "/mnt/data/Summer-data/MathPile/train/*/*.jsonl.gz",
    },
    # ---- CN 40% ----
    {
        "name": "SkyPile",
        "weight": 0.14,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/SkyPile_json_files/skypile-*.json",
    },
    {
        "name": "Wikipedia_CN",
        "weight": 0.12,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/Wikipedia_cn_s_json_files/wiki-20231101-zh-*.json",
    },
    {
        "name": "CCI3-HQ",
        "weight": 0.08,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/Summer-data/CCI3-HQ/data/part_*.jsonl",
    },
    {
        "name": "PeopleDaily",
        "weight": 0.04,
        "format": "txt",
        "glob": "/home/tfbao/Data/data/PeopleDaily.documents.txt",
    },
    {
        "name": "C4_CN",
        "weight": 0.02,
        "format": "jsonl_gz",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/c2-cn/*.json.gz",
    },
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
            tbl = pq.read_table(path, columns=[field])
            for txt in tbl.column(field).to_pylist():
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


def collect_chunks(source, tok, eos, target_tokens, seq_len, max_line_chars):
    """Pre-allocate one int32 array of shape [target_chunks, seq_len] and fill in.

    Pre-allocation avoids the Python-list-of-list-of-int memory blow-up
    (~28 B/int → ~100 GB for 4B tokens). int32 ndarray is 4 B/int → ~16 GB.
    """
    target_chunks = max(1, target_tokens // seq_len)
    out = np.empty((target_chunks, seq_len), dtype=np.int32)
    n_filled = 0
    buf = []
    n_docs, last_log = 0, 0
    t0 = time.time()
    for text in iter_text(source):
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
        if n_docs - last_log >= 20000:
            elapsed = time.time() - t0
            print(f"  [{source['name']}] {n_filled:,}/{target_chunks:,} chunks "
                  f"| {n_docs:,} docs | {elapsed:.0f}s", flush=True)
            last_log = n_docs
    elapsed = time.time() - t0
    print(f"  [{source['name']}] DONE {n_filled:,}/{target_chunks:,} chunks "
          f"({n_filled*seq_len:,} tok) | {n_docs:,} docs | {elapsed:.0f}s", flush=True)
    return out[:n_filled]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, default=4_000_000_000)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    total_w = sum(s["weight"] for s in SOURCES)
    print(f"Source count: {len(SOURCES)}, total weight {total_w:.3f}")
    print(f"Budget: {args.total_tokens:,} tokens "
          f"({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")

    tok = pt.Tokenizer()
    tok.load(args.tokenizer_model, cn_dict=args.cn_dict)
    eos = tok.piece_to_id("</s>")
    print(f"Vocab: {tok.vocab_size()}, eos={eos}")

    per_source_arrays = []
    t_start = time.time()
    for src in SOURCES:
        target = int(args.total_tokens * src["weight"])
        print(f"\n=== {src['name']} weight={src['weight']:.3f} target={target:,} tokens ===", flush=True)
        arr_src = collect_chunks(src, tok, eos, target, args.seq_length, args.max_line_chars)
        per_source_arrays.append(arr_src)

    arr = np.concatenate(per_source_arrays, axis=0)
    del per_source_arrays
    rng = np.random.default_rng(args.seed)
    rng.shuffle(arr, axis=0)

    data = torch.from_numpy(arr)
    elapsed = time.time() - t_start
    print(f"\nTotal: {arr.shape[0]:,} chunks x {args.seq_length} = "
          f"{data.numel():,} tokens | {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
