"""v7 weighted multi-source pretokenizer.

Reads from many JSON / parquet / txt sources, each capped at
total_tokens * weight, packs into [N, seq_len] int32 chunks with single </s>
separator (Qwen3-style), shuffles, saves as .pt.

No upsampling — every source has plenty of data, weight is sampling fraction
not epoch count.

Run:
    python data_prep/pretokenize_v7.py \
        --tokenizer_model ./piece_mt.model \
        --cn_dict ./dict.txt \
        --output ./output/phase1_train_512_v7.pt \
        --total_tokens 500000000
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


# Total weight = 1.0. EN 65% / CN 35%. No data is repeated; weight is sampling
# fraction at the chunk level.
SOURCES = [
    # ---- EN 65% ----
    {
        "name": "FineWebEdu",
        "weight": 0.30,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/FinewebEdu_json_files/sample-10bt-*.json",
    },
    {
        "name": "Wikipedia_EN",
        "weight": 0.18,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/Wikipedia_en_json_files/wiki-20231101-en-*.json",
    },
    {
        "name": "Gutenberg",
        "weight": 0.10,
        "format": "parquet",
        "field": "text",
        "glob": "/mnt/data/Summer-data/gutenberg/data/train-*.parquet",
    },
    {
        "name": "C4_EN",
        "weight": 0.05,
        "format": "jsonl_gz",
        "field": "text",
        "glob": "/mnt/data/now/Nano-1.5/nano/c2-en/*.json.gz",
    },
    {
        "name": "CnnDailyMail",
        "weight": 0.02,
        "format": "txt",
        "glob": "/home/tfbao/Data/data/CnnDailyMail.documents.txt",
    },
    # ---- CN 35% ----
    {
        "name": "SkyPile",
        "weight": 0.12,
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
        "weight": 0.06,
        "format": "jsonl",
        "field": "text",
        "glob": "/mnt/data/Summer-data/CCI3-HQ/data/part_*.jsonl",
    },
    {
        "name": "PeopleDaily",
        "weight": 0.03,
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
    """Yield text strings from one source. Skips records with missing/empty text."""
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
    """Pull from source until target_tokens worth of chunks collected.

    GPT-style packing: each doc gets a single </s> appended; the buffer is
    sliced into seq_len chunks (so adjacent docs span chunk boundaries).
    """
    target_chunks = max(1, target_tokens // seq_len)
    buf = []
    chunks = []
    n_docs = 0
    t0 = time.time()
    last_log = 0

    for text in iter_text(source):
        if len(text) > max_line_chars:
            text = text[:max_line_chars]
        buf.extend(tok.encode_as_ids(text))
        buf.append(eos)
        n_docs += 1

        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(chunks) >= target_chunks:
                break

        if len(chunks) >= target_chunks:
            break

        if n_docs - last_log >= 20000:
            elapsed = time.time() - t0
            print(f"  [{source['name']}] {len(chunks):,}/{target_chunks:,} chunks "
                  f"| {n_docs:,} docs | {elapsed:.0f}s")
            last_log = n_docs

    elapsed = time.time() - t0
    print(f"  [{source['name']}] DONE {len(chunks):,}/{target_chunks:,} chunks "
          f"({len(chunks)*seq_len:,} tok) | {n_docs:,} docs | {elapsed:.0f}s")
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, default=500_000_000)
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

    all_chunks = []
    t_start = time.time()
    for src in SOURCES:
        target = int(args.total_tokens * src["weight"])
        print(f"\n=== {src['name']} weight={src['weight']:.3f} target={target:,} tokens ===")
        chunks = collect_chunks(src, tok, eos, target,
                                args.seq_length, args.max_line_chars)
        all_chunks.extend(chunks)

    random.seed(args.seed)
    random.shuffle(all_chunks)

    arr = np.array(all_chunks, dtype=np.int32)
    data = torch.from_numpy(arr)
    elapsed = time.time() - t_start
    print(f"\nTotal: {len(all_chunks):,} chunks x {args.seq_length} = "
          f"{data.numel():,} tokens | {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
