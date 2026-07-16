"""v9 multi-process pretokenizer with file-level sharding.

Strategy: split each source's file list into shards proportional to source weight,
so heavy sources get more workers. Target: 4B tokens in ≤20 min on 32-core box.

Per shard: 1 worker, ~125M tokens (~12 min single-thread @ 180k tok/s).
All shards run in parallel via Pool, max wall = longest single shard.

Each worker:
  - loads piece_tokenizer (~3s)
  - pre-allocates int32 ndarray (shard_target × 512 = ~500MB for 125M tokens)
  - tokenizes its assigned files only
  - writes .npy temp file (avoids pickling 500MB across pipe)
"""
import argparse
import glob
import gzip
import json
import math
import multiprocessing as mp
import os
import tempfile
import time

import numpy as np
import torch


SOURCES = [
    {"name": "FineWebEdu", "weight": 0.20, "format": "jsonl", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/FinewebEdu_json_files/sample-10bt-*.json"},
    {"name": "Wikipedia_EN", "weight": 0.10, "format": "jsonl", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/Wikipedia_en_json_files/wiki-20231101-en-*.json"},
    {"name": "Gutenberg", "weight": 0.05, "format": "parquet", "field": "text",
     "glob": "/mnt/data/Summer-data/gutenberg/data/train-*.parquet"},
    {"name": "C4_EN", "weight": 0.02, "format": "jsonl_gz", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/c2-en/*.json.gz"},
    {"name": "OpenWebMath", "weight": 0.10, "format": "parquet", "field": "text",
     "glob": "/mnt/data/Summer-data/OpenWebMath/data/train-*.parquet"},
    {"name": "MathPile", "weight": 0.13, "format": "jsonl_gz", "field": "text",
     "glob": "/mnt/data/Summer-data/MathPile/train/*/*.jsonl.gz"},
    {"name": "SkyPile", "weight": 0.14, "format": "jsonl", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/SkyPile_json_files/skypile-*.json"},
    {"name": "Wikipedia_CN", "weight": 0.12, "format": "jsonl", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/Wikipedia_cn_s_json_files/wiki-20231101-zh-*.json"},
    {"name": "CCI3-HQ", "weight": 0.08, "format": "jsonl", "field": "text",
     "glob": "/mnt/data/Summer-data/CCI3-HQ/data/part_*.jsonl"},
    {"name": "PeopleDaily", "weight": 0.04, "format": "txt",
     "glob": "/home/tfbao/Data/data/PeopleDaily.documents.txt"},
    {"name": "C4_CN", "weight": 0.02, "format": "jsonl_gz", "field": "text",
     "glob": "/mnt/data/now/Nano-1.5/nano/c2-cn/*.json.gz"},
]


def iter_text(fmt, files, field):
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


def worker_process_shard(args):
    """One shard = subset of files from a source, fixed token budget."""
    (src_name, fmt, field, files, target_tokens, seq_len, max_line_chars,
     tok_model, cn_dict, out_path, shard_idx) = args

    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(tok_model, cn_dict=cn_dict)
    eos = tok.piece_to_id("</s>")

    target_chunks = max(1, target_tokens // seq_len)
    out = np.empty((target_chunks, seq_len), dtype=np.int32)
    n_filled = 0
    buf = []
    n_docs, last_log = 0, 0
    t0 = time.time()
    label = f"{src_name}#{shard_idx}"
    for text in iter_text(fmt, files, field):
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


def build_shards(sources, total_tokens, n_workers, tmpdir, seq_len, max_line_chars,
                 tok_model, cn_dict):
    """Build the task list, sharded so that no shard exceeds ~total/n_workers tokens
    (subject to per-source file count cap)."""
    total_w = sum(s["weight"] for s in sources)
    # Cap per-shard work so the longest shard ~= total_tokens / n_workers,
    # avoiding the case where a heavy single-shard source becomes the wall-time bottleneck.
    max_per_shard = max(1, total_tokens // n_workers)
    tasks = []
    for src in sources:
        files = sorted(glob.glob(src["glob"]))
        if not files:
            raise RuntimeError(f"No files match {src['glob']}")
        src_target = int(total_tokens * src["weight"])
        ideal_shards = max(1, math.ceil(src_target / max_per_shard))
        n_shards = min(ideal_shards, len(files))
        per_shard_target = src_target // n_shards
        shards = [[] for _ in range(n_shards)]
        for i, f in enumerate(files):
            shards[i % n_shards].append(f)
        for idx, fset in enumerate(shards):
            out_path = os.path.join(tmpdir, f"{src['name']}_s{idx}.npy")
            tasks.append((src["name"], src["format"], src.get("field", "text"),
                          fset, per_shard_target, seq_len, max_line_chars,
                          tok_model, cn_dict, out_path, idx))
        print(f"  {src['name']:14s} w={src['weight']:.2f} "
              f"files={len(files):3d} shards={n_shards} "
              f"target_per_shard={per_shard_target/1e6:.0f}M tokens")
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, default=4_000_000_000)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=28)
    p.add_argument("--sources", default="", help="Comma-separated names (default all)")
    args = p.parse_args()

    selected = SOURCES
    if args.sources:
        names = set(args.sources.split(","))
        selected = [s for s in SOURCES if s["name"] in names]
        if not selected:
            raise SystemExit(f"No sources match {names}")

    total_w = sum(s["weight"] for s in selected)
    print(f"Source count: {len(selected)}, total weight {total_w:.3f}")
    print(f"Budget: {args.total_tokens:,} tokens "
          f"({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")
    print(f"Workers: {args.num_workers}")

    tmpdir = tempfile.mkdtemp(prefix="pretok_v9_mp_")
    print(f"Temp dir: {tmpdir}")

    print("\n=== Shard plan ===")
    tasks = build_shards(selected, args.total_tokens, args.num_workers, tmpdir,
                         args.seq_length, args.max_line_chars,
                         args.tokenizer_model, args.cn_dict)
    print(f"Total shards: {len(tasks)} (queue depth = {max(0, len(tasks) - args.num_workers)})")

    # Sort heaviest first so longest jobs start immediately.
    tasks.sort(key=lambda t: -t[4])  # t[4] = target_tokens

    t_start = time.time()
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        results = pool.map(worker_process_shard, tasks)
    elapsed_workers = time.time() - t_start
    print(f"\nAll {len(tasks)} shards done in {elapsed_workers:.0f}s")

    arrays = []
    for name, idx, path, n in results:
        a = np.load(path)
        assert a.shape[0] == n, f"{name}#{idx}: stored {a.shape[0]} != reported {n}"
        arrays.append(a)
    arr = np.concatenate(arrays, axis=0)
    del arrays

    rng = np.random.default_rng(args.seed)
    rng.shuffle(arr, axis=0)

    data = torch.from_numpy(arr)
    elapsed = time.time() - t_start
    print(f"\nTotal: {arr.shape[0]:,} chunks x {args.seq_length} = "
          f"{data.numel():,} tokens | {elapsed:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")

    for _, _, path, _ in results:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
