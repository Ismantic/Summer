"""v10 pretokenizer — two-mix variant for ReTok-proper.

Mixes (selected via --mix):
  main  : Phase 1 frozen-transformer broad-coverage mix (RegMix-informed: web-heavy)
  anneal: Phase 2 unfreeze annealed mix (high-quality narrow, code+math up-weighted)

Source mix follows published bilingual recipes (Skywork 52:40:8, MAP-Neo decay 17% code)
and RegMix's finding that web > Wikipedia for downstream perf. Six sources max for simplicity.

Reuses the file-sharded multi-process machinery from pretokenize_v9_mp.py.
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


# ---- Source definitions ----

SOURCE_DEFS = {
    "FineWebEdu": dict(format="jsonl", field="text",
        glob="/mnt/data/now/Nano-1.5/nano/FinewebEdu_json_files/sample-10bt-*.json"),
    "Wikipedia_EN": dict(format="jsonl", field="text",
        glob="/mnt/data/now/Nano-1.5/nano/Wikipedia_en_json_files/wiki-20231101-en-*.json"),
    "SkyPile": dict(format="jsonl", field="text",
        glob="/mnt/data/now/Nano-1.5/nano/SkyPile_json_files/skypile-*.json"),
    "Wikipedia_CN": dict(format="jsonl", field="text",
        glob="/mnt/data/now/Nano-1.5/nano/Wikipedia_cn_s_json_files/wiki-20231101-zh-*.json"),
    "Code": dict(format="parquet", field="code",
        glob="/mnt/data/Summer-data/github-code-clean/data/train-*.parquet"),
    "OpenWebMath": dict(format="parquet", field="text",
        glob="/mnt/data/Summer-data/OpenWebMath/data/train-*.parquet"),
    "Gutenberg": dict(format="parquet", field="text",
        glob="/mnt/data/Summer-data/gutenberg/data/train-*.parquet"),
    "CCI3-HQ": dict(format="jsonl", field="text",
        glob="/mnt/data/Summer-data/CCI3-HQ/data/part_*.jsonl"),
}

# Phase 1 broad mix (RegMix-informed: web 73%, code 9%, math 7%, encyclopedic 11%)
MAIN_WEIGHTS = {
    "FineWebEdu": 0.38,
    "SkyPile":    0.35,
    "Code":       0.09,
    "OpenWebMath":0.07,
    "Wikipedia_EN":0.06,
    "Wikipedia_CN":0.05,
}

# Phase 2 anneal mix: up-weight Wikipedia + Code + Math slightly while
# preserving EN/CN language balance (~42:40). Less aggressive than MAP-Neo
# decay (17% code) to avoid hurting CN performance.
ANNEAL_WEIGHTS = {
    "FineWebEdu": 0.30,
    "SkyPile":    0.30,
    "Wikipedia_EN":0.12,
    "Wikipedia_CN":0.10,
    "Code":       0.10,
    "OpenWebMath":0.08,
}


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


def build_shards(weights, total_tokens, n_workers, tmpdir, seq_len, max_line_chars,
                 tok_model, cn_dict):
    max_per_shard = max(1, total_tokens // n_workers)
    tasks = []
    print("\n=== Shard plan ===")
    for src_name, w in weights.items():
        defn = SOURCE_DEFS[src_name]
        files = sorted(glob.glob(defn["glob"]))
        if not files:
            raise RuntimeError(f"No files match {defn['glob']} for {src_name}")
        src_target = int(total_tokens * w)
        ideal_shards = max(1, math.ceil(src_target / max_per_shard))
        n_shards = min(ideal_shards, len(files))
        per_shard_target = src_target // n_shards
        shards = [[] for _ in range(n_shards)]
        for i, f in enumerate(files):
            shards[i % n_shards].append(f)
        for idx, fset in enumerate(shards):
            out_path = os.path.join(tmpdir, f"{src_name}_s{idx}.npy")
            tasks.append((src_name, defn["format"], defn["field"], fset,
                          per_shard_target, seq_len, max_line_chars,
                          tok_model, cn_dict, out_path, idx))
        print(f"  {src_name:14s} w={w:.2f} files={len(files):3d} "
              f"shards={n_shards} target_per_shard={per_shard_target/1e6:.0f}M tokens")
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix", choices=["main", "anneal"], required=True)
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, required=True)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=28)
    args = p.parse_args()

    weights = MAIN_WEIGHTS if args.mix == "main" else ANNEAL_WEIGHTS
    total_w = sum(weights.values())
    print(f"Mix: {args.mix} | sources: {len(weights)} | total weight {total_w:.3f}")
    print(f"Budget: {args.total_tokens:,} tokens "
          f"({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")
    print(f"Workers: {args.num_workers}")

    tmpdir = tempfile.mkdtemp(prefix=f"pretok_v10_{args.mix}_")
    print(f"Temp dir: {tmpdir}")

    tasks = build_shards(weights, args.total_tokens, args.num_workers, tmpdir,
                         args.seq_length, args.max_line_chars,
                         args.tokenizer_model, args.cn_dict)
    print(f"Total shards: {len(tasks)} (queue depth = {max(0, len(tasks) - args.num_workers)})")
    tasks.sort(key=lambda t: -t[4])  # heaviest first

    t_start = time.time()
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        results = pool.map(worker_process_shard, tasks)
    print(f"\nAll {len(tasks)} shards done in {time.time()-t_start:.0f}s")

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
    print(f"\nTotal: {arr.shape[0]:,} chunks x {args.seq_length} = "
          f"{data.numel():,} tokens | {time.time()-t_start:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")

    for _, _, path, _ in results:
        try: os.remove(path)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass


if __name__ == "__main__":
    main()
