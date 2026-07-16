"""v19 pretokenizer — from-scratch 10B token mix。
基于 pretokenize_v18.py,加入 Cosmopedia + CN_FineWeb_Edu 大池,确保各源池容量 >= 1.2× 需求。
EN 62% / CN 38%(略偏 EN,from-scratch 通常吃更多 EN)。
"""
import argparse, glob, gzip, json, math, multiprocessing as mp, os, tempfile, time
import numpy as np
import torch

A6000 = "/home/tfbao/a6000"

SOURCE_DEFS = {
    "FineWebEdu":     dict(format="jsonl",     field="text",
                           glob=f"{A6000}/FinewebEdu_json_files/sample-10bt-*.json"),
    "Wikipedia_EN":   dict(format="jsonl",     field="text",
                           glob=f"{A6000}/Wikipedia_en_json_files/wiki-20231101-en-*.json"),
    "Gutenberg":      dict(format="parquet",   field="text",
                           glob=f"{A6000}/Summer-data/gutenberg/data/train-*.parquet"),
    "C4_EN":          dict(format="jsonl_gz",  field="text",
                           glob=f"{A6000}/c2-en/*.json.gz"),
    "Cosmopedia":     dict(format="parquet",   field="text",
                           glob=f"{A6000}/Summer-data/cosmopedia-v2/cosmopedia-v2/train-*.parquet"),
    "SkyPile":        dict(format="parquet",   field="text",
                           glob=f"{A6000}/SkyPile/*.parquet"),
    "Wikipedia_CN":   dict(format="jsonl",     field="text",
                           glob=f"{A6000}/Wikipedia_cn_json_files/wiki-cn-*.json"),
    "CCI3-HQ":        dict(format="jsonl",     field="text",
                           glob=f"{A6000}/Summer-data/CCI3-HQ/data/part_*.jsonl"),
    "C4_CN":          dict(format="jsonl_gz",  field="text",
                           glob=f"{A6000}/c2-cn/*.json.gz"),
    "CN_FineWeb_Edu": dict(format="parquet",   field="text",
                           glob=f"{A6000}/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet"),
}

# v19 mix: 10B token,各源池容量 >= 1.2×
MAIN_WEIGHTS = {
    "FineWebEdu":     0.30,
    "Wikipedia_EN":   0.13,
    "Gutenberg":      0.04,
    "C4_EN":          0.05,
    "Cosmopedia":     0.10,
    # EN total = 0.62
    "SkyPile":        0.13,
    "Wikipedia_CN":   0.01,
    "CCI3-HQ":        0.08,
    "C4_CN":          0.01,
    "CN_FineWeb_Edu": 0.15,
    # CN total = 0.38
}


def iter_text(fmt, files, field):
    for path in files:
        if fmt == "parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4000, columns=[field]):
                for txt in batch.column(0).to_pylist():
                    if txt: yield txt
            continue
        opener = gzip.open if fmt == "jsonl_gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except json.JSONDecodeError: continue
                txt = obj.get(field)
                if txt: yield txt


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
    print(f"  [{label}] DONE {n_filled:,}/{target_chunks:,} chunks "
          f"({n_filled*seq_len:,} tok) | {n_docs:,} docs | {time.time()-t0:.0f}s", flush=True)
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
        print(f"  {src_name:18s} w={w:.3f} files={len(files):4d} "
              f"shards={n_shards} target/shard={per_shard_target/1e6:.0f}M tokens")
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_model", required=True)
    p.add_argument("--cn_dict", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--total_tokens", type=int, required=True)
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--max_line_chars", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=28)
    args = p.parse_args()

    print(f"v19 mix: {len(MAIN_WEIGHTS)} sources, total weight {sum(MAIN_WEIGHTS.values()):.3f}")
    print(f"Budget: {args.total_tokens:,} tokens ({args.total_tokens // args.seq_length:,} chunks of {args.seq_length})")
    tmpdir = tempfile.mkdtemp(prefix="pretok_v19_")
    print(f"Temp dir: {tmpdir}")

    tasks = build_shards(MAIN_WEIGHTS, args.total_tokens, args.num_workers, tmpdir,
                         args.seq_length, args.max_line_chars, args.tokenizer_model, args.cn_dict)
    print(f"Total shards: {len(tasks)}")
    tasks.sort(key=lambda t: -t[4])

    t_start = time.time()
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        results = pool.map(worker_process_shard, tasks)
    print(f"\nAll {len(tasks)} shards done in {time.time()-t_start:.0f}s")

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
