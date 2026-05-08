"""
Multi-process pretokenize: split each input file into N byte-aligned ranges,
have N workers tokenize their slice in parallel, then concat + chunk.

Usage matches pretokenize.py except for --workers. Each worker loads its own
piece_tokenizer (with dict) once at startup, so initialization cost is paid
N times — fine for the speedup it buys on 32-core boxes.
"""
import argparse
import os
import time
import multiprocessing as mp
import numpy as np
import torch


# Per-worker globals (init'd once per process)
_TOK = None
_BOS = None
_EOS = None


def _worker_init(tokenizer_model: str, cn_dict: str):
    """Load tokenizer once per worker process."""
    global _TOK, _BOS, _EOS
    import piece_tokenizer as pt
    _TOK = pt.Tokenizer()
    _TOK.load(tokenizer_model, cn_dict=cn_dict or "")
    _BOS = _TOK.piece_to_id("<s>")
    _EOS = _TOK.piece_to_id("</s>")


def _worker_run(args_tuple):
    """Tokenize a [start, end) byte range of a single file, return packed ids
    as int32 numpy array (raw flat stream — main slices into chunks)."""
    path, start, end = args_tuple
    bos, eos = _BOS, _EOS
    # int list buffer; numpy at the end so we don't pay per-append cost.
    ids = []
    n_lines = 0
    with open(path, "rb") as f:
        # If we're not at byte 0, skip to next newline so we don't mid-cut a line.
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        else:
            f.seek(0)
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # GPT-style packing: single </s> separator (matches Qwen3 native
            # where bos_id == eos_id == <|endoftext|>).
            ids.extend(_TOK.encode_as_ids(line))
            ids.append(eos)
            n_lines += 1
    return np.asarray(ids, dtype=np.int32), n_lines


def _split_file(path: str, n_parts: int):
    """Return n_parts (start, end) byte ranges that together cover the file."""
    size = os.path.getsize(path)
    step = size // n_parts
    ranges = []
    for i in range(n_parts):
        s = i * step
        e = (i + 1) * step if i < n_parts - 1 else size
        ranges.append((path, s, e))
    return ranges


def main(args):
    files = [f.strip() for f in args.input.split(",")]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    # Pre-build the task list: each file split into args.workers byte ranges,
    # interleaved across files so that an early stop (max_chunks) gives a
    # roughly balanced sample from all sources.
    per_file_ranges = [_split_file(f, args.workers) for f in files]
    tasks = []
    for i in range(args.workers):
        for ranges in per_file_ranges:
            tasks.append(ranges[i])
    print(f"[mp] files={len(files)} workers={args.workers} tasks={len(tasks)}")

    t0 = time.time()
    # Concatenate with imap_unordered → faster, but we lose order. We don't
    # care: shuffle happens at training time anyway.
    flat_buffers = []
    total_lines = 0
    target_tokens = args.max_chunks * args.seq_length if args.max_chunks else None

    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(args.tokenizer_model, args.cn_dict or "")) as pool:
        running_tokens = 0
        for ids_arr, n_lines in pool.imap_unordered(_worker_run, tasks):
            flat_buffers.append(ids_arr)
            running_tokens += ids_arr.size
            total_lines += n_lines
            elapsed = time.time() - t0
            print(f"  +{ids_arr.size:>10,} tokens (+{n_lines:>7,} lines) "
                  f"| total {running_tokens:>12,} tokens / {total_lines:>10,} lines "
                  f"| {elapsed:5.0f}s")
            if target_tokens and running_tokens >= target_tokens:
                # tell remaining workers we don't need their results — but
                # imap_unordered will still drain. With Pool.terminate we
                # stop early; small bias but saves time.
                pool.terminate()
                break

    print(f"[mp] tokenize done in {time.time()-t0:.0f}s, concatenating buffers...")
    flat = np.concatenate(flat_buffers) if flat_buffers else np.zeros(0, dtype=np.int32)
    n_full_chunks = flat.size // args.seq_length
    if args.max_chunks:
        n_full_chunks = min(n_full_chunks, args.max_chunks)
    flat = flat[: n_full_chunks * args.seq_length]
    chunks = flat.reshape(n_full_chunks, args.seq_length)

    elapsed = time.time() - t0
    print(f"[mp] {n_full_chunks:,} chunks x {args.seq_length} = {chunks.size:,} tokens "
          f"| {elapsed:.0f}s total | {n_full_chunks/elapsed:.0f} chunks/s")

    torch.save(torch.from_numpy(chunks), args.output)
    print(f"[mp] Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Comma-separated file list")
    p.add_argument("--tokenizer_model", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--cn_dict", type=str, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    main(args)
