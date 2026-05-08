"""
Pre-tokenize text files into packed token chunks saved as a .pt file.
Single-threaded, simple and fast.
"""
import argparse
import time
import numpy as np
import torch
import piece_tokenizer as pt


def main(args):
    tok = pt.Tokenizer()
    tok.load(args.tokenizer_model, cn_dict=args.cn_dict if args.cn_dict else "")
    bos = tok.piece_to_id("<s>")
    eos = tok.piece_to_id("</s>")
    print(f"Vocab: {tok.vocab_size()}, seq_len={args.seq_length}")

    files = [f.strip() for f in args.input.split(",")]
    handles = [open(f, 'r', encoding='utf8', errors='replace') for f in files]
    if args.skip_lines > 0:
        for fh in handles:
            for _ in range(args.skip_lines):
                if not fh.readline():
                    break
        print(f"Skipped first {args.skip_lines} lines of each file")
    exhausted = [False] * len(handles)

    buf = []
    chunks = []
    lines_read = 0
    t0 = time.time()

    while not all(exhausted):
        for i, fh in enumerate(handles):
            if exhausted[i]:
                continue
            line = fh.readline()
            if not line:
                exhausted[i] = True
                continue
            line = line.strip()
            if not line:
                continue

            # GPT-style packing: single </s> separator between docs (matching
            # Qwen3 base format where bos_id == eos_id == <|endoftext|>).
            # No leading <s> — adjacent docs become "...text1</s>text2</s>..."
            buf.extend(tok.encode_as_ids(line))
            buf.append(eos)
            lines_read += 1

            while len(buf) >= args.seq_length:
                chunks.append(buf[:args.seq_length])
                buf = buf[args.seq_length:]

                if args.max_chunks and len(chunks) >= args.max_chunks:
                    break

            if lines_read % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  {len(chunks):,} chunks | {lines_read:,} lines | {elapsed:.0f}s | {lines_read/elapsed:.0f} lines/s")

            if args.max_chunks and len(chunks) >= args.max_chunks:
                break

        if args.max_chunks and len(chunks) >= args.max_chunks:
            break

    for fh in handles:
        fh.close()

    n = len(chunks)
    arr = np.array(chunks, dtype=np.int32)
    data = torch.from_numpy(arr)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s | {n:,} chunks x {args.seq_length} = {data.numel():,} tokens")
    torch.save(data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer_model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=384)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--cn_dict", type=str, default=None)
    parser.add_argument("--skip_lines", type=int, default=0,
                        help="Skip first N lines of each input file (for held-out valid set)")
    args = parser.parse_args()
    main(args)
