"""Process MNBVC_Subtitle.jsonl into bilingual parallel docs.

For each {title=zh, content=en} pair:
1. OpenCC T2S to normalize traditional → simplified Chinese
2. Output 2 lines (both directions) in template format matching our WMT eval:
   "English: <en> Chinese: <zh>"
   "Chinese: <zh> English: <en>"
"""
import argparse
import json
import time

from opencc import OpenCC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/home/tfbao/Data/data/MNBVC_Subtitle.jsonl")
    p.add_argument("--output", default="/home/tfbao/Data/data/Subtitle.parallel.txt")
    p.add_argument("--max_lines", type=int, default=None)
    args = p.parse_args()

    cc = OpenCC("t2s")
    n_pairs = n_skipped = 0
    t0 = time.time()
    with open(args.input, encoding="utf-8", errors="replace") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.max_lines and i >= args.max_lines:
                break
            try:
                d = json.loads(line)
            except Exception:
                n_skipped += 1
                continue
            zh = cc.convert((d.get("title") or "").strip())
            en = (d.get("content") or "").strip()
            if not zh or not en:
                n_skipped += 1
                continue
            # Two directions per pair
            fout.write(f"English: {en} Chinese: {zh}\n")
            fout.write(f"Chinese: {zh} English: {en}\n")
            n_pairs += 1
            if n_pairs % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"  {n_pairs:,} pairs ({2*n_pairs:,} lines) | {elapsed:.0f}s | {n_pairs/elapsed:.0f} pairs/s")

    elapsed = time.time() - t0
    print(f"\nDone. {n_pairs:,} pairs → {2*n_pairs:,} lines | "
          f"{elapsed:.0f}s | skipped {n_skipped} | {args.output}")


if __name__ == "__main__":
    main()
