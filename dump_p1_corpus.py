"""Dump v17 Phase 1 corpus 源数据 → en.txt / cn.txt（一行一个 document）。
用于重新训练 piece tokenizer。流式读取，复用 pretokenize_v17.SOURCES + iter_text。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretokenize_v17 import SOURCES, iter_text

# v17 SOURCES 的 9 个源拆 EN / CN
EN = {"FineWebEdu", "Wikipedia_EN", "Gutenberg", "C4_EN", "CnnDailyMail"}
CN = {"SkyPile", "Wikipedia_CN", "CCI3-HQ", "C4_CN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_per_source", type=int, default=0,
                    help="每个源最多 docs（0 = 全部）")
    ap.add_argument("--max_chars", type=int, default=200_000,
                    help="单 doc 截断字符数（避免极长行）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    en_path = os.path.join(args.out_dir, "en.txt")
    cn_path = os.path.join(args.out_dir, "cn.txt")

    t0 = time.time()
    counts = {}
    chars_total = {"en": 0, "cn": 0}
    with open(en_path, "w", encoding="utf-8") as fen, \
         open(cn_path, "w", encoding="utf-8") as fcn:
        for src in SOURCES:
            name = src["name"]
            if name in EN:
                fout, side = fen, "en"
            elif name in CN:
                fout, side = fcn, "cn"
            else:
                print(f"skip {name} (not classified)")
                continue
            print(f"\n=== {name} ({side}) ===", flush=True)
            n = 0
            ts = time.time()
            for text in iter_text(src):
                # 一行一 doc：折叠所有空白（含 \n \r \t）为单空格
                t = " ".join(text.split())
                if not t:
                    continue
                if len(t) > args.max_chars:
                    t = t[:args.max_chars]
                fout.write(t + "\n")
                n += 1
                chars_total[side] += len(t)
                if args.max_per_source > 0 and n >= args.max_per_source:
                    break
                if n % 200000 == 0:
                    elapsed = time.time() - ts
                    print(f"  {n:,} docs | {elapsed:.0f}s", flush=True)
            counts[name] = n
            print(f"  {name} DONE: {n:,} docs | {time.time()-ts:.0f}s", flush=True)

    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===")
    for k, v in counts.items():
        side = "en" if k in EN else "cn"
        print(f"  {k:18s} ({side}): {v:>12,} docs")
    print(f"\n  total EN chars: {chars_total['en']:>14,}")
    print(f"  total CN chars: {chars_total['cn']:>14,}")
    print(f"  -> {en_path}")
    print(f"  -> {cn_path}")


if __name__ == "__main__":
    main()
