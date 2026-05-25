"""扫描中文语料,构建字级 vocab。
输出 vocab.txt(每行一字,id = 行号),5 个 specials + 频次 >= min_freq 的字。

用法:
  python build_char_vocab.py --corpus cn.txt --output vocab.txt --min_freq 10
"""
import argparse, collections, time

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="一行一个 document 的中文文本")
    ap.add_argument("--output", default="vocab.txt")
    ap.add_argument("--min_freq", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    cnt = collections.Counter()
    total = 0
    with open(args.corpus, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            for c in line.strip():
                cnt[c] += 1
                total += 1
            if (i + 1) % 200000 == 0:
                print(f"  scanned {i+1} lines, {total/1e9:.2f}B chars, "
                      f"{len(cnt)} unique | {time.time()-t0:.0f}s", flush=True)

    vocab = list(SPECIALS) + [c for c, n in cnt.most_common() if n >= args.min_freq]
    kept_chars = sum(n for c, n in cnt.most_common() if n >= args.min_freq)
    coverage = kept_chars / max(1, total)

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(c + "\n" for c in vocab)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  total chars scanned: {total:,}")
    print(f"  unique chars: {len(cnt):,}")
    print(f"  kept (freq >= {args.min_freq}): {len(vocab)-len(SPECIALS):,} + {len(SPECIALS)} specials = {len(vocab):,}")
    print(f"  coverage: {coverage:.4%}")
    print(f"  vocab → {args.output}")


if __name__ == "__main__":
    main()
