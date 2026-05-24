"""把 cn.txt 补充到 ~20GB（从 a6000 上的 Chinese-FineWeb-Edu-V2.2 加料）。
流式读 parquet → 折叠空白 → append 到 cn.txt,达到 TARGET 字节就停。
"""
import argparse
import glob
import os
import time
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/tfbao/Shiyu/Summer/tokenizer_corpus/cn.txt")
    ap.add_argument("--target_gb", type=float, default=20.0)
    ap.add_argument("--source_glob",
                    default="/home/tfbao/a6000/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet")
    ap.add_argument("--max_chars", type=int, default=200_000)
    args = ap.parse_args()

    target = int(args.target_gb * 1024**3)
    start_size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    print(f"cn.txt 当前 {start_size/1e9:.2f}GB，目标 {args.target_gb}GB（差 {(target-start_size)/1e9:.2f}GB）")

    if start_size >= target:
        print("已达标，不动")
        return

    files = sorted(glob.glob(args.source_glob))
    print(f"source: {len(files)} parquet files matched {args.source_glob}")

    written_bytes = 0
    written_docs = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, path in enumerate(files):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4000, columns=["text"]):
                for txt in batch.column(0).to_pylist():
                    if not txt:
                        continue
                    t = " ".join(txt.split())
                    if len(t) > args.max_chars:
                        t = t[:args.max_chars]
                    if not t:
                        continue
                    line = t + "\n"
                    f.write(line)
                    written_bytes += len(line.encode("utf-8"))
                    written_docs += 1
                cur = start_size + written_bytes
                if cur >= target:
                    break
            cur = start_size + written_bytes
            if i % 50 == 0 or cur >= target:
                print(f"  file {i+1}/{len(files)} | written {written_docs:,} docs "
                      f"+ {written_bytes/1e9:.2f}GB | total {cur/1e9:.2f}GB "
                      f"| {time.time()-t0:.0f}s", flush=True)
            if cur >= target:
                break

    f_size = os.path.getsize(args.out)
    print(f"\nDONE: cn.txt 最终 {f_size/1e9:.2f}GB（+{(f_size-start_size)/1e9:.2f}GB, "
          f"+{written_docs:,} docs, {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
