"""把 en.txt 补充到 ~60GB（从 a6000 上的 FineWebEdu 加料）。
Phase 1 EN 占比最大就是 FineWebEdu（v7 mix 0.30），且 a6000 有 47GB 富余。
"""
import argparse
import glob
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/tfbao/Shiyu/Summer/tokenizer_corpus/en.txt")
    ap.add_argument("--target_gb", type=float, default=60.0)
    ap.add_argument("--source_glob",
                    default="/home/tfbao/a6000/FinewebEdu_json_files/sample-10bt-*.json")
    ap.add_argument("--max_chars", type=int, default=200_000)
    args = ap.parse_args()

    target = int(args.target_gb * 1024**3)
    start_size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    print(f"en.txt 当前 {start_size/1e9:.2f}GB，目标 {args.target_gb}GB（差 {(target-start_size)/1e9:.2f}GB）")

    if start_size >= target:
        print("已达标，不动")
        return

    files = sorted(glob.glob(args.source_glob))
    print(f"source: {len(files)} files matched {args.source_glob}")

    written_bytes = 0
    written_docs = 0
    t0 = time.time()
    done = False
    with open(args.out, "a", encoding="utf-8") as f:
        for i, path in enumerate(files):
            if done: break
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("text")
                    if not t: continue
                    t = " ".join(t.split())
                    if len(t) > args.max_chars:
                        t = t[:args.max_chars]
                    if not t: continue
                    line_out = t + "\n"
                    f.write(line_out)
                    written_bytes += len(line_out.encode("utf-8"))
                    written_docs += 1
                    if start_size + written_bytes >= target:
                        done = True
                        break
            cur = start_size + written_bytes
            print(f"  file {i+1}/{len(files)} | written {written_docs:,} docs + "
                  f"{written_bytes/1e9:.2f}GB | total {cur/1e9:.2f}GB | {time.time()-t0:.0f}s",
                  flush=True)

    f_size = os.path.getsize(args.out)
    print(f"\nDONE: en.txt 最终 {f_size/1e9:.2f}GB（+{(f_size-start_size)/1e9:.2f}GB, "
          f"+{written_docs:,} docs, {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
