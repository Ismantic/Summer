"""build_cn_corpus_30g.py — 拼接 5 个中文源到 cn_v2.txt(目标 30GB)。

源:
  Wikipedia_CN (全):     /home/tfbao/a6000/Wikipedia_cn_json_files/wiki-cn-*.json
  PeopleDaily (全):      /home/tfbao/Shiyu/Data/data/PeopleDaily.sentences.txt
  CCI3-HQ (全):          /home/tfbao/a6000/Summer-data/CCI3-HQ/data/part_*.jsonl
  SkyPile (取~9G):       /home/tfbao/a6000/SkyPile/*.parquet
  Chinese-FineWeb-Edu (取~10.6G): /home/tfbao/a6000/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet

格式:每行一个 document(空白折叠为单空格,过长截 200K 字)。
"""
import glob, gzip, json, os, sys, time


OUT = "/home/tfbao/Shiyu/Summer/tokenizer_corpus/cn_v2.txt"
MAX_CHARS = 200_000


def write_doc(fout, text, target_bytes, current_bytes):
    """text 折叠+截断后写一行;返回新 current_bytes。如果超 target 返回 -1 表示停。"""
    if not text:
        return current_bytes
    t = " ".join(text.split())
    if not t:
        return current_bytes
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS]
    line = t + "\n"
    fout.write(line)
    new = current_bytes + len(line.encode("utf-8"))
    return new


def iter_jsonl(path, field="text"):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            t = obj.get(field)
            if t: yield t


def iter_parquet(path, field="text"):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4000, columns=[field]):
        for t in batch.column(0).to_pylist():
            if t: yield t


def iter_plain_text(path):
    """PD.sentences.txt: 每行一句,合并成 doc(粗暴 1 行 = 1 doc 也可)。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line: yield line


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    t0 = time.time()
    fout = open(OUT, "w", encoding="utf-8")
    total = 0
    n_docs = 0

    plan = [
        # (name, iter_fn args, target_GB)
        # 全量加(目标 = inf 实际由源决定)
        ("Wikipedia_CN", "jsonl", "/home/tfbao/a6000/Wikipedia_cn_json_files/wiki-cn-*.json", None),
        ("PeopleDaily",  "plain", "/home/tfbao/Shiyu/Data/data/PeopleDaily.sentences.txt", None),
        ("CCI3-HQ",      "jsonl", "/home/tfbao/a6000/Summer-data/CCI3-HQ/data/part_*.jsonl", None),
        # 取限量
        ("SkyPile",      "parquet", "/home/tfbao/a6000/SkyPile/*.parquet", 9.0),
        ("Chinese-FineWeb-Edu", "parquet",
         "/home/tfbao/a6000/Summer-data/Chinese-FineWeb-Edu-V2.2/4_5/*.parquet", 11.0),
    ]

    for name, fmt, pattern, cap_gb in plan:
        files = sorted(glob.glob(pattern))
        if not files and fmt != "plain":
            print(f"  [SKIP] {name}: no files match {pattern}")
            continue
        if fmt == "plain":
            files = [pattern]
        before = total
        cap_bytes = int(cap_gb * 1024**3) if cap_gb else None
        print(f"\n=== {name} ({fmt}, {len(files)} files, "
              f"cap={cap_gb}GB if set) ===", flush=True)
        done = False
        for fp in files:
            if done: break
            if fmt == "jsonl":
                gen = iter_jsonl(fp)
            elif fmt == "parquet":
                gen = iter_parquet(fp)
            else:
                gen = iter_plain_text(fp)
            for txt in gen:
                total = write_doc(fout, txt, cap_bytes, total)
                n_docs += 1
                if cap_bytes and (total - before) >= cap_bytes:
                    done = True
                    break
                if n_docs % 200000 == 0:
                    print(f"  ... {n_docs:,} docs, {total/1e9:.2f}GB, "
                          f"{time.time()-t0:.0f}s", flush=True)
        added = (total - before) / 1e9
        print(f"  {name} done: +{added:.2f}GB (total {total/1e9:.2f}GB)")

    fout.close()
    print(f"\n=== Saved {OUT} ===")
    print(f"  Final size: {os.path.getsize(OUT)/1e9:.2f}GB")
    print(f"  Total docs: {n_docs:,}")
    print(f"  Elapsed:    {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
