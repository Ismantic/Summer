"""本地文件的定位 —— 所有 Task 都从这里拿路径,不各自 glob。

`data/source.py` 是数据源的唯一真相来源;Task 只问它「这个源的文件在哪」。
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def files(name: str) -> list[str]:
    """→ 该数据源的本地文件列表(排序过,保证顺序确定)。"""
    from data import source as _source
    src = _source.get(name)
    pats = src.allow_patterns or [src.part_glob]
    out: list[str] = []
    for p in pats:
        out.extend(sorted(glob.glob(str(src.dir() / p))))
    if not out:
        raise FileNotFoundError(
            f"{name} 没有本地文件({src.dir()} / {pats})。"
            f"先 `make -C data probe {name}` 或 `python data/download.py {name}`。")
    return out


def parquet_batches(name: str, columns: list[str], batch_size: int = 2000):
    """按批读 parquet 的某几列。→ 逐批 yield pyarrow 的 RecordBatch。"""
    import pyarrow.parquet as pq
    for path in files(name):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield batch
