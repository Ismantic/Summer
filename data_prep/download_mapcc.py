"""Download MAP-CC (Chinese Tiny LLM corpus) targeted subset.

Fills two gaps in our pipeline:
1. zh_baike (~2.6GB) — Baidu Baike Chinese encyclopedia (distinct from Wikipedia)
2. zh_books.part01 (~37GB) — Chinese books (completely untouched domain in our mix)

Skip zh_cc (~280GB) — already covered by SkyPile/CCI3-HQ/Chinese-FineWeb-Edu V2.2.
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from huggingface_hub import snapshot_download

ROOT = "/mnt/data/Summer-data"

snapshot_download(
    repo_id="m-a-p/MAP-CC",
    repo_type="dataset",
    local_dir=f"{ROOT}/MAP-CC",
    allow_patterns=[
        "zh_baike.jsonl.gz.part*",   # full encyclopedia, 2.6GB
        "zh_books.jsonl.gz.part01",  # 1 part of books, 37GB
    ],
    max_workers=4,
)
print("MAP-CC subset (zh_baike full + zh_books part01) download done")
