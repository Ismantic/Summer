"""Download Chinese-FineWeb-Edu V2.2 score-4-5 (highest quality) subset.

Full 4_5/ subset: 9745 parquet files, 69 GB, ~12B tokens.
We only need ~1B tokens for v12 CN slot, so grab 800/9745 files (~5.5GB, ~1B tokens).

Score ≥4 mirrors SmolLM's Python-Edu approach (score≥4 → 3x convergence speedup).
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
    repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
    repo_type="dataset",
    local_dir=f"{ROOT}/Chinese-FineWeb-Edu-V2.2",
    allow_patterns=[f"4_5/{i:06d}.parquet" for i in range(800)],
    max_workers=4,
)
print("Chinese-FineWeb-Edu V2.2 score-4-5 subset (800/9745 files, ~5.5GB, ~1B tokens) done")
