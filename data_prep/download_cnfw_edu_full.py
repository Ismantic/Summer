"""Download full Chinese-FineWeb-Edu V2.2 score 4-5 subset (9745 files / 69GB / ~12B tokens).

snapshot_download is idempotent — already-downloaded 800 files (file 000000-000799) will be
skipped, just fetches the remaining ~8945.
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
    allow_patterns=["4_5/*.parquet"],
    max_workers=8,
)
print("Chinese-FineWeb-Edu V2.2 full 4_5 subset (9745 files / ~69GB) done")
