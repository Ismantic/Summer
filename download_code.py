"""Download codeparrot/github-code-clean subset for v10 Phase 1 mix.

Total dataset is ~50GB / 880 parquet files. We only need ~280M tokens (~9% of 3-4B pool).
Grab first 20 parquets (~1.2GB) which gives ~300-400M tokens with headroom.
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
    repo_id="codeparrot/github-code-clean",
    repo_type="dataset",
    local_dir=f"{ROOT}/github-code-clean",
    allow_patterns=[f"data/train-{i:05d}-of-00880.parquet" for i in range(20)],
    max_workers=4,
)
print("github-code-clean subset (20/880 parquets) download done")
