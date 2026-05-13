"""Download Cosmopedia v2 subset for v12 mix.

Cosmopedia v2 = HuggingFaceTB's synthetic textbooks/stories/articles (28B tokens, 104 parquets, 114GB).
For our 0.6B Qwen3 ReTok experiment, we only need ~600M tokens worth.
Grab 8 parquets (~9GB, ~2.1B tokens) as headroom for 10-15% weight in v12 mix.
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
    repo_id="HuggingFaceTB/cosmopedia-v2",
    repo_type="dataset",
    local_dir=f"{ROOT}/cosmopedia-v2",
    allow_patterns=[f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(8)],
    max_workers=4,
)
print("Cosmopedia v2 subset (8/104 parquets, ~9GB / ~2.1B tokens) download done")
