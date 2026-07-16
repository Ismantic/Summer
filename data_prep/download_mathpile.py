"""Download MathPile (GAIR/MathPile) — now that access is granted.

22 train files (~30GB estimate): arXiv, commoncrawl, stackexchange,
textbooks, wikipedia math domains. Saves to /mnt/data/Summer-data/MathPile.
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
    repo_id="GAIR/MathPile",
    repo_type="dataset",
    local_dir=f"{ROOT}/MathPile",
    allow_patterns=["train/*/*.jsonl.gz"],
    max_workers=4,
)
print("MathPile download done")
