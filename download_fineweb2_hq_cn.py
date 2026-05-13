"""Download FineWeb2-HQ Chinese (cmn_Hani) subset.

Full Chinese: 975 files × ~1.1GB = 783 GB / ~130B tokens.
We only need ~1B tokens for v12 alternative CN slot, so grab 15/975 files
(~17 GB, ~2.5B tokens) as headroom.

Different from Chinese-FineWeb-Edu V2.2 (educational filter, OpenCSG):
FineWeb2-HQ uses EPFL XLM-RoBERTa classifier on general FineWeb2 web data.
More general-purpose, larger volume per file.
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
    repo_id="epfml/FineWeb2-HQ",
    repo_type="dataset",
    local_dir=f"{ROOT}/FineWeb2-HQ-CN",
    allow_patterns=[f"cmn_Hani/000_{i:05d}.parquet" for i in range(15)],
    max_workers=4,
)
print("FineWeb2-HQ Chinese subset (15/975 files, ~17GB, ~2.5B tokens) done")
