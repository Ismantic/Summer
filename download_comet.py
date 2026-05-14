"""Download Unbabel/wmt22-comet-da model (~400MB) for translation eval.

COMET = neural-based translation metric, WMT22+ standard. Correlates much better
with human judgments than BLEU. Uses XLM-RoBERTa base.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Unbabel/wmt22-comet-da",
    local_dir="/mnt/data/Summer-data/comet-wmt22-da",
    max_workers=4,
)
print("COMET wmt22-da model done")
