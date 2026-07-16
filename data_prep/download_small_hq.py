"""Download small high-quality EN datasets (<3GB total) for v13+ mix candidates.

1. open-phi/textbooks (~0.13GB) — Phi-1 style synthetic textbooks, complement Cosmopedia
2. JeanKaddour/minipile (~3GB) — Pile subset with multi-domain coverage:
   ArXiv + Books + StackExchange + Reddit + etc. Single source fills several gaps.
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from huggingface_hub import snapshot_download

ROOT = "/mnt/data/Summer-data"

for repo_id, local_subdir in [
    ("open-phi/textbooks", "open-phi-textbooks"),
    ("JeanKaddour/minipile", "minipile"),
]:
    print(f"--- {repo_id} ---")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=f"{ROOT}/{local_subdir}",
        max_workers=4,
    )
    print(f"  done -> {ROOT}/{local_subdir}")
print("Small HQ downloads complete")
