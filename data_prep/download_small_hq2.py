"""Download small-but-distinctive datasets, strict <1GB rule, balance EN/CN.

EN:
  sentence-transformers/eli5 (108MB)   — EN Q&A explanations (Reddit ELI5)
  storytracer/US-PD-Books (83MB)        — US public-domain books, literary EN
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
    ("sentence-transformers/eli5", "eli5"),
    ("storytracer/US-PD-Books", "US-PD-Books"),
]:
    print(f"--- {repo_id} ---")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=f"{ROOT}/{local_subdir}",
        max_workers=4,
    )
    print(f"  done -> {ROOT}/{local_subdir}")
print("Small HQ batch 2 (EN) downloads complete")
