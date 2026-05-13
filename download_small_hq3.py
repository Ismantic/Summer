"""Small HQ batch 3 - more <1GB curated, balance EN/CN.

EN:
  pszemraj/simple_wikipedia_LM (279MB) — Simple Wikipedia, middle-school level
  alexfabbri/multi_news (722MB)        — high-quality EN news summaries

CN:
  suolyer/baike (63MB)                  — small Chinese encyclopedia
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
    ("pszemraj/simple_wikipedia_LM", "simple_wikipedia_LM"),
    ("alexfabbri/multi_news", "multi_news"),
    ("suolyer/baike", "baike-cn-small"),
]:
    print(f"--- {repo_id} ---")
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=f"{ROOT}/{local_subdir}", max_workers=4)
        print(f"  done -> {ROOT}/{local_subdir}")
    except Exception as e:
        print(f"  FAIL: {e}")
print("Small HQ batch 3 downloads complete")
