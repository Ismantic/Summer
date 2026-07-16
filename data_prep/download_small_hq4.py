"""Small HQ batch 4 - narrative / poetic genre diversity.

EN:
  garethpaul/children-stories-dataset (44MB) — children stories, narrative coherence
  deven367/babylm-100M-children-stories (12MB) — BabyLM curated children stories

CN:
  Iess/chinese_modern_poetry (186MB) — modern Chinese poetry, stylistic depth
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
    ("garethpaul/children-stories-dataset", "children-stories-en"),
    ("deven367/babylm-100M-children-stories", "babylm-children-stories"),
    ("Iess/chinese_modern_poetry", "chinese-modern-poetry"),
]:
    print(f"--- {repo_id} ---")
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=f"{ROOT}/{local_subdir}", max_workers=4)
        print(f"  done -> {ROOT}/{local_subdir}")
    except Exception as e:
        print(f"  FAIL: {e}")
print("Small HQ batch 4 downloads complete")
