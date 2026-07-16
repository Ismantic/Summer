"""Small HQ batch 5 - domain/genre coverage expansion.

EN:
  iblai/ibl-khanacademy-transcripts (86MB) — Khan Academy educational transcripts
  BetterHF/wikipedia-biography-dataset (563MB) — wiki bios (narrative+factual)

CN:
  twang2218/chinese-law-and-regulations (153MB) — CN legal/regulatory (formal style)
  yuyijiong/booksum-zh (215MB) — CN book summaries (compact literary)
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
    ("iblai/ibl-khanacademy-transcripts", "khanacademy-transcripts"),
    ("BetterHF/wikipedia-biography-dataset", "wikipedia-biography"),
    ("twang2218/chinese-law-and-regulations", "chinese-law"),
    ("yuyijiong/booksum-zh", "booksum-zh"),
]:
    print(f"--- {repo_id} ---")
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=f"{ROOT}/{local_subdir}", max_workers=4)
        print(f"  done -> {ROOT}/{local_subdir}")
    except Exception as e:
        print(f"  FAIL: {e}")
print("Small HQ batch 5 downloads complete")
