"""Retry failed downloads.

1. iblai/ibl-khanacademy-transcripts (~86MB) — failed with ConnectTimeout
2. m-a-p/MAP-CC zh_books.part01 (37GB) — failed mid-stream at 313MB
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from huggingface_hub import snapshot_download

ROOT = "/mnt/data/Summer-data"

print("--- iblai/ibl-khanacademy-transcripts ---")
try:
    snapshot_download(
        repo_id="iblai/ibl-khanacademy-transcripts",
        repo_type="dataset",
        local_dir=f"{ROOT}/khanacademy-transcripts",
        max_workers=4,
    )
    print("  done")
except Exception as e:
    print(f"  FAIL: {e}")

print("--- m-a-p/MAP-CC zh_books.part01 + zh_baike (resume) ---")
try:
    snapshot_download(
        repo_id="m-a-p/MAP-CC",
        repo_type="dataset",
        local_dir=f"{ROOT}/MAP-CC",
        allow_patterns=[
            "zh_baike.jsonl.gz.part*",
            "zh_books.jsonl.gz.part01",
        ],
        max_workers=2,
    )
    print("  done")
except Exception as e:
    print(f"  FAIL: {e}")

print("Retries complete")
