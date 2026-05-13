"""Auto-retry wrapper for snapshot_download — retries on transient network errors.

Use this for unstable downloads. Sleeps 10s between attempts, max 20 attempts.
"""
import argparse
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--local_dir", required=True)
    p.add_argument("--allow_patterns", nargs="+", default=None)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=20)
    p.add_argument("--sleep_s", type=int, default=10)
    args = p.parse_args()

    for attempt in range(1, args.max_retries + 1):
        try:
            print(f"[attempt {attempt}/{args.max_retries}] downloading {args.repo_id}...", flush=True)
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=args.local_dir,
                allow_patterns=args.allow_patterns,
                max_workers=args.max_workers,
            )
            print(f"  done ({args.repo_id})", flush=True)
            return 0
        except Exception as e:
            print(f"  ATTEMPT {attempt} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt < args.max_retries:
                print(f"  sleeping {args.sleep_s}s before retry...", flush=True)
                time.sleep(args.sleep_s)
    print(f"GIVING UP on {args.repo_id} after {args.max_retries} attempts", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
