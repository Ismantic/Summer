"""Download STEM datasets for v9 Phase 1 mix.

OpenWebMath (open-web-math/open-web-math) — ~50GB, 14B tokens
MathPile (GAIR/MathPile) — ~20GB, 10B tokens

Both via hf-mirror.com for fast download. Saved to /mnt/data/Summer-data/
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import argparse
from huggingface_hub import snapshot_download

ROOT = "/mnt/data/Summer-data"

JOBS = {
    "openwebmath": dict(
        repo_id="open-web-math/open-web-math",
        local_dir=f"{ROOT}/OpenWebMath",
        # Files have UUID hash suffixes: data/train-NNNNN-of-00114-HASH.parquet
        # Grab 24/114 ≈ 21% of dataset (~3B tok, ~10GB).
        allow_patterns=[f"data/train-{i:05d}-of-00114-*.parquet" for i in range(0, 24)],
    ),
    "mathpile": dict(
        repo_id="GAIR/MathPile",
        local_dir=f"{ROOT}/MathPile",
        # Train split only (skip validation). Covers arXiv / textbooks /
        # wikipedia / stackexchange / commoncrawl_filtered.
        allow_patterns=["train/*/*.jsonl.gz"],
    ),
}


def run(name):
    cfg = JOBS[name]
    print(f"[{name}] -> {cfg['local_dir']}")
    print(f"[{name}] patterns: {cfg['allow_patterns'][:3]}{'…' if len(cfg['allow_patterns'])>3 else ''}")
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        local_dir=cfg["local_dir"],
        allow_patterns=cfg["allow_patterns"],
        max_workers=4,
    )
    print(f"[{name}] done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=list(JOBS.keys()) + ["all"])
    args = p.parse_args()
    if args.job == "all":
        for j in JOBS:
            run(j)
    else:
        run(args.job)


if __name__ == "__main__":
    main()
