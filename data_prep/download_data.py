"""Resume / start dataset downloads to /mnt/data/Summer-data.

Each call is idempotent: completed files are skipped, partials resume via the
huggingface_hub cache. Run as:
    python data_prep/download_data.py skypile
    python data_prep/download_data.py fineweb
    python data_prep/download_data.py cci3
    python data_prep/download_data.py all
"""
import argparse
import os
import sys

# Use Chinese mirror — direct connection is ~14x faster than going through
# the proxy to huggingface.co from this network.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from huggingface_hub import snapshot_download

ROOT = "/mnt/data/Summer-data"

JOBS = {
    "skypile": dict(
        repo_id="Skywork/SkyPile-150B",
        local_dir=f"{ROOT}/SkyPile",
        # 2022-40/49 head_0000 + 2022-27/40/49 head_0001 (already partly in-flight)
        # + 2023-06/14 head_0000/0001
        allow_patterns=[
            "data/2022-40_zh_head_0000.jsonl",
            "data/2022-49_zh_head_0000.jsonl",
            "data/2022-27_zh_head_0001.jsonl",
            "data/2022-40_zh_head_0001.jsonl",
            "data/2022-49_zh_head_0001.jsonl",
            "data/2023-06_zh_head_0000.jsonl",
            "data/2023-06_zh_head_0001.jsonl",
            "data/2023-14_zh_head_0000.jsonl",
            "data/2023-14_zh_head_0001.jsonl",
        ],
    ),
    "fineweb": dict(
        repo_id="HuggingFaceFW/fineweb-edu",
        local_dir=f"{ROOT}/fineweb-edu",
        allow_patterns=[
            # 2024-10 first 5
            "data/CC-MAIN-2024-10/000_00000.parquet",
            "data/CC-MAIN-2024-10/000_00001.parquet",
            "data/CC-MAIN-2024-10/000_00002.parquet",
            "data/CC-MAIN-2024-10/000_00003.parquet",
            "data/CC-MAIN-2024-10/000_00004.parquet",
            # 2024-22 first 3
            "data/CC-MAIN-2024-22/000_00000.parquet",
            "data/CC-MAIN-2024-22/000_00001.parquet",
            "data/CC-MAIN-2024-22/000_00002.parquet",
        ],
    ),
    "cci3": dict(
        repo_id="BAAI/CCI3-HQ",
        local_dir=f"{ROOT}/CCI3-HQ",
        allow_patterns=[f"data/part_{i:06d}.jsonl" for i in range(5)],
    ),
}


def run(job_name):
    cfg = JOBS[job_name]
    print(f"[{job_name}] -> {cfg['local_dir']}")
    print(f"[{job_name}] {len(cfg['allow_patterns'])} files")
    for p in cfg["allow_patterns"]:
        print(f"  + {p}")
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        local_dir=cfg["local_dir"],
        allow_patterns=cfg["allow_patterns"],
        max_workers=4,
    )
    print(f"[{job_name}] done")


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
