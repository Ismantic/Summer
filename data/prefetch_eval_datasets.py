"""One-time download of all eval datasets used by our mono benchmark suite.

After running once (online), everything else can run with HF_HUB_OFFLINE=1.

Covers:
  - cais/mmlu                  (57 subjects)
  - haonan-li/cmmlu            (67 subjects, Chinese MMLU)
  - ceval/ceval-exam           (52 subjects, Chinese eval)
  - openai/gsm8k               (math word problems)
  - hails/mmlu_no_train        (lm_eval variant)
  - piqa, hellaswag, ai2_arc, lambada_openai  (already cached)
"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import time
from datasets import load_dataset, get_dataset_config_names


def download_all_configs(dataset_path: str, *, force_check_configs: bool = True):
    """Trigger download of all configs (subjects/subtasks) for a multi-config dataset."""
    print(f"\n=== {dataset_path} ===")
    t0 = time.time()
    try:
        configs = get_dataset_config_names(dataset_path)
    except Exception as e:
        # single-config dataset
        try:
            ds = load_dataset(dataset_path)
            print(f"  single-config, splits={list(ds.keys())} ({time.time()-t0:.1f}s)")
        except Exception as e2:
            print(f"  FAIL: {e2}")
        return
    print(f"  {len(configs)} configs: {configs[:3]}...")
    for cfg in configs:
        t1 = time.time()
        try:
            ds = load_dataset(dataset_path, cfg)
            n = sum(len(v) for v in ds.values())
            print(f"  ✓ {cfg}: {n} rows ({time.time()-t1:.1f}s)", flush=True)
        except Exception as e:
            print(f"  ✗ {cfg}: {e}", flush=True)
    print(f"=== total {time.time()-t0:.1f}s ===")


def main():
    # English benchmarks (most already cached, but verify)
    download_all_configs("cais/mmlu")           # ~57 subjects
    # Chinese
    download_all_configs("haonan-li/cmmlu")     # ~67 subjects
    download_all_configs("ceval/ceval-exam")    # ~52 subjects
    # Math / generative
    download_all_configs("openai/gsm8k")
    # lm_eval may also reference these
    download_all_configs("hails/mmlu_no_train")


if __name__ == "__main__":
    main()
