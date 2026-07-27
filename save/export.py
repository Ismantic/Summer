#!/usr/bin/env python3
"""Prepare a clean Hugging Face upload directory for v18 ReTok."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output" / "phase2_ckpt_v18_tie"
DEFAULT_OUT = ROOT / "hf_upload" / "Qwen3-1.7B-Base-ReTok"
DEFAULT_REPO_ID = "Ismantic/Qwen3-1.7B-Base-ReTok"

MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- retok
- tokenizer-replacement
- continued-pretraining
- bilingual
language:
- en
- zh
---

# Qwen3-1.7B-Base-ReTok

Qwen3-1.7B-Base-ReTok is a tokenizer-replaced and continued-pretrained variant
of `Qwen/Qwen3-1.7B-Base`. The original Qwen tokenizer was replaced with a
custom Piece tokenizer, then the model was recovered with continued pretraining.

This is the final tie-preserving v18 checkpoint from the Summer/ReTok
experiments: `phase2_ckpt_v18_tie`.

Hugging Face repo id: `Ismantic/Qwen3-1.7B-Base-ReTok`

## Important Tokenizer Note

This repository contains the custom tokenizer assets:

- `piece.model`
- `dict.txt`
- `token_mapping.json`
- `tokenizer_wrapper.py`

The model architecture can be loaded by Transformers as Qwen3, but the tokenizer
is not a standard Qwen tokenizer. For generation, encode prompts with the
provided Piece tokenizer wrapper or the Summer/ReTok evaluation scripts.

## Training Summary

1. Replaced the original Qwen3-1.7B-Base tokenizer with an 81,903-token Piece
   tokenizer.
2. Initialized new embeddings by mapping each new piece through the original
   Qwen tokenizer and averaging old embeddings.
3. Phase 1: trained new embeddings on about 999M packed tokens while freezing
   the transformer.
4. Phase 2: annealed on about 200M packed tokens with LoRA q/v adapters, Aurora,
   and tied embedding/head preservation.

See `training_lineage.md` for the full reproduction record.

## Evaluation

WMT22 1000-sample translation:

| Model | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B-Base | 22.3408 | 0.8122 | 38.3380 | 0.8597 |
| ReTok v18 Phase 1 | 20.2588 | 0.7821 | 35.1632 | 0.8276 |
| ReTok v18 Phase 2 tie | 20.4599 | 0.7933 | 36.0314 | 0.8444 |

WMT23 full-set translation:

| Model | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| ReTok v18 Phase 1 | 19.1310 | 0.7767 | 38.8251 | 0.8198 |
| ReTok v18 Phase 2 tie | 19.6046 | 0.7834 | 40.9905 | 0.8377 |

General benchmark results:

| Model | LAMBADA | PIQA | ARC-C | HellaSwag | CEVAL | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReTok v18 Phase 1 | 0.5674 | 0.7301 | 0.5137 | 0.6375 | 0.6263 | 0.0417 |
| ReTok v18 Phase 2 tie | 0.5768 | 0.7367 | 0.5145 | 0.6389 | 0.6204 | 0.0356 |

## Limitations

- This is a base model, not an instruction-tuned assistant.
- Generic Hugging Face hosted inference may not work until the custom Piece
  tokenizer is packaged as a standard `AutoTokenizer` implementation.
- Results remain below the original Qwen3-1.7B-Base on the WMT22 translation
  sample after tokenizer replacement.

## License

The base model `Qwen/Qwen3-1.7B-Base` is released under Apache 2.0. This
derivative checkpoint is prepared with the same license.
"""


GITATTRIBUTES = """*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
"""


REQUIREMENTS = """torch
transformers>=4.56.0
safetensors
protobuf
"""


FILES_TO_COPY = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "piece.model",
    "dict.txt",
    "token_mapping.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking them.")
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    missing = [name for name in FILES_TO_COPY if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {source}: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    for name in FILES_TO_COPY:
        link_or_copy(source / name, out / name, args.copy)

    shutil.copy2(ROOT / "prepare" / "tokenizer.py", out / "tokenizer_wrapper.py")
    shutil.copy2(ROOT / "docs" / "reports" / "v18_tie_lineage.md", out / "training_lineage.md")
    write_text(out / "README.md", MODEL_CARD.replace(DEFAULT_REPO_ID, args.repo_id))
    write_text(out / ".gitattributes", GITATTRIBUTES)
    write_text(out / "requirements.txt", REQUIREMENTS)

    print(f"Prepared HF upload directory: {out}")
    print("Excluded intermediate checkpoint-* directories.")


if __name__ == "__main__":
    main()
