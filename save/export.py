#!/usr/bin/env python3
"""Prepare a clean Hugging Face upload directory for v18 ReTok."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output" / "phase2_ckpt_v18_tie"
DEFAULT_OUT = ROOT / "save" / "releases" / "Qwen3-1.7B-Base-ReTok"
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

This is the final tie-preserving v18 checkpoint from the
[**Summer**](https://github.com/Ismantic/Summer) project.

| | |
|---|---|
| Code, recipes, full reproduction | https://github.com/Ismantic/Summer |
| Design notes and known pitfalls | [`docs/WHY.md`](https://github.com/Ismantic/Summer/blob/main/docs/WHY.md) |
| Tokenizer (C++, ships the 81,903 vocab) | https://github.com/Ismantic/PieceTokenizer |
| Downstream translation models | https://github.com/Ismantic/Interpreter |

Hugging Face repo id: `Ismantic/Qwen3-1.7B-Base-ReTok`

## Important Tokenizer Note

This repository contains the custom tokenizer assets:

- `piece.model` — the 81,903-piece vocabulary
- `dict.txt` — **Chinese segmentation dictionary. Not optional.**
- `token_mapping.json` — pad / bos / eos ids
- `tokenizer_wrapper.py` — the loader you should use

The model architecture loads through Transformers as Qwen3, but **the tokenizer
is not a standard Qwen tokenizer and `AutoTokenizer` will not work.** Use the
bundled wrapper:

```python
from tokenizer_wrapper import PieceTokenizerWrapper
tok = PieceTokenizerWrapper(".")          # the directory holding these files
ids = tok.encode("中国科学院计算技术研究所在北京")
```

**Keep `dict.txt` next to `piece.model`.** Without it Chinese text tokenizes to
*different ids* — not just slower. Round-trip decoding still returns the original
string, so the breakage is silent; the model simply receives input it was never
trained on. The wrapper raises rather than falling back.

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


# 发布包**只**要这两个。模型 forward、safetensors 读取都在随包的 model.py /
# checkpoint.py 里 —— 不需要 transformers,也不需要 safetensors 库。
REQUIREMENTS = """torch>=2.6
# 分词器(C++,同时提供 81903 词表):
#   pip install git+https://github.com/Ismantic/PieceTokenizer
"""


EXAMPLE_LOAD = '''"""跑通这个发布包 —— 只需要 torch 和 piece_tokenizer。

    pip install torch
    pip install git+https://github.com/Ismantic/PieceTokenizer
    python example_load.py
"""
import torch

from model import Qwen3ForCausalLM
from tokenizer import PieceTokenizerWrapper

HERE = "."

tok = PieceTokenizerWrapper(HERE)
model = Qwen3ForCausalLM.from_pretrained(
    HERE, device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16)

prompt = "中国科学院计算技术研究所"
ids = tok.encode(prompt, add_special_tokens=False)
print(f"{prompt!r} -> {len(ids)} tokens")

# 贪心续写 40 步。没有 KV cache —— 每步重算前缀,短续写够用。
x = torch.tensor([ids], device=next(model.parameters()).device)
out = []
with torch.no_grad():
    for _ in range(40):
        nxt = int(model(x)[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=x.device)], dim=1)

print(prompt + tok.decode(out))
'''


# 发布包里的文件名。**词表用上游名** —— 与 PieceTokenizer 仓库 save/ 下同名,
# 这样拿到发布包的人一眼就知道词表出自哪里(BERTc 的发布包里就叫
# BERTc-Tokenizer.pt)。源 checkpoint 里可能是旧名,所以按 (源名候选, 目标名) 写。
FILES_TO_COPY = [
    ("config.json", "config.json"),
    ("generation_config.json", "generation_config.json"),
    ("model.safetensors", "model.safetensors"),
    (("Summer-Tokenizer.pt", "piece.model"), "Summer-Tokenizer.pt"),
    (("Summer-Tokenizer.dict.txt", "dict.txt"), "Summer-Tokenizer.dict.txt"),
    ("token_mapping.json", "token_mapping.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
]


def _pick(source, names):
    for n in (names if isinstance(names, tuple) else (names,)):
        if (source / n).exists():
            return source / n
    return None


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

    missing = [dst for src_names, dst in FILES_TO_COPY
               if _pick(source, src_names) is None]
    if missing:
        raise FileNotFoundError(f"Missing required files in {source}: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    for src_names, dst in FILES_TO_COPY:
        link_or_copy(_pick(source, src_names), out / dst, args.copy)

    # **把模型代码一起带上** —— 发布包只依赖 torch + piece_tokenizer,
    # 用户不需要 transformers 也不需要 safetensors 库。BERTc 的发布包一直如此。
    # 文件名保持源文件的名字(tokenizer.py / model.py / checkpoint.py),
    # 不加 _wrapper 之类的后缀 —— 那是实现细节,对拿到包的人没意义。
    shutil.copy2(ROOT / "prepare" / "tokenizer.py", out / "tokenizer.py")
    shutil.copy2(ROOT / "src" / "model.py", out / "model.py")
    shutil.copy2(ROOT / "src" / "checkpoint.py", out / "checkpoint.py")
    write_text(out / "example_load.py", EXAMPLE_LOAD)
    # lineage 报告开头有一段给**仓库内读者**的时效声明(讲 core/ evals/ 那些
    # 目录改造后不存在了)。那段对只下模型的人没意义,拷进发布包反而突兀 ——
    # 剥掉开头连续的引用块。
    lineage = (ROOT / "docs" / "reports" / "v18_tie_lineage.md").read_text().split("\n")
    kept, i = [], 0
    # 保留一级标题
    while i < len(lineage) and not lineage[i].startswith("#"):
        i += 1
    if i < len(lineage):
        kept.append(lineage[i])
        i += 1
    # 跳过标题之后的空行 + 第一个引用块
    while i < len(lineage) and not lineage[i].strip():
        i += 1
    while i < len(lineage) and lineage[i].startswith(">"):
        i += 1
    write_text(out / "training_lineage.md",
               "\n".join(kept + lineage[i:]).lstrip("\n"))
    write_text(out / "README.md", MODEL_CARD.replace(DEFAULT_REPO_ID, args.repo_id))
    write_text(out / ".gitattributes", GITATTRIBUTES)
    write_text(out / "requirements.txt", REQUIREMENTS)

    print(f"Prepared HF upload directory: {out}")
    print("Excluded intermediate checkpoint-* directories.")


if __name__ == "__main__":
    main()
