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

- `Summer-Tokenizer.pt` — the 81,903-piece vocabulary
- `Summer-Tokenizer.dict.txt` — **Chinese segmentation dictionary. Not optional.**
- `token_mapping.json` — pad / bos / eos ids
- `tokenizer.py` — the loader you should use

The model architecture loads through Transformers as Qwen3, but **the tokenizer
is not a standard Qwen tokenizer and `AutoTokenizer` will not work.** Use the
bundled wrapper:

```python
from tokenizer import PieceTokenizerWrapper
tok = PieceTokenizerWrapper(".")          # the directory holding these files
ids = tok.encode("中国科学院计算技术研究所在北京")
```

**Keep `Summer-Tokenizer.dict.txt` next to `Summer-Tokenizer.pt`.** Without it
Chinese text tokenizes to *different ids* — not just slower. Round-trip decoding
still returns the original string, so the breakage is silent; the model simply
receives input it was never trained on. The loader raises rather than falling back.

## Running with vLLM

vLLM loads the **weights** fine — `config.json` declares `Qwen3ForCausalLM`, so
vLLM uses its own Qwen3 implementation and maps weights by state-dict key. It
cannot use the **tokenizer**, so pass `skip_tokenizer_init=True` and feed token
ids yourself:

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from tokenizer import PieceTokenizerWrapper

tok = PieceTokenizerWrapper(".")
llm = LLM(model=".", skip_tokenizer_init=True, dtype="bfloat16")

ids = tok.encode("中国科学院计算技术研究所", add_special_tokens=False)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                   SamplingParams(temperature=0.0, max_tokens=64,
                                  stop_token_ids=[tok.eos_token_id]))
print(tok.decode(list(out[0].outputs[0].token_ids)))
```

See `example_vllm.py`. **`vllm serve` does not work out of the box** — the
OpenAI-compatible server needs to turn text into tokens and cannot do so with
this vocabulary; callers must send token ids.

## Files

| | |
|---|---|
| `model.safetensors` | weights, 310 tensors (tied — no `lm_head.weight`) |
| `Summer-Tokenizer.pt` | the 81,903-piece vocabulary, same file as in [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer)'s `save/` |
| `Summer-Tokenizer.dict.txt` | Chinese segmentation dictionary — **required** |
| `model.py` `checkpoint.py` `tokenizer.py` | pure-torch inference code, so **no `transformers` and no `safetensors` needed** |
| `example_load.py` `example_vllm.py` | runnable examples |

## Training Summary

1. Replaced the original Qwen3-1.7B-Base tokenizer with an 81,903-token Piece
   tokenizer.
2. Initialized new embeddings by mapping each new piece through the original
   Qwen tokenizer and averaging old embeddings.
3. Phase 1: trained new embeddings on about 999M packed tokens while freezing
   the transformer.
4. Phase 2: annealed on about 200M packed tokens with LoRA q/v adapters, Aurora,
   and tied embedding/head preservation.

The full reproduction record (data mix, hyperparameters, timings) lives in
the GitHub repo under `docs/reports/`.

## Evaluation

All numbers come from the vLLM backend. **Do not mix backends** — measured on
the same base model, lambada differs by 0.0223 between transformers and vLLM,
and the direction is not even consistent across tasks.

WMT22, 1000 samples, 5-shot (sacrebleu / COMET wmt22-da):

| Model | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B-Base | 22.34 | 0.8122 | 38.34 | 0.8597 |
| ReTok v18 Phase 1 | 20.26 | 0.7821 | 35.16 | 0.8276 |
| **ReTok v18 Phase 2 tie** (this model) | **20.46** | **0.7933** | **36.03** | **0.8444** |

WMT23, full set:

| Model | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| ReTok v18 Phase 1 | 19.13 | 0.7767 | 38.83 | 0.8198 |
| **ReTok v18 Phase 2 tie** (this model) | **19.60** | **0.7834** | **40.99** | **0.8377** |

**BLEU is quoted to two decimals on purpose.** vLLM's greedy decoding is not
reproducible: over 6 runs of the same checkpoint the BLEU range is 0.10–0.13,
so a difference of that order is noise, not a result. COMET is two orders of
magnitude more stable and is the more reliable of the two.

General benchmarks (lm-evaluation-harness):

| Model | LAMBADA | PIQA | ARC-C | HellaSwag | CEVAL | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B-Base | 0.6513 | 0.7731 | 0.5512 | 0.6705 | 0.6560 | 0.6710 |
| ReTok v18 Phase 1 | 0.5674 | 0.7301 | 0.5137 | 0.6375 | 0.6263 | 0.0341 |
| **ReTok v18 Phase 2 tie** (this model) | **0.5768** | **0.7367** | **0.5145** | **0.6389** | **0.6204** | **0.0349** |

Metrics: `acc` for LAMBADA and CEVAL, `acc_norm` for PIQA / ARC-C / HellaSwag,
`exact_match,strict-match` for GSM8K. Shots: 0 / 5 / 25 / 10 / 5 / 5.

**GSM8K is a known, permanent loss.** Replacing the vocabulary breaks Qwen3's
numeric tokenization, and neither phase recovers it. This is the price of the
new vocabulary, not a regression to chase.

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



EXAMPLE_VLLM = '''"""用 vLLM 跑这个模型。

    pip install vllm
    pip install git+https://github.com/Ismantic/PieceTokenizer
    python example_vllm.py

## 为什么要 skip_tokenizer_init

vLLM 能加载**权重** —— config.json 里 architectures 是 Qwen3ForCausalLM,
它用自己那份 Qwen3 实现,按 state_dict 的 key 名灌进去。

但它**认不了这个词表**:81903 的 piece 词表不是 HF 标准格式,AutoTokenizer
走不通。所以要 skip_tokenizer_init=True,自己编码好 token id 用 TokensPrompt
喂进去,拿回来的 id 自己解码。

推论:`vllm serve` 那种 OpenAI 兼容服务**开箱用不了** —— 它要把文本转成
token,而它转不了。调用方得自己传 token id。
"""
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from tokenizer import PieceTokenizerWrapper

HERE = "."

tok = PieceTokenizerWrapper(HERE)
llm = LLM(model=HERE, skip_tokenizer_init=True, dtype="bfloat16",
          gpu_memory_utilization=0.85, max_model_len=4096,
          trust_remote_code=True)

prompts = [
    "中国科学院计算技术研究所",
    "The capital of France is",
]

outputs = llm.generate(
    [TokensPrompt(prompt_token_ids=tok.encode(p, add_special_tokens=False))
     for p in prompts],
    SamplingParams(temperature=0.0, max_tokens=64,
                   stop_token_ids=[tok.eos_token_id]),
)

for prompt, out in zip(prompts, outputs):
    print(f"\\n{prompt}", end="")
    print(tok.decode(list(out.outputs[0].token_ids)))
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
    write_text(out / "example_vllm.py", EXAMPLE_VLLM)
    # **不拷 lineage 报告。** 它记的是本机的复现路径(哪个目录、哪个脚本),
    # 里面全是本机绝对路径 —— 那是本机信息,不该跟着模型发出去。复现记录留在
    # 仓库的 docs/reports/v18_tie_lineage.md,模型卡给出仓库链接就够了。
    write_text(out / "README.md", MODEL_CARD.replace(DEFAULT_REPO_ID, args.repo_id))
    write_text(out / ".gitattributes", GITATTRIBUTES)
    write_text(out / "requirements.txt", REQUIREMENTS)

    # **把不该在的文件清掉。** 发布目录是构建产物,不是攒东西的地方。
    # 少了这一步,「不再生成某个文件」就等于「那个文件永远留在这儿」——
    # 2026-07-27 就是这么把 training_lineage.md(整篇本机路径)又传到 HF 上的:
    # export 已经不生成它了,但旧的那份还躺在目录里,upload 照传。
    expected = ({dst for _, dst in FILES_TO_COPY} |
                {"tokenizer.py", "model.py", "checkpoint.py",
                 "example_load.py", "example_vllm.py",
                 "README.md", ".gitattributes", "requirements.txt"})
    for p in sorted(out.iterdir()):
        if p.name in expected:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  removed stale: {p.name}")

    print(f"Prepared HF upload directory: {out}")
    print("Excluded intermediate checkpoint-* directories.")


if __name__ == "__main__":
    main()
