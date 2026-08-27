#!/usr/bin/env python3
"""Prepare a clean Hugging Face upload directory for v18 ReTok."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output" / "phase2_ckpt_v18_tie"
DEFAULT_OUT = ROOT / "save" / "releases" / "Qwen3-1.7B-Base-ReTok"
DEFAULT_REPO_ID = "Ismantic/Qwen3-1.7B-Base-ReTok"

RETOK_CARD = """---
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
ids = tok.encode("机器翻译的基本任务是")
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

ids = tok.encode("机器翻译的基本任务是", add_special_tokens=False)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                   SamplingParams(temperature=0.0, max_tokens=64,
                                  stop_token_ids=tok.stop_token_ids))
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

prompt = "机器翻译的基本任务是"
__BOS_LINE__
print(f"{prompt!r} -> {len(ids)} tokens")

# 贪心续写 40 步。没有 KV cache —— 每步重算前缀,短续写够用。
x = torch.tensor([ids], device=next(model.parameters()).device)
out = []
with torch.no_grad():
    for _ in range(40):
        nxt = int(model(x)[0, -1].argmax())
        if nxt in tok.stop_token_ids:
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
    "机器翻译的基本任务是",
    "中文分词的目标是",
]

def encode(p):
    __BOS_LINE_VLLM__

outputs = llm.generate(
    [TokensPrompt(prompt_token_ids=encode(p)) for p in prompts],
    SamplingParams(temperature=0.0, max_tokens=64,
                   stop_token_ids=tok.stop_token_ids),
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


# **从零预训练那条线的模型卡。** 与 RETOK_CARD 是两个不同的模型,不能共用一份 ——
# 之前 export 只做 `MODEL_CARD.replace(旧名, 新名)`,换了标题却照抄正文,于是
# Summer-0.5B 的发布包里写着 `base_model: Qwen/Qwen3-1.7B-Base` 和
# `tags: retok / tokenizer-replacement`。**我们是随机初始化,没有 base_model,
# 也不是换词表。** 这种错不报错,但传上去就是一份错误的模型描述。
SCRATCH_CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- from-scratch
- pretraining
- bilingual
- piece-tokenizer
language:
- en
- zh
---

# {name}

{name} is a **from-scratch** bilingual (Chinese/English) base model:
524,336,128 parameters, trained from random initialization on
**{tokens} tokens** with a self-trained 81,903-piece tokenizer.

It is *not* a fine-tune or a tokenizer-swap of any existing model. The
architecture follows `Qwen/Qwen3-0.6B-Base` (28 layers / hidden 1024 /
GQA 16:8 / head_dim 128 / tied embeddings / RoPE theta 1e6), but every weight
starts from `N(0, 0.02)`.

{stage_desc}

## What to expect

**This is a {tokens}-token model.** For scale: `Qwen3-0.6B-Base` saw 36T tokens —
about 2,700x more. Treat the numbers below as what that budget buys, not as a
competitive result.

{results}

## Tokenizer

The tokenizer is a compiled C++ extension, **not** loadable by
`AutoTokenizer`. The release ships `tokenizer.py` and `example_load.py`:

```bash
pip install git+https://github.com/Ismantic/PieceTokenizer
python example_load.py
```

The model code (`model.py`, `checkpoint.py`) is bundled too — the package
depends only on `torch` plus the tokenizer extension, not on `transformers`.

## Training

Full pipeline, data mixes and every design decision (including the mistakes)
are documented in <https://github.com/Ismantic/Summer>. Notably
`docs/WHY.md` records why fp32 master weights are mandatory, why the learning
rate schedule is WSD rather than cosine, and what the vocabulary swap cost.

## License

Apache-2.0. Training corpora are public datasets (FineWeb-Edu, Cosmopedia,
CCI3-HQ, SkyPile, WMT19, OPUS-100 and others; see `data/source.py` upstream).
Please observe their respective licenses.
"""

S0_DESC = """## Stage

**S0** — monolingual only, from-scratch pretraining. 14.6B tokens, English
70% / Chinese 30%, 55,694 steps.

This is a retrain of the original `Summer-0.5B-S0`, redesigned to align with
nanochat's actual training recipe:

- **English is a single source** (FineWebEdu, `HuggingFaceFW/fineweb-edu`),
  matching nanochat's `karpathy/fineweb-edu-100b-shuffle` — not the five-source
  blend the original release used.
- **Chinese is still multiple sources**, but screened first: two sources
  found to be 51.8% / 55.7% Traditional-Chinese-dominant by sampled character
  frequency (against a downstream instruction mix that is 100% Simplified)
  were dropped rather than mixed in.
- **Packing is BOS-aligned best-fit** (`bos_bestfit`): every training row
  starts with `<bos>` and packs whole documents greedily by best fit, cropping
  only the one document (if any) that doesn't fit the remainder — the same
  algorithm as nanochat's `tokenizing_distributed_data_loader_with_state_bos_bestfit`,
  verified line-by-line against nanochat's source. The original release used a
  continuous stream with no BOS token at all.
- **Sequence length is 2048**, matching nanochat (`max_seq_len=2048` since its
  first commit, used unchanged at every depth from d4 to d26+).

**Every input to this model must start with `<bos>`.** It has never seen a
sequence that doesn't. `example_load.py` / `example_vllm.py` in this repo do
this for you — if you tokenize text yourself without prepending `<bos>`, the
model will produce degenerate repetitive output regardless of prompt.

No parallel or instruction data at any point."""

S1_DESC = """## Stage

**S1** — S0 plus a parallel-data anneal, same nanochat-aligned recipe as
`Summer-0.5B-S0` (BOS-aligned best-fit packing, seq_len 2048). Branched from
S0's **final** checkpoint (optimizer state reset — verified by A/B test to
behave identically to resuming from a checkpoint with saved optimizer
momentum) and trained for 5,103 steps on 1.34B tokens containing **~30%
Chinese-English parallel text**, packed the same BOS-aligned way as S0 (the
anneal data must match the pretraining packing convention, or the model sees
an out-of-distribution input shift mid-training).

**Every input must start with `<bos>`**, same as `Summer-0.5B-S0` — see that
model's card for why."""

S0_RESULTS = """### Letter multiple-choice (nanochat's primary format — render the question, model answers a single letter)

| | previous release | this release |
|---|---|---|
| ARC-Easy | 0.2563 | 0.2668 |
| ARC-Challenge | 0.2338 | 0.2654 |
| MMLU | 0.2306 | 0.2551 |
| C-Eval | 0.2166 | 0.2527 |

This release scores above the random baseline (0.25) on all four tasks; the
previous release was below it on three of four.

### Likelihood-scoring protocol (lm-evaluation-harness convention)

| | previous release | this release |
|---|---|---|
| ARC-Easy (acc_norm) | 0.4949 | 0.5391 |
| ARC-Challenge (acc_norm) | 0.2671 | 0.3003 |
| MMLU (acc) | 0.2520 | 0.2465 |
| C-Eval (acc_norm) | 0.2363 | 0.2348 |
| GSM8K 8-shot (flexible-extract) | 0.0121 | 0.0167 |

### WMT22 5-shot translation

| | BLEU | COMET |
|---|---|---|
| zh->en | 0.29 | 0.4354 |
| en->zh | 2.78 | 0.5505 |

**Few-shot translation is still essentially zero — expected, this checkpoint
has never seen parallel text.** The model produces fluent but off-topic
continuations rather than translations; it does not yet follow the in-context
examples. See `Summer-0.5B-S1` for the version where in-context translation
appears. Its value is as (a) a from-scratch bilingual base model in its own
right, and (b) a starting point for annealing / SFT."""

S1_RESULTS = """| WMT22 5-shot | BLEU | COMET |
|---|---|---|
| zh->en | 8.99 | 0.6883 |
| en->zh | 28.36 | 0.7736 |

Against this model's own pre-anneal state (S0, 0.29 / 2.78 BLEU) this is
qualitative: S0 ignores the in-context examples and produces off-topic
continuations, S1 actually translates. Against the *previous* `Summer-0.5B-S1`
release (8.99 / 27.29 BLEU, COMET 0.6855 / 0.7743) this new release performs
at parity — the nanochat-aligned data recipe did not cost any translation
quality while improving the base model on every other tracked metric."""

# S0/S1 都需要在推理时手动 prepend BOS —— bos_bestfit 打包训出来的模型,训练
# 时每一行都以 <bos> 开头,不加它是彻底的分布外输入(2026-08 在自己的评测
# 脚本上踩过:不加 BOS,似然协议的 arc_easy 从 0.54 掉到接近随机,WMT22 的
# 5-shot 翻译直接复读崩溃)。ReTok 是旧 tokenizer 换皮 + 连续预训练,从来没有
# 这个约定,不需要。
CARDS = {
    "Summer-0.5B-S0": dict(card=SCRATCH_CARD, tokens="14.6B",
                           stage_desc=S0_DESC, results=S0_RESULTS, needs_bos=True),
    "Summer-0.5B-S1": dict(card=SCRATCH_CARD, tokens="14.6B + 1.34B anneal",
                           stage_desc=S1_DESC, results=S1_RESULTS, needs_bos=True),
}


def render_card(repo_id: str) -> str:
    """按发布身份选模型卡。不认识的名字**报错,不要默默套 ReTok 那份**。"""
    name = repo_id.split("/")[-1]
    if name in CARDS:
        c = CARDS[name]
        return c["card"].format(name=name, tokens=c["tokens"],
                                stage_desc=c["stage_desc"], results=c["results"])
    if "ReTok" in name:
        return RETOK_CARD.replace(DEFAULT_REPO_ID, repo_id)
    raise SystemExit(
        f"没有 {name} 的模型卡。在 save/export.py 的 CARDS 里加一份 ——"
        f"套用别的模型那份不会报错,但传上去就是错的描述。")


def needs_bos(repo_id: str) -> bool:
    """这个模型是不是 bos_bestfit 打包训出来的 —— 训练时每行都以 <bos> 开头,
    推理时不加就是分布外输入。2026-08 在自己的评测脚本上踩过:不加 BOS,
    似然协议的 arc_easy 从 0.54 掉到接近随机,WMT22 5-shot 直接复读崩溃。
    发布包里的示例脚本必须按这个模型是不是这一类来决定加不加,不能通用一份。
    """
    name = repo_id.split("/")[-1]
    return CARDS.get(name, {}).get("needs_bos", False)


def render_example_load(repo_id: str) -> str:
    bos_line = (
        "ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)  "
        "# 这个模型训练时每行都以 <bos> 开头,不加是分布外输入——会生成退化的重复内容"
        if needs_bos(repo_id) else
        "ids = tok.encode(prompt, add_special_tokens=False)"
    )
    return EXAMPLE_LOAD.replace("__BOS_LINE__", bos_line)


def render_example_vllm(repo_id: str) -> str:
    bos_line = (
        "return [tok.bos_token_id] + tok.encode(p, add_special_tokens=False)  "
        "# 训练时每行都以 <bos> 开头,不加是分布外输入"
        if needs_bos(repo_id) else
        "return tok.encode(p, add_special_tokens=False)"
    )
    return EXAMPLE_VLLM.replace("__BOS_LINE_VLLM__", bos_line)


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


def _reexport_weights(src: Path, dst: Path, cfg_path: Path, dtype: str) -> None:
    """按 dtype 重写权重,并把 config.json 的 torch_dtype 改成一致。

    两者不一致会让下游按 config 的 dtype 去读一份别的 dtype 的张量 —— 不报错,
    只是悄悄降精度或占双倍显存。所以一起改,不分开。
    """
    import json

    import torch

    sys.path.insert(0, str(ROOT))
    from src.checkpoint import load_safetensors, save_safetensors

    want = getattr(torch, dtype)
    sd = load_safetensors(str(src))
    n_cast = 0
    for k, v in sd.items():
        if v.is_floating_point() and v.dtype != want:
            sd[k] = v.to(want)
            n_cast += 1
    save_safetensors(sd, str(dst))

    cfg = json.loads(cfg_path.read_text())
    cfg["torch_dtype"] = dtype
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"权重转为 {dtype}({n_cast} 个张量),config.json 的 torch_dtype 已同步")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking them.")
    # **发布 bf16,fp32 主权重留在 output/。**
    #
    # 训练用 fp32 主权重(bf16 的相对精度 2^-8=0.39%,退火末尾一步的更新只有
    # 权重量级的 0.15%,fp32 是必需的 —— 见 docs/WHY.md 第四点五)。但那是
    # **训练**的需要:发布包是给推理和下游微调用的,下游照样 autocast 到 bf16,
    # fp32 只是把体积翻倍。已发布的 ReTok 也是 bf16。
    #
    # 不覆盖源 checkpoint —— output/ 里的 fp32 是真相来源,转换只发生在导出。
    parser.add_argument("--dtype", choices=["keep", "bfloat16", "float16", "float32"],
                        default="keep", help="导出权重的 dtype(默认原样)")
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
        if dst == "model.safetensors" and args.dtype != "keep":
            continue          # 下面单独转
        link_or_copy(_pick(source, src_names), out / dst, args.copy)

    if args.dtype != "keep":
        _reexport_weights(_pick(source, "model.safetensors"),
                          out / "model.safetensors", out / "config.json", args.dtype)

    # **把模型代码一起带上** —— 发布包只依赖 torch + piece_tokenizer,
    # 用户不需要 transformers 也不需要 safetensors 库。BERTc 的发布包一直如此。
    # 文件名保持源文件的名字(tokenizer.py / model.py / checkpoint.py),
    # 不加 _wrapper 之类的后缀 —— 那是实现细节,对拿到包的人没意义。
    shutil.copy2(ROOT / "prepare" / "tokenizer.py", out / "tokenizer.py")
    shutil.copy2(ROOT / "src" / "model.py", out / "model.py")
    shutil.copy2(ROOT / "src" / "checkpoint.py", out / "checkpoint.py")
    write_text(out / "example_load.py", render_example_load(args.repo_id))
    write_text(out / "example_vllm.py", render_example_vllm(args.repo_id))
    # **不拷 lineage 报告。** 它记的是本机的复现路径(哪个目录、哪个脚本),
    # 里面全是本机绝对路径 —— 那是本机信息,不该跟着模型发出去。复现记录留在
    # 仓库的 docs/reports/v18_tie_lineage.md,模型卡给出仓库链接就够了。
    write_text(out / "README.md", render_card(args.repo_id))
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
