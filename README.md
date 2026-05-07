# Summer

Tokenizer-replacement experiment on `Qwen3-0.6B-Base` / `Qwen3-1.7B-Base`,
following the ReTok paper (Gu et al., [arXiv:2410.04335](https://arxiv.org/abs/2410.04335)).
Goal: produce a Qwen-derived base model with a custom piece tokenizer (vocab
65007), which the sister project [Interpreter](../Interpreter) will use as its
starting point for a fully self-built zh↔en translation model.

## Pipeline

```
Qwen3-*-Base ──┐
               ├─► replace_tokenizer.py ─► Qwen3-*-Base-new-tok/
piece_mt.model ┘     (resized embeds, piece.model, token_mapping.json)
                                    │
                                    ▼
                  finetune_muon.py phase 1 (CLM, freeze transformer)
                                    │
                                    ▼
              lm-evaluation-harness on ReTok Table 2 benchmarks
                  (eval-base via lm_eval CLI; eval-new via eval_with_piece.py)
```

Phase 1 training is intentionally restricted to `embed_tokens` + `lm_head`
(transformer frozen) so the ~26% multi-mapped / fallback rows can be learned
without disturbing the original Qwen weights. Phase 2 (full fine-tune) is
optional and not part of this subproject — the goal is producing a base, not a
chat model.

## Layout

```
Makefile                 # download / replace / eval-base / eval-new
replace_tokenizer.py     # embedding-mapping surgery (one-to-one ≈ 73.5%)
finetune_muon.py         # CLM / SFT trainer with Muon + AdamW split
muon.py                  # local Muon optimizer
pretokenize.py           # pack text into [N, seq_len] int32 chunks
get_frozen_ids.py        # dump IDs of one-to-one mapped tokens
tokenizer_wrapper.py     # HF-compatible PieceTokenizerWrapper
eval.py                  # WMT BLEU/COMET (legacy — Interpreter style)
eval_with_piece.py       # lm-eval-harness adapter (registers `hf_piece`)
smoke_test.sh            # parallel base-vs-new smoke across 9 tasks
piece_mt.model           # custom tokenizer (incl. <pad>/<user>/...)
dict.txt                 # 320K-entry CN segmentation dict (REQUIRED — see below)
```

## Dependencies

- `/home/tfbao/.venv` — Python 3.11 venv with `torch`, `transformers`,
  `lm_eval`, `accelerate`, `huggingface_hub`, `modelscope`.
- `piece_tokenizer` — built from sibling repo
  `/home/tfbao/Shiyu/PieceTokenizer` via `pip install -e .`. The Python
  binding's `load(model_file, cn_dict)` accepts the dict path as a second arg.
- Base model weights live at `/home/tfbao/new/Qwen3-0.6B-Base` and
  `/home/tfbao/new/Qwen3-1.7B-Base` (downloaded via ModelScope).

### `dict.txt` is mandatory

Without `cn_dict`, `PieceTokenizer::Encode` runs greedy BPE merges across the
whole input as one chunk — ~O(n²), ~100-1500× slower on long prompts (verified:
2.3K-char prompt 707ms → 2.4ms with the dict). The wrapper auto-loads
`dict.txt` if present in the model directory. After `make replace`, copy
`dict.txt` into the new-tok dir alongside `piece.model`.

## Common commands

```bash
make download    # huggingface-cli download → /home/tfbao/new/Qwen3-0.6B-Base
make replace     # tokenizer + embedding swap → Qwen3-0.6B-Base-new-tok/
make eval-base   # full ReTok Table 2 on base BBPE
make eval-new    # same on swapped piece tokenizer

bash smoke_test.sh    # parallel base/new across 9 tasks, --limit 20 each (~5min)
```

`smoke_test.sh` runs base on GPU 0 and new on GPU 1 in parallel. `eval-base`
auto-passes `--confirm_run_unsafe_code` for HumanEval; `eval-new` does the
same via `eval_with_piece.py`.

## Known smoke-test failures

- `cmmlu` — `datasets 4.x` removed `.py`-script-style dataset loading; the
  upstream `haonan-li/cmmlu` task config still uses one. Workaround: downgrade
  `datasets` to 3.x, or point the task at a parquet mirror.
- `humaneval` — also requires `HF_ALLOW_CODE_EVAL=1` (separate from lm_eval's
  `--confirm_run_unsafe_code`). Set it in env before running.

## Status

`make replace` works on Qwen3-0.6B-Base; smoke verifies pipeline end-to-end on
7/9 tasks. Phase 1 training (data + schedule) and full eval not yet started.
