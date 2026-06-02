# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Summer reproduces the **ReTok** methodology ([arXiv:2410.04335](https://arxiv.org/abs/2410.04335)) on `Qwen3-0.6B-Base`: swap the native HuggingFace BBPE tokenizer (~151K vocab) for a custom **piece tokenizer** (vocab 65007) and recover model quality with continued pretraining. The output is a Qwen-derived *base* model that the sibling project `../Interpreter` uses as the starting point for a self-built zh↔en translation model. This is not a chat model — no instruction tuning is in scope.

Two-stage training:
- **Phase 1** — freeze the transformer, train only `embed_tokens` + `lm_head` so the ~26% multi-mapped/fallback embedding rows can be learned without disturbing original Qwen weights.
- **Phase 2** — unfreeze (full or LoRA), joint-train. Full unfreeze was destructive in the Qwen3-0.6B / Muon-AdamW split era (v7–v16); the winning P2 recipe ended up being **LoRA tie-safe** on Qwen3-1.7B (`v17`–`v18`).

Run versions are tracked as `v7`…`v19` (see `run_v*.sh`). **Current overall SOTA: `v18_p2_tie`** — Qwen3-1.7B + new 81903-piece tokenizer + LoRA tie-safe Phase 2 (1500 steps). WMT22 zh-en BLEU **20.46** vs base **22.34** (−1.88). Ckpt: `output/phase2_ckpt_v18_tie/checkpoint-1500/` (self-contained: `piece.model` + `dict.txt` + `adapter_model.safetensors` + `token_mapping.json`). Eval artifacts: `eval_results/full/v18_p2_tie_vllm/` + `eval_results/translate_wmt22/v18_p2_tie_*.json`.

Earlier milestones (kept for context): Phase 1 SOTA `v15` and P2 anneal `v16` on Qwen3-0.6B; `v17` switched to Qwen3-1.7B + LoRA; `v18` added the new 81903 piece + tie-safe LoRA. `v19` is an unrun from-scratch experiment. `program.md` is an autonomous-research loop spec for iterating on Phase 2.

## Environment & external paths

These are absolute and assumed by nearly every script — do not relativize them:
- Python venv: `/home/tfbao/.venv/bin/python` (Python 3.11; `torch`, `transformers`, `vllm`, `lm_eval`, `accelerate`).
- `piece_tokenizer` — built from sibling repo `/home/tfbao/Shiyu/PieceTokenizer` (`pip install -e .`). Its `load(model_file, cn_dict)` takes the CN segmentation dict as a second arg.
- Base model weights: `/home/tfbao/new/Qwen3-0.6B-Base`, `/home/tfbao/new/Qwen3-1.7B-Base`.
- Tokenizer-swapped model: `/home/tfbao/new/Qwen3-0.6B-Base-new-tok`.
- Checkpoints and pretokenized data live under `./output/`; eval artifacts under `./eval_results/`. Both are gitignored.

**`dict.txt` is mandatory.** Without the CN segmentation dict, `PieceTokenizer::Encode` runs greedy BPE over the whole input as one chunk (~O(n²), 100–1500× slower on long prompts). `tokenizer_wrapper.py` auto-loads `dict.txt` if present in the model dir; after `make replace`, copy `dict.txt` into the new-tok dir alongside `piece.model`.

## Commands

```bash
# --- Setup / tokenizer surgery (Makefile) ---
make download    # huggingface-cli download Qwen3-0.6B-Base
make replace     # replace_tokenizer.py: tokenizer + embedding swap -> *-new-tok/
make eval-base   # ReTok Table 2 benchmarks on base BBPE (lm_eval CLI)
make eval-new    # same on swapped piece tokenizer (eval_with_piece.py adapter)

# --- Training (a Phase runs via torchrun on 2 GPUs) ---
bash run_v16.sh  # each run_v*.sh is a self-contained hyperparameter script

# --- Evaluation (vLLM pipeline — see EVAL_README.md) ---
python prefetch_eval_datasets.py        # one-time: pull mmlu/ceval/gsm8k datasets
bash smoke_new_tasks.sh                 # ~2min sanity check (--limit 30)
python retro_eval_mono_vllm.py --gpus 0 1   # full retro sweep, ckpts sharded across GPUs
python eval_analysis.py                 # build the % loss vs base comparison table

# --- Single task on one ckpt ---
python eval_with_piece_vllm.py --model_path <ckpt> --task mmlu --num_fewshot 5
# Batch (loads vLLM once, runs all tasks back-to-back):
python eval_with_piece_vllm.py --model_path <ckpt> --tasks "lambada:0,piqa:5,mmlu:5"
```

There is no test suite — `smoke_test.sh` / `smoke_new_tasks.sh` are end-to-end harness sanity checks (small `--limit`), not unit tests.

## Architecture

**`replace_tokenizer.py`** — the irreversible surgery. `build_embedding_mapping` maps each new-vocab piece to old BBPE embeddings: one-to-one (~73.5%), multi-to-one averaged, or mean-vector fallback. Writes `piece.model`, `token_mapping.json`, resized weights. Changing this mapping invalidates every existing checkpoint.

**`finetune_muon.py`** — the trainer (the file you touch most for experiments). `--mode clm` for pretraining, `sft` for JSONL. Optimizer split: params with `ndim >= 2` and not `embed`/`lm_head` go to Muon/Aurora; embed + head go to an auxiliary AdamW (`SingleDeviceMuonWithAuxAdam` from `muon.py`). `--freeze_transformer` collapses this to plain AdamW on embed+head only (Phase 1). `--freeze_mapped_embeds` zeros grads for one-to-one-mapped rows via a backward hook. `--inline_eval_cmd` runs an eval script at every save with `{step}` substituted. LR schedule: `cosine`/`linear` with `--min_lr_ratio` (never decay to 0 — that was a known pathology).

**`muon.py` / `aurora.py`** — local optimizers. Aurora is a leverage-uniform polar variant of Muon (`--use_aurora`); per `program.md` it is the default for any Phase-2-style unfrozen training. `--moonshot_scaling` adds Moonshot's per-param LR scaling, orthogonal to the optimizer choice.

**`tokenizer_wrapper.py`** — `PieceTokenizerWrapper`, an HF-`PreTrainedTokenizer`-compatible shim over the `piece_tokenizer` C++ binding. Used everywhere the piece model meets HF/lm-eval code.

**`pretokenize*.py`** — pack raw text into `[N, seq_len]` int32 tensors saved as `.pt`. Versioned (`_v7`, `_v9`, `_v10`, `_v12`) — each version is a different data mix / sampler. `download_*.py` fetch source corpora (Cosmopedia, Chinese-FineWeb-Edu, MAP-CC, etc.).

**Eval pipeline** — `eval_with_piece_vllm.py` is the primary entry: a `lm-eval-harness` `TemplateLM` subclass backed by vLLM, auto-detecting piece vs HF tokenizer. `eval_pretrain_translate_vllm.py` does WMT22 BLEU+COMET. `retro_eval_mono_vllm.py` / `retro_eval_vllm.py` are batch drivers (`--gpus 0 1` shards checkpoints across GPUs). `eval_analysis.py` aggregates into the final table. See `EVAL_README.md` for the deprecated non-vLLM files and why they were superseded.

## Critical conventions

- **Never mix eval backends in one comparison.** `transformers` and vLLM produce different bf16 attention kernels; on under-trained checkpoints this drifts results up to ±10% on `arc_challenge`. All vLLM-era numbers go under `eval_results/full/<tag>_vllm/`. This caused a real false conclusion once (see EVAL_README "Backend-stability notes").
- The eval harnesses (`eval_pretrain_translate*.py`, `eval_ppl.py`, `eval_with_piece*.py`, `smoke_test.sh`) are **ground truth — do not edit them to make results look better.**
- Training launches with `torchrun --nproc_per_node=2` on 2× A6000. Run scripts background-launch and write tagged artifacts; long runs should not block.
- `gsm8k` is ~0 for every trained checkpoint — the 65K piece tokenizer disrupts Qwen3 numerical reasoning and neither phase recovers it. Treat that as a known limitation, not a regression.

## Autonomous research loop

`program.md` defines a self-directed Phase-2 experimentation loop (one-knob-at-a-time hypotheses, log to an untracked `experiments.tsv`, keep/discard against a BLEU+MMLU aggregate score). If asked to "run the loop" or iterate on Phase 2, follow that file's protocol, scope rules (what may/may not be modified), and scoring formula.
