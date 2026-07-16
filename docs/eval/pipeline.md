# Evaluation Pipeline

Unified vLLM-backed eval for the Summer (Qwen3 + piece tokenizer) project.

Covers translation (WMT22), English mono (PIQA/ARC/HellaSwag/MMLU/LAMBADA),
Chinese mono (C-Eval), and math (GSM8K) — 7 mono tasks + 2 translation
directions — all on one vLLM engine, multi-GPU parallel across checkpoints.

## TL;DR

```bash
# One-time: download all eval datasets (~3min)
python evals/prefetch_eval_datasets.py

# Smoke (~2min, --limit 30)
bash runs/smoke_new_tasks.sh

# Full retro on all ckpts (~55min on 2× A6000)
python evals/retro_eval_mono_vllm.py --gpus 0 1

# Comparison table
python evals/eval_analysis.py
```

## File map

| File | Purpose |
|---|---|
| `evals/eval_with_piece_vllm.py` | **Primary entry.** TemplateLM subclass that runs lm-eval-harness on a single ckpt via vLLM. Supports either `--task` (single) or `--tasks "lambada:0,piqa:5,..."` (batch — loads vLLM once, runs all tasks back-to-back). Auto-detects piece vs HF tokenizer. |
| `evals/eval_pretrain_translate_vllm.py` | WMT22 translation eval (BLEU + COMET) via vLLM PagedAttention. Uses `piece_tokenizer` + `skip_tokenizer_init`. |
| `evals/retro_eval_mono_vllm.py` | Batch retro driver: all listed ckpts × all listed tasks. `--gpus 0 1` shards ckpts across GPUs in a ThreadPoolExecutor (~2x wall-time). |
| `evals/retro_eval_vllm.py` | Same idea but for translation only (WMT22). |
| `runs/inline_eval_vllm_v2.sh` | Called from `train/finetune_muon.py --inline_eval_cmd` during training. Runs WMT22 every save, full mono suite at `--max_steps`. |
| `evals/prefetch_eval_datasets.py` | One-time online pull of mmlu / ceval / gsm8k / mmlu_no_train. Run once; subsequent eval stays offline. |
| `evals/eval_analysis.py` | Builds the final comparison table (% loss vs base for each ckpt × task). |
| `runs/smoke_new_tasks.sh` | `--limit 50` sanity check for new tasks before a full run. |
| `retest_base_mono.sh` | One-off helper to re-run base on the current `transformers` version (used to quantify the 4.57.6 BLEU-drift question). |

### Deprecated / superseded

| File | Replaced by | Why |
|---|---|---|
| `evals/eval_with_piece.py` | `evals/eval_with_piece_vllm.py` | transformers backend; bf16 attention kernels shift the argmax of under-trained ckpts. ckpt-dependent drift up to -10% on arc_c. |
| `runs/inline_eval_progressive.sh` | `runs/inline_eval_vllm_v2.sh` | uses old `eval_with_piece.py`; subject to backend drift. |
| `runs/inline_eval_vllm.sh` | `runs/inline_eval_vllm_v2.sh` | v1 only ran translation on vLLM; mono still on transformers. v2 routes everything through vLLM. |
| `evals/retro_eval_comet.py` | `evals/retro_eval_vllm.py` | superseded by vLLM translation. |

Kept for legacy ckpts whose results were already produced under those paths.

## Benchmark suite

| Task | Shots | Type | Dataset path |
|---|---|---|---|
| `piqa` | 5 | loglikelihood (2 choice) | `ybisk/piqa` |
| `hellaswag` | 10 | loglikelihood (4 choice) | `Rowan/hellaswag` |
| `arc_challenge` | 25 | loglikelihood (4 choice) | `allenai/ai2_arc` (`ARC-Challenge`) |
| `mmlu` | 5 | loglikelihood (4 choice × 57 subjects) | `cais/mmlu` |
| `lambada_openai` | 0 | last-word argmax (greedy) | `EleutherAI/lambada_openai` |
| `ceval-valid` | 5 | loglikelihood (4 choice × 52 subjects) | `ceval/ceval-exam` |
| `gsm8k` | 5 | `generate_until` + exact_match | `openai/gsm8k` |
| WMT22 zh↔en | 5 | BLEU (sacrebleu) + COMET (`Unbabel/wmt22-comet-da`) | local |

### Missing / TODO

- **CMMLU** (`haonan-li/cmmlu`, `lmlmcat/cmmlu`, `XiaHan19/cmmlu` all use a
  Python loading script that `datasets >= 4.0` deprecated). Workaround:
  download CSV files from upstream GitHub and write a local parquet wrapper.
- **HumanEval / MBPP** (code) — intentionally skipped per project decision to
  exclude code from the pretraining mix.
- **MATH-500, AGIEval, BBH** — present in YuLan-Mini / ReTok Table 2 but not
  yet in our suite.

## Performance characteristics

| Step | Cost | Notes |
|---|---|---|
| HF datasets HEAD (per task) | 10s timeout × N tasks | **eliminated** by `HF_HUB_OFFLINE=1` (set inside `evals/eval_with_piece_vllm.py`) |
| mmlu task-dict construct (57 subj) | 166s online → **2s offline** | same flags |
| vLLM model load | ~20s | amortized across all tasks via `--tasks` batch mode |
| arc_challenge inference (4687 reqs) | ~95s | vLLM continuous batching |
| mmlu inference (56K reqs) | ~3-5min | KV cache size 360K tokens @ `max_model_len=4096` |
| gsm8k generation (1319 docs) | ~80s | batched generate_until; was ~22min sequential |

Total one-ckpt full-suite: **~12-15min** (vs ~50min pre-optimization).
Eight-ckpt sweep on 2 GPUs: **~55min** (vs ~8h single-GPU).

## Backend-stability notes

All numbers comparing trained ckpts must be on the **same backend**.

- `transformers` and `vLLM` produce different bf16 attention kernels. On
  **base** the drift is ≤2.6% (only lambada-style argmax-cascade tasks affected).
- On **under-trained ckpts (v8 P1, etc.)** the drift can be -10% on arc_c
  because logit margins are smaller and a few argmax flips compound.
- Earlier-this-week confusion ("v15 is mono champion" vs "v15 is mono worst")
  was 100% backend artifact — same ckpt evaluated on transformers vs vLLM gave
  -5% vs -15% on arc_c. Lesson: **never mix backends in a comparison table**.

## Output structure

```
eval_results/full/
├── base_vllm/
│   ├── lambada_openai/result.json
│   ├── piqa/result.json
│   ├── ...
│   ├── wmt22.json          # translation
│   └── batch_run.log
├── v8_p1_vllm/
│   └── ...
├── v16_p2_step2000/        # legacy translation tag (eval_analysis.py
│                             special-cases this)
└── ...
```

`evals/eval_analysis.py` walks these and prints a unified `% loss vs base` table
for all ckpts × all 7 mono tasks + 4 translation metrics.

## Latest results (2026-05-15)

```
                  EN avg     ZH(ceval)    gsm8k     all avg     zh-en COMET    en-zh COMET
base               —           —            —         —           —              —
v8_p1            -14.34%     -27.82%     -100.0%   -28.50%     -11.86%        -8.74%
v10_p2           -14.15%     -23.39%      -95.9%   -27.15%     -10.33%        -8.48%
v11_p2           -13.90%     -29.97%     -100.0%   -28.50%     -10.96%        -8.94%
v12_p1           -14.35%     -24.33%      -95.8%   -27.41%     -11.79%        -8.18%
v12_p2           -13.52%     -24.33%      -94.5%   -26.64%     -10.43%        -7.66%
v15_p1           -11.79%     -22.58%     -100.0%   -25.94%      -7.86%        -6.49%
v16_p2  ★        -10.11%     -18.82%     -100.0%   -24.19%      -5.15%        -3.36%
```

v16 is best across all metrics. Notable: `gsm8k` is essentially zero for every
trained ckpt — the 65K piece tokenizer disrupted Qwen3's numerical reasoning
in a way that P1 frozen-transformer + P2 anneal does not recover. Compare
ReTok paper (Llama3 tokenizer + 15K Chinese expansion, no vocab replacement):
mmlu -3.0%, no math collapse.
