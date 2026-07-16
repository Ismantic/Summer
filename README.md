# Summer

Summer is a tokenizer-replacement and continued-pretraining workspace built
around the ReTok idea on Qwen base models. The main goal is to swap Qwen's
native tokenizer for a custom piece tokenizer, recover quality with continued
training, and produce a better base model for downstream zh↔en translation
work in the sibling `Interpreter` project.

This repository is experiment-heavy. The important distinction is:

- `tools/replace_tokenizer.py` does the one-time model surgery.
- `train/finetune_muon.py` is the main training entrypoint.
- `evals/` contains Python evaluation entrypoints.
- `runs/` contains versioned shell launchers and smoke checks.

## Repository Layout

```text
core/        shared runtime modules: Muon, Aurora, tokenizer wrapper
train/       training entrypoints
evals/       evaluation scripts and batch drivers
data_prep/   dataset download and pretokenization scripts
runs/        shell launchers, smoke tests, inline eval callbacks
analysis/    diagnostics and verification helpers
tools/       one-off utilities such as tokenizer replacement
docs/        reports, eval notes, and research plans
papers/      reference PDFs
output/      checkpoints and generated training artifacts
eval_results/ benchmark outputs and summaries
```

## Common Commands

```bash
make download               # fetch Qwen3-0.6B-Base
make replace                # build tokenizer-swapped model directory
make eval-base              # run baseline lm-eval tasks
make eval-new               # run the same tasks through evals/eval_with_piece.py

bash runs/smoke_test.sh     # quick base-vs-new sanity check
bash runs/run_v16.sh        # example training launcher
python evals/prefetch_eval_datasets.py
python evals/eval_with_piece_vllm.py --model_path <ckpt> --task mmlu --num_fewshot 5
```

## Environment Assumptions

Most scripts assume local paths under `/home/tfbao/...` and a prepared Python
environment at `/home/tfbao/.venv`. The custom `piece_tokenizer` package is
expected to be installed from the sibling repository
`/home/tfbao/Shiyu/PieceTokenizer`.

`dict.txt` is required for the piece tokenizer's Chinese pre-segmentation
path. Without it, encoding can degrade from normal throughput to effectively
quadratic behavior on long inputs.

## Where To Read More

- `docs/README.md` for document index
- `docs/reports/` for experiment writeups
- `docs/eval/pipeline.md` for the evaluation stack
- `docs/plans/autoresearch.md` for the autonomous experiment loop
- `CLAUDE.md` for repository-specific operator guidance
