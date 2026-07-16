# Repository Guidelines

## Project Structure & Module Organization
Keep the top level clean: shared runtime modules live in `core/`, training entrypoints in `train/`, corpus download and packing scripts in `data_prep/`, Python evaluation entrypoints in `evals/`, diagnostics in `analysis/`, one-off artifact utilities in `tools/`, shell launchers in `runs/`, and supporting writeups in `docs/`. Reference papers live in `papers/`, tokenizer assets in `piece_*.model` and `dict.txt`, and generated artifacts belong in `output/` or `eval_results/`.

## Build, Test, and Development Commands
Use the checked-in `Makefile` for the main tokenizer replacement loop:

- `make download` downloads `Qwen/Qwen3-0.6B-Base` into the configured local path.
- `make replace` runs `tools/replace_tokenizer.py` and creates the `*-new-tok` model directory.
- `make eval-base` runs the baseline `lm_eval` benchmark set.
- `make eval-new` runs the piece-tokenizer adapter in `evals/eval_with_piece.py`.
- `bash runs/smoke_test.sh` compares base vs. swapped tokenizer on the reduced smoke suite.
- `bash runs/test_inline_eval.sh` checks the inline evaluation path before longer runs.

## Coding Style & Naming Conventions
Follow the style already used in the repo: 4-space indentation, snake_case for Python functions and filenames, and `UPPER_CASE` for shared constants such as dataset weights or path defaults. Keep new experiment files versioned and descriptive, for example `pretokenize_v20.py` or `run_v20_train.sh`. Prefer `argparse` for new Python entrypoints and make machine-specific paths overrideable with flags or environment variables.

## Testing Guidelines
There is no dedicated unit-test package here; validation is script-based. For tokenizer or evaluation changes, run `bash runs/smoke_test.sh` first, then the relevant `make eval-*` target. For data-pipeline or training edits, run the narrowest affected launcher in `runs/` and store logs under `output/` or `eval_results/`. If a benchmark requires code execution, document the needed environment variables, such as `HF_ALLOW_CODE_EVAL=1` for HumanEval.

## Commit & Pull Request Guidelines
Match the existing history's short, component-first subjects, such as `CLAUDE.md: ...`, `BERT/train_bert_mlm.py: ...`, or `v19 ...`. Keep each commit scoped to one experiment or tooling change. Pull requests should state the experiment goal, touched scripts, required local paths or GPU assumptions, commands executed, and where reviewers can find metrics or logs. Do not commit model weights, downloaded corpora, or other large generated artifacts.

## Environment Notes
Many scripts assume `/home/tfbao/...` paths and fixed CUDA devices. When updating shared scripts, preserve those defaults only if they remain overrideable via variables like `QWEN_BASE`, `QWEN_NEW`, `OUT`, and `LIMIT`.
