# Evals Directory

This directory contains Python evaluation entrypoints and batch drivers.

- `eval_with_piece*.py`: mono-task evaluation for piece-tokenizer checkpoints.
- `eval_pretrain_translate*.py`: WMT translation evaluation.
- `eval_analysis.py`, `prefetch_eval_datasets.py`: reporting and dataset setup.
- `retro_eval*.py`: batch sweeps across checkpoints.

Run these from the repository root, for example:

```bash
python evals/prefetch_eval_datasets.py
python evals/eval_with_piece_vllm.py --model_path <ckpt> --task mmlu --num_fewshot 5
```
