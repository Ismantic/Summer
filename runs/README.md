# Runs Directory

This directory holds shell entrypoints for long-running experiments and
evaluation wrappers.

- `run_v*.sh`: versioned training and chaining scripts.
- `smoke_*.sh`: lightweight end-to-end checks.
- `inline_eval*.sh`: callbacks used from `train/finetune_muon.py --inline_eval_cmd`.
- `eval_*.sh`: full shell-driven evaluation batches.

Prefer calling these from the repository root, for example:

```bash
bash runs/run_v16.sh
bash runs/smoke_test.sh
```
