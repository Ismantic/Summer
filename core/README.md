# Core Directory

This directory contains shared runtime modules used by training and evaluation.

- `tokenizer_wrapper.py`: HuggingFace-like shim for the piece tokenizer.
- `muon.py`: Muon optimizer implementations.
- `aurora.py`: Aurora update rule used on top of Muon-style state.

Keep these stable; many entrypoints in `train/`, `evals/`, and `analysis/`
import them directly.
