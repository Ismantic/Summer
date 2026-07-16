# Tools Directory

This directory contains one-off project utilities that mutate or derive model
artifacts.

- `build_v19_init.py`: initialize the v19 from-scratch model skeleton.
- `fix_piece_specials.py`: repair special-token metadata.
- `get_frozen_ids.py`: dump one-to-one mapped token IDs for gradient freezing.

Use these intentionally; they sit closer to model surgery than to routine
training or evaluation.
