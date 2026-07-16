# Data Prep Directory

This directory contains corpus acquisition and token-packing utilities.

- `download_*.py`: fetch raw datasets or mirrors.
- `pretokenize*.py`: build packed `.pt` training tensors from raw corpora.
- `append_*_corpus.py`, `dump_p1_corpus.py`, `process_subtitle.py`: smaller
  corpus assembly helpers.

Run these from the repository root so existing relative paths and output
locations continue to work:

```bash
python data_prep/pretokenize_v19.py --help
python data_prep/download_data.py all
```
