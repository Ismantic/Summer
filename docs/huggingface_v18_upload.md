# Hugging Face Upload: Qwen3-1.7B-Base-ReTok

This guide prepares and uploads the v18 tie-preserving model:
`output/phase2_ckpt_v18_tie`.

## Repository Name

Use:

```bash
Ismantic/Qwen3-1.7B-Base-ReTok
```

The base model is `Qwen/Qwen3-1.7B-Base`, whose Hugging Face model card lists
the license as Apache 2.0. Keep the derivative model under `apache-2.0`.

## What To Upload

Upload only the final merged model files from the root of
`output/phase2_ckpt_v18_tie`:

- `config.json`
- `generation_config.json`
- `model.safetensors`
- `piece.model`
- `dict.txt`
- `token_mapping.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- generated `README.md`
- generated `.gitattributes`
- generated `requirements.txt`
- generated `tokenizer_wrapper.py`
- generated `training_lineage.md`

Do not upload `checkpoint-500/`, `checkpoint-1000/`, or `checkpoint-1500/`.
Those are intermediate LoRA checkpoints, not the final release artifact.

## Prepare A Clean Upload Directory

```bash
/home/tfbao/.venv/bin/python tools/prepare_hf_v18_retok.py
```

This creates:

```bash
hf_upload/Qwen3-1.7B-Base-ReTok
```

By default the script uses hardlinks for large files, so it should not duplicate
the 3GB `model.safetensors` on the same filesystem. Add `--copy` if a real copy
is needed.

## Upload

Login once:

```bash
/home/tfbao/.venv/bin/huggingface-cli login
```

Create the model repo:

```bash
/home/tfbao/.venv/bin/huggingface-cli repo create Ismantic/Qwen3-1.7B-Base-ReTok --type model
```

Upload the prepared directory:

```bash
/home/tfbao/.venv/bin/huggingface-cli upload \
  Ismantic/Qwen3-1.7B-Base-ReTok \
  hf_upload/Qwen3-1.7B-Base-ReTok \
  . \
  --repo-type model
```

If `repo create` does not infer the organization correctly, create it explicitly
from the Hugging Face web UI under `Ismantic`, then run the same upload command.

## Compatibility Notes

The Qwen3 model weights load through Transformers, but the tokenizer is custom.
The local environment confirmed `AutoModelForCausalLM` can load the model config
and weights as Qwen3. `AutoTokenizer` is not the intended path because this
release uses `piece.model` plus `token_mapping.json`; generation should use
`tokenizer_wrapper.py` or the Summer/ReTok evaluation scripts.

Hosted HF inference may not work until the Piece tokenizer is packaged as a
standard `AutoTokenizer` implementation.
