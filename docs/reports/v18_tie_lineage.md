# v18 Tie Model Lineage

This document records how `/home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18_tie`
was produced, what data and commands were used, and how each stage performed.

## Final Artifact

- Final model: `output/phase2_ckpt_v18_tie`
- Downstream copy: `/home/tfbao/Shiyu/Interpreter/ReTok/models/phase2_ckpt_v18_tie`
- Verified matching files: `model.safetensors`, `piece.model`, `token_mapping.json`
- Interpreter entrypoint: `/home/tfbao/Shiyu/Interpreter/ReTok/run_sft.sh`

`phase2_ckpt_v18_tie` is the tie-preserving Phase 2 model. It was selected for
Interpreter/ReTok because it keeps `tie_word_embeddings=true` after LoRA merge,
unlike the normal `phase2_ckpt_v18` run.

## Stage 0: Tokenizer Replacement

Tokenizer surgery replaced the original Qwen tokenizer with the Piece tokenizer.

- Log: `output/v18_replace.log`
- Base model: `/home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base`
- Piece tokenizer: `/home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece.model`
- Init output: `/home/tfbao/new/Qwen3-1.7B-Base-new-tok-v18`
- Old vocab size: `151643`
- New vocab size: `81903`
- Old embedding shape: `(151936, 2048)`
- New embedding shape: `(81903, 2048)`
- Mapping stats: one-to-one `54343` (`66.4%`), multi-to-one `26660`
  (`32.6%`), fallback `895` (`1.1%`)
- Embedding tie: `tie_word_embeddings=true`

## Data Preparation

Data was packed with `data_prep/pretokenize_v18.py` at sequence length `1024`.

Main corpus for Phase 1:

- Log: `output/v18_pretok_main.log`
- Output: `output/v18_main_1024.pt`
- Budget: `1,000,000,000` tokens
- Final size: `975,568` chunks, `998,981,632` tokens
- Mix: FineWebEdu `0.316`, Wikipedia_EN `0.189`, Gutenberg `0.105`,
  C4_EN `0.053`, SkyPile `0.126`, Wikipedia_CN `0.063`, CCI3-HQ `0.126`,
  C4_CN `0.021`

Anneal corpus for Phase 2:

- Log: `output/v18_pretok_anneal.log`
- Output: `output/v18_anneal_1024.pt`
- Budget: `200,000,000` tokens
- Final size: `195,300` chunks, `199,987,200` tokens
- Mix: FineWebEdu `0.20`, Cosmopedia `0.25`, Wikipedia_EN `0.15`,
  CN_FineWeb_Edu `0.25`, Wikipedia_CN `0.10`, CCI3-HQ `0.05`

## Training Pipeline

Phase 1 trained only the new embeddings while freezing the transformer.

- Script: `runs/run_v18_p1.sh`
- Init: `/home/tfbao/new/Qwen3-1.7B-Base-new-tok-v18`
- Data: `output/v18_main_1024.pt`
- Output: `output/phase1_ckpt_v18`
- Key settings: `--freeze_transformer`, `--max_seq_length 1024`,
  `--batch_size 16`, `--gradient_accumulation_steps 16`, `--max_steps 3815`,
  `--warmup_steps 250`, `--adam_lr 1e-4`, `--lr_schedule cosine`
- Log: `output/v18_p1_train.log`
- Runtime: `3815` steps in `90821.7s`

Phase 2 normal annealed with LoRA and Aurora, but the final merge broke the
embedding tie.

- Script: `runs/run_v18_p2.sh`
- Init: `output/phase1_ckpt_v18`
- Data: `output/v18_anneal_1024.pt`
- Output: `output/phase2_ckpt_v18`
- Key settings: `--use_lora --lora_r 16 --lora_alpha 32`,
  `--lora_target q_proj,v_proj`, `--use_aurora`, `--max_steps 1500`,
  `--warmup_steps 200`, `--muon_lr 5e-5`, `--adam_lr 5e-5`
- Final train loss: `2.1229`
- Log warning: merged input/output embeddings were no longer tied

Phase 2 tie-safe repeated the anneal run with embedding/head tie preservation.

- Script: `runs/run_v18_p2_tie.sh`
- Init: `output/phase1_ckpt_v18`
- Data: `output/v18_anneal_1024.pt`
- Output: `output/phase2_ckpt_v18_tie`
- Extra setting: `--lora_tie_embed_head`
- Trainable params: Aurora `3,211,264`, Adam `167,737,344`
- Final train loss: `2.3942`
- Runtime: `1500` steps in `18648.6s`
- Final save: LoRA merged into base while preserving tied embeddings

## Evaluation Results

Translation results use WMT22 1000-sample COMET unless noted.

| Stage | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B base | 22.3408 | 0.8122 | 38.3380 | 0.8597 |
| v18 Phase 1 | 20.2588 | 0.7821 | 35.1632 | 0.8276 |
| v18 Phase 2 | 20.5701 | 0.7933 | 36.0749 | 0.8433 |
| v18 Phase 2 tie | 20.4599 | 0.7933 | 36.0314 | 0.8444 |

WMT23 full-set translation:

| Stage | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
| --- | ---: | ---: | ---: | ---: |
| v18 Phase 1 | 19.1310 | 0.7767 | 38.8251 | 0.8198 |
| v18 Phase 2 | 19.6000 | 0.7814 | 40.7933 | 0.8373 |
| v18 Phase 2 tie | 19.6046 | 0.7834 | 40.9905 | 0.8377 |

General benchmark results from `eval_results/full/*/result.json`:

| Stage | LAMBADA | PIQA | ARC-C | HellaSwag | CEVAL | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v18 Phase 1 | 0.5674 | 0.7301 | 0.5137 | 0.6375 | 0.6263 | 0.0417 |
| v18 Phase 2 | 0.5791 | 0.7361 | 0.5171 | 0.6392 | 0.6218 | 0.0402 |
| v18 Phase 2 tie | 0.5768 | 0.7367 | 0.5145 | 0.6389 | 0.6204 | 0.0356 |

## Interpretation

Phase 1 recovered a usable model after tokenizer replacement, but it remained
below the original Qwen base on translation. Phase 2 annealing improved both
translation and most English benchmark scores over Phase 1. The tie-safe Phase 2
model was slightly behind normal Phase 2 on some general benchmarks, but it
matched or improved WMT23 translation and preserved the tied embedding invariant,
which is important for downstream SFT and packaging.

## Reproduction Checklist

1. Run tokenizer replacement with `tools/replace_tokenizer.py` or `make replace`
   using the v18 Piece tokenizer.
2. Pack Phase 1 data with `data_prep/pretokenize_v18.py --mix main`.
3. Pack Phase 2 data with `data_prep/pretokenize_v18.py --mix anneal`.
4. Run `bash runs/run_v18_p1.sh`.
5. Run `bash runs/run_v18_p2_tie.sh`.
6. Evaluate with `bash runs/run_v18_eval_one.sh` or the v18 chain launcher.
7. Copy `output/phase2_ckpt_v18_tie` into Interpreter/ReTok when needed.
