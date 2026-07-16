#!/usr/bin/env bash
# v19 pretokenize: 10B token,纯文本 mix(复用 v18 main mix:8 源 EN/CN 66/34)
# 用 v18 piece tokenizer(81903)+ dict.txt
set -e
export PYTHONUNBUFFERED=1

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv/bin/python

$PY $SUMMER/data_prep/pretokenize_v19.py \
    --tokenizer_model /home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece_fixed.model \
    --cn_dict /home/tfbao/Shiyu/PieceTokenizer/dict.txt \
    --output $SUMMER/output/v19_main_1024.pt \
    --total_tokens 10000000000 \
    --seq_length 1024 \
    --num_workers 28

echo "=== v19 pretok DONE ==="
