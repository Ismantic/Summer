#!/usr/bin/env bash
# Char-level RoBERTa-style MLM 预训(本目录独立,只用相对路径)。
#
# 流程:
#   1. python build_char_vocab.py --corpus <cn.txt> --output vocab.txt --min_freq 10
#   2. python encode_char_data.py --corpus <cn.txt> --vocab vocab.txt --output train.pt
#                                  --seq_len 512 --total_tokens 1000000000
#   3. python build_bert_init.py --vocab vocab.txt --output_dir bert_init
#   4. bash run_bert_train.sh ./bert_init ./train.pt ./bert_train_ckpt
#
# eff_bs = 32 × 8 × 512 = 131,072 字 / step,1B 字 → ~7,629 steps,取整 8000。
set -e
export PYTHONUNBUFFERED=1

INIT=${1:-./bert_init}
TRAIN_PT=${2:?usage: bash run_bert_train.sh <init_dir> <train_data.pt> [output_dir]}
OUT=${3:-./bert_train_ckpt}

PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "=== [$(date +%H:%M:%S)] Char-RoBERTa MLM: init=$INIT data=$TRAIN_PT out=$OUT ==="
$PYTHON $HERE/train_bert_mlm.py \
    --model_path $INIT \
    --train_data $TRAIN_PT \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --max_steps 8000 \
    --warmup_steps 1000 \
    --lr 1e-4 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --mlm_prob 0.15 \
    --save_steps 2000 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] BERT TRAIN DONE ==="
