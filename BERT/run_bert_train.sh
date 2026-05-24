#!/usr/bin/env bash
# BERT-base MLM 预训(独立可移植脚本,只用本目录相对路径 + 外部传入的数据)
#
# 用法:
#   bash run_bert_train.sh <init_dir> <train_data.pt> <output_dir>
#   bash run_bert_train.sh ./bert_init /path/to/train.pt ./bert_train_ckpt
#
# eff_bs = batch 16 × accum 8 × seqlen 1024 = 131,072 tok/step
# 1B token / 131k ≈ 7,629 steps,设 8000 步
set -e
export PYTHONUNBUFFERED=1

INIT=${1:-./bert_init}
TRAIN_PT=${2:?usage: bash run_bert_train.sh <init_dir> <train_data.pt> [output_dir]}
OUT=${3:-./bert_train_ckpt}

PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "=== [$(date +%H:%M:%S)] BERT-base MLM: init=$INIT data=$TRAIN_PT out=$OUT ==="
$PYTHON $HERE/train_bert_mlm.py \
    --model_path $INIT \
    --train_data $TRAIN_PT \
    --output_dir $OUT \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --max_steps 8000 \
    --warmup_steps 500 \
    --lr 1e-4 \
    --min_lr_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --mlm_prob 0.15 \
    --save_steps 2000 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] BERT TRAIN DONE ==="
