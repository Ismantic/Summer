#!/usr/bin/env bash
# BERT-base MLM 预训:1B token POC,seqlen 1024,v18 piece vocab
# eff_bs = batch 16 × accum 8 × seqlen 1024 = 131,072 tok/step
# 1B / 131,072 ≈ 7,629 steps,留点余量训 8000 步
set -e
export PYTHONUNBUFFERED=1

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv/bin/python

INIT=$SUMMER/BERT/bert_init
TRAIN_PT=$SUMMER/output/v18_main_1024.pt   # 复用 v18 main(1B,seqlen 1024)
OUT=$SUMMER/BERT/bert_train_ckpt

echo "=== [$(date +%H:%M:%S)] BERT-base MLM POC: v18 piece + 1B token + seqlen 1024 ==="
$PY $SUMMER/BERT/train_bert_mlm.py \
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
