#!/usr/bin/env bash
# v18 Phase 1: 1.7B + freeze transformer + NEW piece.model (81903) + v18 main mix
# 数据:1B token (v8 mix 归一化 a6000 路径,EN66/CN34),seqlen 1024
# 配方:cosine + min_lr 0.01(继承 v15 P1 SOTA);1.7B + GC + compile(继承 v17 P1)
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

INIT=/home/tfbao/new/Qwen3-1.7B-Base-new-tok-v18
MAIN_PT=$SUMMER/output/v18_main_1024.pt
V18_P1=$SUMMER/output/phase1_ckpt_v18

# 1B token / (batch 16 × accum 16 × 1024) = 3815 steps,走满 1 epoch(对齐 v15 1B/7500 步 seqlen 512)
echo "=== [$(date +%H:%M:%S)] PHASE 1 v18: 1.7B freeze + 1B token + seqlen 1024 + 3815 步 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29520 finetune_muon.py \
    --model_path $INIT \
    --train_data $MAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V18_P1 \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --compile \
    --max_steps 3815 \
    --warmup_steps 250 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] v18 P1 DONE ==="
