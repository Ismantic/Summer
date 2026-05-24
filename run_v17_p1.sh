#!/usr/bin/env bash
# v17 Phase 1 — Qwen3-1.7B-Base + 新 81903 piece 词表 + v15 配方（v7-mix 复现）。
#
# 与 v15 的差异：
#  - 基座换成 Qwen3-1.7B-Base（v15 用的是 0.6B）
#  - 单卡 RTX 4090：--nproc_per_node=1（v15 是双卡 A6000）
#  - seqlen 1024（v15 是 512）—— GPT-2 级别，更贴近现代预训练；单卡 24GB 可承受的上限
#  - effective batch 保持「每步 131072 tokens」（= v15 的 256seq × 512）：
#    batch 16 × accum 8 × 1 卡 = 128 序列 × 1024 = 131072 tokens/step
#    （batch 8 时显存仅 10.5GB，太保守；16 约 16GB）。OOM 就降 batch、同步抬 accum
#  - --gradient_checkpointing 防 24GB OOM
#  - 不带 inline_eval；每 1000 步存 ckpt，训练后用 eval_with_piece_vllm.py 单独评测
#
# 走通流程版：3000 步 × 131072 ≈ 0.39B token（约 v15 训练量 0.98B 的 40%）。
# 其余超参（cosine / min_lr 0.01 / adam_lr 1e-4 / warmup 1000）= v15。
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
TRAIN_PT=$SUMMER/output/phase1_train_1024_v17.pt
V17_P1=$SUMMER/output/phase1_ckpt_v17

echo "=== [$(date +%H:%M:%S)] PHASE 1 v17: 1.7B, seqlen 1024, 3000 steps frozen, cosine + min_lr 0.01 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29517 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V17_P1 \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --max_steps 3000 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 10

echo "=== [$(date +%H:%M:%S)] v17 P1 DONE ==="
