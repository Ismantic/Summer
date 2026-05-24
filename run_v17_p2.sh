#!/usr/bin/env bash
# v17 Phase 2 — Qwen3-1.7B + LoRA + v16 anneal mix(a6000 真数据)+ seqlen 1024。
#
# 配方（≈ v16）：
#  - 基座：Phase 1 ckpt v17（output/phase1_ckpt_v17）
#  - 数据：v17_anneal_1024.pt（v16 anneal 6 源 a6000 真数据，200M token）
#  - LoRA：r=16, alpha=32, target=q_proj,v_proj；embed_tokens + lm_head 全参训
#  - effective batch 128 序列 × 1024 = 131072 tokens/step（= P1 / v15）
#  - 1500 步 ≈ 197M token（≈ 1 epoch over the anneal pool）
#  - cosine + min_lr 0.01 + warmup 200 + muon_lr=adam_lr=5e-5（= v16）
#  - use_aurora（= v16 默认；LoRA 矩阵走 Muon/Aurora，embed/head 走 AdamW）
#  - gradient_checkpointing（防 24GB OOM）
#  - 单卡 RTX 4090
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V17_P1=$SUMMER/output/phase1_ckpt_v17
ANNEAL_PT=$SUMMER/output/v17_anneal_1024.pt
V17_P2=$SUMMER/output/phase2_ckpt_v17

echo "=== [$(date +%H:%M:%S)] PHASE 2 v17: 1.7B + LoRA + anneal seqlen 1024 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29518 finetune_muon.py \
    --model_path $V17_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V17_P2 \
    --max_seq_length 1024 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_target q_proj,v_proj \
    --max_steps 1500 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 10

echo "=== [$(date +%H:%M:%S)] v17 P2 DONE ==="
