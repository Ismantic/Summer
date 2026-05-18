#!/usr/bin/env bash
# v17 Phase 1 — 新 81903 piece 词表 + v15 配方（v7-mix 复现）。
#
# 与 v15 的差异（仅硬件/观测，训练数学等价）：
#  - 单卡 RTX 4090：--nproc_per_node=1（v15 是双卡 A6000）
#  - effective batch 严格保持 256：batch 16 × accum 16 × 1 卡（= v15 的 32×4×2）
#  - 加 --gradient_checkpointing 防 24GB OOM（纯显存优化，不改梯度）
#  - 不带 inline_eval（单卡装不下「训练 + vLLM eval」同时跑）；每 1000 步存 ckpt，
#    训练后用 retro_eval_mono_vllm.py / eval_with_piece_vllm.py 单独评测
#
# 其余超参（7500 步 / cosine / min_lr 0.01 / adam_lr 1e-4 / warmup 1000）= v15。
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok-v2
TRAIN_PT=$SUMMER/output/phase1_train_512_v17.pt
V17_P1=$SUMMER/output/phase1_ckpt_v17

echo "=== [$(date +%H:%M:%S)] PHASE 1 v17: 7500 steps frozen, v7-mix, cosine + min_lr 0.01 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29517 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V17_P1 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --max_steps 7500 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 100

echo "=== [$(date +%H:%M:%S)] v17 P1 DONE ==="
