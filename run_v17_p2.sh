#!/usr/bin/env bash
# v17 Phase 2 — v16 配方：在 v17 Phase 1 ckpt 上解冻、Aurora、anneal 数据、2000 步。
#
# 与 v16 的差异（仅硬件/观测，训练数学等价）：
#  - 单卡 RTX 4090：--nproc_per_node=1（v16 是双卡 A6000）
#  - effective batch 严格保持 128：batch 8 × accum 16 × 1 卡（= v16 的 16×4×2）
#  - 加 --gradient_checkpointing 防 24GB OOM
#  - 不带 inline_eval；每 500 步存 ckpt，训练后单独评测
#
# 其余超参（2000 步 / cosine / min_lr 0.01 / muon_lr=adam_lr=5e-5 / warmup 200 /
# use_aurora）= v16。
#
# 依赖 output/v17_anneal_512.pt（pretokenize_v12.py --mix anneal，用新词表；
# 需先下载 Cosmopedia + CN_FineWeb_Edu 两个 v12-anneal 特有源）。
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V17_P1=$SUMMER/output/phase1_ckpt_v17
ANNEAL_PT=$SUMMER/output/v17_anneal_512.pt
V17_P2=$SUMMER/output/phase2_ckpt_v17

echo "=== [$(date +%H:%M:%S)] PHASE 2 v17: 2000 steps unfreeze Aurora anneal ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29518 finetune_muon.py \
    --model_path $V17_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V17_P2 \
    --max_seq_length 512 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --max_steps 2000 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 100

echo "=== [$(date +%H:%M:%S)] v17 P2 DONE ==="
