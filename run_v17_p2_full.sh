#!/usr/bin/env bash
# v17 Phase 2 全参数版（对比 LoRA 用，明天跑）。
# 与 run_v17_p2.sh 的差异：
#  - 不带 --use_lora（全参数解冻,transformer 全训）
#  - batch 8 × accum 16 = 128 序列 × 1024（同 LoRA effective batch）
#    实测 batch 4 时显存仅 17GB（远低于估算），batch 8 预计 ~20-22GB，吃满显存
#    OOM 退回 batch 4 + accum 32
#  - output_dir 换名（不覆盖 LoRA P2 输出）
#
# 其余超参（1500 步 / cosine / min_lr 0.01 / muon_lr=adam_lr=5e-5 / warmup 200 /
# use_aurora / gradient_checkpointing）= LoRA 版（= v16）。
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V17_P1=$SUMMER/output/phase1_ckpt_v17
ANNEAL_PT=$SUMMER/output/v17_anneal_1024.pt
V17_P2_FULL=$SUMMER/output/phase2_ckpt_v17_full

echo "=== [$(date +%H:%M:%S)] PHASE 2 v17 FULL: 1.7B + 全参数 + anneal seqlen 1024 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29519 finetune_muon.py \
    --model_path $V17_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V17_P2_FULL \
    --max_seq_length 1024 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
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

echo "=== [$(date +%H:%M:%S)] v17 P2 FULL DONE ==="
