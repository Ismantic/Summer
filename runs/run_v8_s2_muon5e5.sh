#!/usr/bin/env bash
# v8_s2_muon5e5: Muon @ 5e-5 control for the Aurora ablation.
# Same exact config as run_v8_s2_aurora.sh except --use_aurora removed.
# This is the LR that destroyed v5_s2; the test is whether the better
# Phase 1 starting point (v8 vs v5) alone is enough to survive, or
# whether Aurora's row-uniform polar is doing the heavy lifting.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/phase2_ckpt_v8_s2_muon5e5

echo "=== [$(date +%H:%M:%S)] PHASE 2 v8_s2_muon5e5: muon_lr=5e-5 + STANDARD MUON ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V8 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --warmup_steps 100 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] full eval on v8_s2_muon5e5 ==="
bash $SUMMER/runs/eval_full.sh phase2_ckpt_v8_s2_muon5e5
echo "=== [$(date +%H:%M:%S)] v8_s2_muon5e5 ALL DONE ==="
