#!/usr/bin/env bash
# v9 Phase 2 — Aurora 500 steps from checkpoint-3000 (early-stopped Phase 1).
# Phase 1 saturated at ~step 1000; ckpt-3000 = 0.39B training tokens.
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
P1_CKPT=$SUMMER/output/phase1_ckpt_v9/checkpoint-3000
TRAIN_PT=$SUMMER/output/phase1_train_512_v9.pt
P2_OUT=$SUMMER/output/phase2_ckpt_v9_p1step3000_aurora

INLINE_EVAL="bash $SUMMER/runs/inline_eval_progressive.sh {step} $P2_OUT v9_p2_aurora_p1at3000 500"

echo "=== [$(date +%H:%M:%S)] PHASE 2 (Aurora 500) from phase1 step 3000 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29502 train/finetune_muon.py \
    --model_path $P1_CKPT \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $P2_OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --warmup_steps 100 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 50 \
    --inline_eval_cmd "$INLINE_EVAL"

echo "=== [$(date +%H:%M:%S)] eval phase2 (full) ==="
ln -sfn $P2_OUT/checkpoint-500 $SUMMER/output/phase2_ckpt_v9_p1step3000_aurora_final
bash $SUMMER/runs/eval_full.sh phase2_ckpt_v9_p1step3000_aurora_final

echo "=== [$(date +%H:%M:%S)] v9 EARLY-STOPPED CHAIN DONE ==="
