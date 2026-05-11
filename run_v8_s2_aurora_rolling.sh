#!/usr/bin/env bash
# v8_s2_aurora_rolling: 3000-step Aurora training with inline save→eval→resume
# every 500 steps. At each save_steps boundary the training process moves
# model+optimizer to CPU, frees CUDA cache, and runs eval_full.sh on the
# just-saved ckpt; then everything moves back to GPU and training continues
# from in-memory state (no on-disk resume).
#
# Gives a 6-point Aurora scaling curve in the natural training flow.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/phase2_ckpt_v8_s2_aurora_rolling

# Inline eval: progressive — light at intermediate steps, full at final.
# Dispatcher takes (step, ckpt_root, tag_prefix, final_step).
INLINE_CMD="bash $SUMMER/inline_eval_progressive.sh {step} $OUT phase2_v8_s2_aurora_rolling 3000"

echo "=== [$(date +%H:%M:%S)] PHASE 2 v8_s2_aurora_rolling: 3000 steps, Aurora muon_lr=5e-5, eval every 500 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $V8 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 3000 \
    --warmup_steps 300 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_CMD"

echo "=== [$(date +%H:%M:%S)] v8_s2_aurora_rolling ALL DONE ==="
