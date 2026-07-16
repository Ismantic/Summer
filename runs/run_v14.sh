#!/usr/bin/env bash
# v14 = v12 data mix + corrected LR schedule.
#
# Lessons from v12/v13:
# - v12 (linear-to-0, 5000 steps): BLEU 15.45 (slight regression vs v8)
# - v13 (cosine + min_lr 0.1, 8000 steps): BLEU 7.79 (CRASH — generation degenerated)
# - Root cause: min_lr 0.1 too high + 8000 steps drifted embedding too far
#
# v14 fix (per "Reuse, Don't Retrain" Nvidia 2024 recommendations):
# - cosine schedule (matches Qwen3 base + standard practice)
# - min_lr_ratio = 0.01 (peak/100, NOT peak/10)
# - 5000 steps P1 (matches v12 working length)
#
# Also: inline_eval now runs BLEU+COMET at EVERY save_step (early warning).
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
MAIN_PT=$SUMMER/output/v12_main_512.pt        # reuse v12 mix
ANNEAL_PT=$SUMMER/output/v12_anneal_512.pt    # reuse v12 mix
V14_P1=$SUMMER/output/phase1_ckpt_v14
V14_P2=$SUMMER/output/phase2_ckpt_v14_aurora_anneal

# Phase 1: frozen transformer, broad mix, 5000 steps cosine + min_lr 0.01
echo "=== [$(date +%H:%M:%S)] PHASE 1 v14: 5000 steps frozen, cosine + min_lr_ratio 0.01 ==="
INLINE_EVAL_P1="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V14_P1 v14_p1 5000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29507 train/finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $MAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V14_P1 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P1"

# Phase 2: unfreeze + Aurora on annealed mix, 2000 steps cosine + min_lr 0.01
echo "=== [$(date +%H:%M:%S)] PHASE 2 v14: 2000 steps unfreeze Aurora anneal, cosine min_lr 0.01 ==="
INLINE_EVAL_P2="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V14_P2 v14_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29507 train/finetune_muon.py \
    --model_path $V14_P1/checkpoint-5000 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V14_P2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] v14 DONE ==="
