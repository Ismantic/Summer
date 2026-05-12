#!/usr/bin/env bash
# v9 Phase 1 + 2 (assumes pretokenize already produced phase1_train_512_v9.pt).
# Adds inline_eval_progressive.sh as --inline_eval_cmd so we see metrics at
# every save checkpoint (step 3000/6000/9000 light, 11500 full).
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
TRAIN_PT=$SUMMER/output/phase1_train_512_v9.pt
V9=$SUMMER/output/phase1_ckpt_v9
V9_S2=$SUMMER/output/phase2_ckpt_v9_s2_aurora

if [ ! -f "$TRAIN_PT" ]; then
  echo "FATAL: $TRAIN_PT missing — pretokenize must finish first."
  exit 1
fi

# Phase 1: 11500 steps × eff_bs 256 × seq 512 ≈ 1.5B tokens
# inline_eval at every save_step (3000/6000/9000 light, 11500 full)
INLINE_EVAL="bash $SUMMER/inline_eval_progressive.sh {step} $V9 v9_p1 11500"

echo "=== [$(date +%H:%M:%S)] PHASE 1 v9: 11500 steps, eff bs=256, lr=1e-4, inline eval @3k/6k/9k/11500 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V9 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 11500 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 3000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL"

echo "=== [$(date +%H:%M:%S)] eval phase1_v9 (full suite — final safety net) ==="
bash $SUMMER/eval_full.sh phase1_ckpt_v9

# Phase 2 — Aurora 500 steps. Single ckpt at end, inline eval = full.
INLINE_EVAL_P2="bash $SUMMER/inline_eval_progressive.sh {step} $V9_S2 v9_p2_aurora 500"

echo "=== [$(date +%H:%M:%S)] PHASE 2 v9_s2_aurora: 500 steps, Aurora muon=5e-5 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $V9 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V9_S2 \
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
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] eval phase2_v9_s2_aurora (full suite — final safety net) ==="
bash $SUMMER/eval_full.sh phase2_ckpt_v9_s2_aurora

echo "=== [$(date +%H:%M:%S)] v9 ALL DONE ==="
