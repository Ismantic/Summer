#!/usr/bin/env bash
# v15 = v8 mix (proven winner) + cosine schedule + min_lr 0.01.
#
# Single-variable test:
# - vs v8 P1 (same mix, same 7500 steps, but LINEAR-to-0): isolates scheduler effect
# - vs v14 (same cosine + min_lr 0.01, but v12 mix): isolates mix effect
#
# Hypothesis: linear was empirically better in v8/v12 vs v13/v14, but those were
# also confounded with mix changes. v15 finally tests cosine on the proven v8 mix.
# If v15 < v8 → linear truly better for our setup.
# If v15 ≈ v8 → mix matters more than scheduler.
# If v15 > v8 → cosine was good all along, v13/v14 failed for mix reasons.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
V8_PT=$SUMMER/output/phase1_train_512_v8.pt    # reuse v8 pretokenize
V15_P1=$SUMMER/output/phase1_ckpt_v15

# Phase 1: frozen transformer, v8 mix, 7500 steps cosine + min_lr 0.01
echo "=== [$(date +%H:%M:%S)] PHASE 1 v15: 7500 steps frozen, v8 mix, cosine + min_lr 0.01 ==="
INLINE_EVAL="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V15_P1 v15_p1 7500"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29508 train/finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $V8_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V15_P1 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 7500 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL"

echo "=== [$(date +%H:%M:%S)] v15 P1 DONE ==="
