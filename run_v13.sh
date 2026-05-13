#!/usr/bin/env bash
# v13 = v12 mix + cosine schedule + min_lr 0.1 + longer P1.
#
# Hypothesis: v12 P1 underperformed slightly (lambada -1.4%, BLEU -4.6%) likely
# due to (a) linear decay-to-0 wasting last steps, and (b) only 0.66B training
# tokens vs v8's 1B. v13 fixes both:
#   - cosine schedule (Llama/Qwen standard) with min_lr_ratio=0.1 (peak/10 floor)
#   - max_steps 5000 -> 8000 (1.05B training tokens, matches v8)
#
# Reuses v12_main_512.pt and v12_anneal_512.pt (same mix, no re-pretokenize).
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
MAIN_PT=$SUMMER/output/v12_main_512.pt        # reuse
ANNEAL_PT=$SUMMER/output/v12_anneal_512.pt    # reuse
V13_P1=$SUMMER/output/phase1_ckpt_v13
V13_P2=$SUMMER/output/phase2_ckpt_v13_aurora_anneal

# Phase 1: frozen transformer, broad mix, 8000 steps (1.05B tokens), cosine schedule
echo "=== [$(date +%H:%M:%S)] PHASE 1 v13: 8000 steps frozen, cosine + min_lr_ratio 0.1 ==="
INLINE_EVAL_P1="bash $SUMMER/inline_eval_progressive.sh {step} $V13_P1 v13_p1 8000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29506 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $MAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V13_P1 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 8000 \
    --warmup_steps 800 \
    --adam_lr 1e-4 \
    --min_lr_ratio 0.1 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --save_steps 2000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P1"

# Phase 2: unfreeze + Aurora on annealed mix, 2000 steps, cosine + min_lr 0.1
echo "=== [$(date +%H:%M:%S)] PHASE 2 v13: 2000 steps unfreeze Aurora anneal, cosine min_lr 0.1 ==="
INLINE_EVAL_P2="bash $SUMMER/inline_eval_progressive.sh {step} $V13_P2 v13_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29506 finetune_muon.py \
    --model_path $V13_P1/checkpoint-8000 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V13_P2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.1 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 1000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] v13 DONE ==="
