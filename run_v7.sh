#!/usr/bin/env bash
# v7 = ReTok-style 2-stage training on Qwen3/DeepSeek-aligned diverse data.
#
# Phase 1 (frozen): align embed/lm_head to v7 distribution
# Phase 2 (unfreeze): low-LR Muon for transformer (protect Qwen3 priors)
#                     standard adam_lr for embed (still needs to learn)
#
# Both stages train on the SAME phase1_train_512_v7.pt (per ReTok methodology).
# v7 effective batch ~256 chunks (Phase 1) / ~128 chunks (Phase 2) via grad_accum.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
TRAIN_PT=$SUMMER/output/phase1_train_512_v7.pt
VALID_PT=$SUMMER/output/valid_512.pt

V7=$SUMMER/output/phase1_ckpt_v7
V7_S2=$SUMMER/output/phase2_ckpt_v7_s2

run_eval() {
    local CKPT=$1
    local TAG=$2
    echo "=== [$(date +%H:%M:%S)] eval $TAG ==="
    QWEN_NEW=$CKPT OUT=$RESULTS/smoke_$TAG bash $SUMMER/smoke_test.sh \
        > $RESULTS/smoke_${TAG}.log 2>&1 || true
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path $CKPT --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 --batch_size 4 \
        --output_path $RESULTS/translate_wmt22/$TAG.json \
        > $RESULTS/translate_wmt22/$TAG.log 2>&1 || true
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_ppl.py \
        --model_path $CKPT --valid_pt $VALID_PT --batch_size 16 \
        --output_path $RESULTS/ppl/$TAG.json \
        > $RESULTS/ppl/$TAG.log 2>&1 || true
}

# ============================================================================
# v7 Phase 1: freeze transformer, train embed/lm_head on v7 diverse mix
# Effective batch: 32 × 2 GPU × 4 accum = 256 chunks × 512 = 131K tok/step
# ~3000 steps × 131K = ~400M tokens
# ============================================================================
echo "=== [$(date +%H:%M:%S)] PHASE 1: v7 (3000 steps, eff bs=256, lr=1e-4) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V7 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 3000 \
    --warmup_steps 500 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 50

run_eval $V7 phase1_v7

# ============================================================================
# v7 Phase 2: unfreeze, low Muon LR (protect Qwen3 priors), normal Adam LR
# Effective batch: 16 × 2 GPU × 4 accum = 128 chunks × 512 = 65K tok/step
# ~2000 steps × 65K = ~130M tokens
# ============================================================================
echo "=== [$(date +%H:%M:%S)] PHASE 2: v7_s2 (2000 steps, eff bs=128, muon=1e-5 adam=5e-5) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $V7 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V7_S2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 500 \
    --muon_lr 1e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --logging_steps 50

run_eval $V7_S2 phase2_v7_s2

echo "=== [$(date +%H:%M:%S)] v7 ALL DONE ==="
