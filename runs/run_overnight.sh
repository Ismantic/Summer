#!/usr/bin/env bash
# Overnight schedule: extend Stage 1 (v6), then Stage 2 on v5, then Stage 2 on v6.
# Each followed by full eval suite.

set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=/home/tfbao/Shiyu/Summer/eval_results

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
TRAIN_PT=$SUMMER/output/phase1_train_512_v5.pt   # reuse v5 pretokenized data
VALID_PT=$SUMMER/output/valid_512.pt

V5=$SUMMER/output/phase1_ckpt_v5
V6=$SUMMER/output/phase1_ckpt_v6           # longer Stage 1
V5_S2=$SUMMER/output/phase2_ckpt_v5_s2     # Stage 2 from v5
V6_S2=$SUMMER/output/phase2_ckpt_v6_s2     # Stage 2 from v6

run_eval() {
    local CKPT=$1
    local TAG=$2
    echo "=== [$(date +%H:%M:%S)] eval $TAG ==="
    QWEN_NEW=$CKPT OUT=$RESULTS/smoke_$TAG bash $SUMMER/runs/smoke_test.sh \
        > $RESULTS/smoke_${TAG}.log 2>&1 || true
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/evals/eval_pretrain_translate.py \
        --model_path $CKPT --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 --batch_size 4 \
        --output_path $RESULTS/translate_wmt22/$TAG.json \
        > $RESULTS/translate_wmt22/$TAG.log 2>&1 || true
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/evals/eval_ppl.py \
        --model_path $CKPT --valid_pt $VALID_PT --batch_size 16 \
        --output_path $RESULTS/ppl/$TAG.json \
        > $RESULTS/ppl/$TAG.log 2>&1 || true
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/evals/eval_with_piece.py \
        --model_path $CKPT --task lambada_openai --num_fewshot 0 --limit 500 \
        --output_path $RESULTS/lambada/$TAG/result.json \
        > $RESULTS/lambada/$TAG.log 2>&1 || true
}

# ============================================================================
# Experiment 1: v6 — extended Stage 1 (6000 steps on v5 data)
# ============================================================================
echo "=== [$(date +%H:%M:%S)] EXPERIMENT 1: v6 (Stage 1 extended, 6000 steps) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V6 \
    --max_seq_length 512 \
    --batch_size 32 \
    --max_steps 6000 \
    --warmup_steps 200 \
    --adam_lr 1e-4 \
    --save_steps 1000 \
    --logging_steps 50

run_eval $V6 phase1_v6

# ============================================================================
# Experiment 2: v5_s2 — Stage 2 on v5 (unfreeze, lr=5e-5, 3000 steps)
# ============================================================================
echo "=== [$(date +%H:%M:%S)] EXPERIMENT 2: v5_s2 (Stage 2 on v5) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V5 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V5_S2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --max_steps 3000 \
    --warmup_steps 100 \
    --adam_lr 5e-5 \
    --muon_lr 5e-5 \
    --save_steps 1000 \
    --logging_steps 50

run_eval $V5_S2 phase2_v5_s2

# ============================================================================
# Experiment 3: v6_s2 — Stage 2 on v6 (after extended Stage 1)
# ============================================================================
echo "=== [$(date +%H:%M:%S)] EXPERIMENT 3: v6_s2 (Stage 2 on v6) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V6 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V6_S2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --max_steps 3000 \
    --warmup_steps 100 \
    --adam_lr 5e-5 \
    --muon_lr 5e-5 \
    --save_steps 1000 \
    --logging_steps 50

run_eval $V6_S2 phase2_v6_s2

echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
