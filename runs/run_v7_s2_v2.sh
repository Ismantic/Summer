#!/usr/bin/env bash
# v7_s2_v2: Re-do Phase 2 with bumped Muon LR.
# v7_s2 used muon_lr=1e-5 -> noop. Try muon_lr=2e-5, halfway between
# the failed 5e-5 (v5_s2) and the safe-but-quiet 1e-5.
# Single-variable change vs run_v7.sh Phase 2 — everything else identical.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

TRAIN_PT=$SUMMER/output/phase1_train_512_v7.pt
VALID_PT=$SUMMER/output/valid_512.pt
V7=$SUMMER/output/phase1_ckpt_v7
V7_S2_V2=$SUMMER/output/phase2_ckpt_v7_s2_v2

echo "=== [$(date +%H:%M:%S)] PHASE 2 v2: muon_lr=2e-5 (was 1e-5) ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V7 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V7_S2_V2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --warmup_steps 100 \
    --muon_lr 2e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] eval phase2_v7_s2_v2 ==="
TAG=phase2_v7_s2_v2
QWEN_NEW=$V7_S2_V2 OUT=$RESULTS/smoke_$TAG bash $SUMMER/runs/smoke_test.sh \
    > $RESULTS/smoke_${TAG}.log 2>&1 || true
CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SUMMER/evals/eval_pretrain_translate.py \
    --model_path $V7_S2_V2 --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 200 --batch_size 4 \
    --output_path $RESULTS/translate_wmt22/$TAG.json \
    > $RESULTS/translate_wmt22/$TAG.log 2>&1 || true
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/evals/eval_ppl.py \
    --model_path $V7_S2_V2 --valid_pt $VALID_PT --batch_size 16 \
    --output_path $RESULTS/ppl/$TAG.json \
    > $RESULTS/ppl/$TAG.log 2>&1 || true

echo "=== [$(date +%H:%M:%S)] v7_s2_v2 ALL DONE ==="
