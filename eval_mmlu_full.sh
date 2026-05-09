#!/usr/bin/env bash
# Full MMLU eval (57 subjects, ~14k questions) for tight stderr (~±0.005)
# vs the abstract_algebra-only smoke (100 q, stderr ±0.046).
#
# Run base on GPU0 and a target model on GPU1 in parallel; a second target
# model runs after the first round finishes.
set -u
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results/mmlu_full

QWEN_BASE=/home/tfbao/new/Qwen3-0.6B-Base
V7=$SUMMER/output/phase1_ckpt_v7
V7_S2_V2=$SUMMER/output/phase2_ckpt_v7_s2_v2
V7_S2=$SUMMER/output/phase2_ckpt_v7_s2

mkdir -p $RESULTS

run_base() {
    local out=$1
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
        --model hf --batch_size auto \
        --model_args "pretrained=$QWEN_BASE,trust_remote_code=True,dtype=bfloat16" \
        --tasks mmlu --num_fewshot 5 \
        --output_path "$out"
}

run_new() {
    local ckpt=$1 out=$2 device=$3
    CUDA_VISIBLE_DEVICES=$device $PYTHON $SUMMER/eval_with_piece.py \
        --model_path "$ckpt" \
        --task mmlu --num_fewshot 5 \
        --output_path "$out/result.json"
}

echo "=== [$(date +%H:%M:%S)] round 1: base (GPU0) + v7 (GPU1) ==="
run_base $RESULTS/base > $RESULTS/base.log 2>&1 &
PID_BASE=$!
run_new $V7 $RESULTS/v7 1 > $RESULTS/v7.log 2>&1 &
PID_V7=$!
wait $PID_BASE $PID_V7
echo "=== [$(date +%H:%M:%S)] round 1 done ==="

echo "=== [$(date +%H:%M:%S)] round 2: v7_s2_v2 (GPU0) + v7_s2 (GPU1) ==="
run_new $V7_S2_V2 $RESULTS/v7_s2_v2 0 > $RESULTS/v7_s2_v2.log 2>&1 &
PID_A=$!
run_new $V7_S2 $RESULTS/v7_s2 1 > $RESULTS/v7_s2.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
echo "=== [$(date +%H:%M:%S)] round 2 done — ALL DONE ==="
