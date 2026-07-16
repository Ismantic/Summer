#!/usr/bin/env bash
# Full mono-LM eval (no --limit) on LAMBADA / PIQA / ARC-C / HellaSwag
# to calibrate the smoke values (200 samples have stderr ~±0.035, too
# loose for our +0.01-level keep decisions).
#
# Runs base on GPU0 and a new-tok ckpt on GPU1 in parallel; multiple
# new-tok ckpts run in sequence.
set -u
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results/mono_full

QWEN_BASE=/home/tfbao/new/Qwen3-0.6B-Base

# tasks: name:shots
TASKS="lambada_openai:0 piqa:5 arc_challenge:25 hellaswag:10"

mkdir -p $RESULTS

run_base() {
    local out=$RESULTS/base
    mkdir -p $out
    for spec in $TASKS; do
        local task=${spec%:*}
        local shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
            --model hf --batch_size auto \
            --model_args "pretrained=$QWEN_BASE,trust_remote_code=True,dtype=bfloat16" \
            --tasks "$task" --num_fewshot "$shots" \
            --output_path "$out/$task" \
            > "$out/$task.log" 2>&1
    done
}

run_new() {
    local ckpt=$1
    local out=$2
    local device=$3
    mkdir -p $out
    for spec in $TASKS; do
        local task=${spec%:*}
        local shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=$device $PYTHON $SUMMER/evals/eval_with_piece.py \
            --model_path "$ckpt" \
            --task "$task" --num_fewshot "$shots" \
            --output_path "$out/$task/result.json" \
            > "$out/$task.log" 2>&1 || true
    done
}

V8=$SUMMER/output/phase1_ckpt_v8
V8A=$SUMMER/output/phase2_ckpt_v8_s2_aurora

echo "=== [$(date +%H:%M:%S)] round 1: base (GPU0) + v8 (GPU1) ==="
run_base &
PID_B=$!
run_new $V8 $RESULTS/v8 1 &
PID_V=$!
wait $PID_B $PID_V
echo "=== [$(date +%H:%M:%S)] round 1 done ==="

echo "=== [$(date +%H:%M:%S)] round 2: v8_s2_aurora (GPU0) ==="
run_new $V8A $RESULTS/v8_s2_aurora 0
echo "=== [$(date +%H:%M:%S)] mono full eval ALL DONE ==="
