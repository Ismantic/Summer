#!/usr/bin/env bash
# Smoke test new tasks (cmmlu / ceval-valid / gsm8k) on base — uses --limit 50
# for fast sanity check before full retro run.
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
OUT=$SUMMER/eval_results/smoke

mkdir -p $OUT

echo "=== Smoke test 3 new tasks on base, --limit 50 ==="
echo "Start: $(date +%H:%M:%S)"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece_vllm.py \
    --model_path /home/tfbao/new/Qwen3-0.6B-Base \
    --tasks "cmmlu:5,ceval-valid:5,gsm8k:5" \
    --output_dir $OUT \
    --max_model_len 4096 \
    --limit 50

echo "End: $(date +%H:%M:%S)"
