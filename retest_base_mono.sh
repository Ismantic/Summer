#!/usr/bin/env bash
# Re-test base Qwen3-0.6B mono on new transformers (4.57.6) to quantify drift
# vs the 2026-05-11 baseline (which used pre-upgrade transformers).
#
# Same lm_eval CLI invocation as the original base eval, only difference is
# the installed transformers version. Results land in base_retest_2026-05-15/.
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
BASE=/home/tfbao/new/Qwen3-0.6B-Base
OUT_ROOT=$SUMMER/eval_results/full/base_retest

mkdir -p $OUT_ROOT

for spec in "lambada_openai:0" "piqa:5" "arc_challenge:25" "hellaswag:10" "mmlu:5"; do
    task=${spec%:*}; shots=${spec#*:}
    OUT=$OUT_ROOT/$task
    mkdir -p $OUT
    echo "=== [$(date +%H:%M:%S)] $task ($shots-shot) ==="
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
        --model hf \
        --model_args pretrained=$BASE,dtype=bfloat16,trust_remote_code=True \
        --tasks $task \
        --num_fewshot $shots \
        --batch_size auto \
        --output_path $OUT \
        > $OUT_ROOT/$task.log 2>&1 || echo "  FAILED — see $OUT_ROOT/$task.log"
    echo "  [$(date +%H:%M:%S)] $task done"
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
