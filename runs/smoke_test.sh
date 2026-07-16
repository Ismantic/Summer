#!/usr/bin/env bash
# Smoke test: run all 9 ReTok Table 2 benchmarks with --limit 20 on
# Qwen3-0.6B-Base (BBPE) and the swapped piece-tokenizer model in parallel.
set -u

PYTHON=/home/tfbao/.venv/bin/python
QWEN_BASE=${QWEN_BASE:-/home/tfbao/new/Qwen3-0.6B-Base}
QWEN_NEW=${QWEN_NEW:-/home/tfbao/new/Qwen3-0.6B-Base-new-tok}
OUT=${OUT:-/home/tfbao/Shiyu/Summer/eval_results/smoke}
LIMIT=${LIMIT:-200}

mkdir -p "$OUT/base" "$OUT/new-tok"

TASKS=(
    "lambada_openai:0"
    "piqa:5"
    "arc_challenge:25"
    "hellaswag:10"
    "mmlu_abstract_algebra:5"
    "gsm8k:5"
)
# Dropped from ReTok Table 2:
#   cmmlu  — datasets 4.x removed .py-script-style loading
#   humaneval — needs HF_ALLOW_CODE_EVAL=1 + sandbox; noisy at limit=20
#   agieval, bbh — not informative enough at smoke scale, slow
# Kept 5: PIQA / ARC-C / HellaSwag / MMLU / GSM8K — covers reasoning, commonsense,
# knowledge, and math (generation), enough signal for smoke-level diffs.

run_base() {
    local task=$1 shots=$2 out=$3
    local extra=""
    [ "$task" = "humaneval" ] && extra="--confirm_run_unsafe_code"
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
        --model hf --batch_size auto \
        --model_args "pretrained=$QWEN_BASE,trust_remote_code=True,dtype=bfloat16" \
        --tasks "$task" --num_fewshot "$shots" --limit "$LIMIT" \
        --output_path "$out" $extra
}

run_new() {
    local task=$1 shots=$2 out=$3
    local extra=""
    [ "$task" = "humaneval" ] && extra="--confirm_run_unsafe_code"
    CUDA_VISIBLE_DEVICES=1 $PYTHON /home/tfbao/Shiyu/Summer/evals/eval_with_piece.py \
        --model_path "$QWEN_NEW" \
        --task "$task" --num_fewshot "$shots" --limit "$LIMIT" \
        --output_path "$out/result.json" $extra
}

for spec in "${TASKS[@]}"; do
    task=${spec%%:*}
    shots=${spec##*:}
    echo "=== [$(date +%H:%M:%S)] $task (${shots}-shot, limit=$LIMIT) ==="

    base_log="$OUT/base/${task}.log"
    new_log="$OUT/new-tok/${task}.log"

    run_base "$task" "$shots" "$OUT/base/$task" >"$base_log" 2>&1 &
    base_pid=$!
    run_new  "$task" "$shots" "$OUT/new-tok/$task" >"$new_log" 2>&1 &
    new_pid=$!

    wait $base_pid; base_rc=$?
    wait $new_pid;  new_rc=$?
    echo "    base rc=$base_rc  new rc=$new_rc  (logs: $base_log, $new_log)"
done

echo "=== [$(date +%H:%M:%S)] smoke test done ==="
