#!/usr/bin/env bash
# v17 P1 完成后的评估 pipeline（transformers 后端版）。
#
# vllm 装不上：Python 3.14 与 numba/llvmlite 不兼容,vllm 官方支持 3.9-3.12。
# 改用 transformers backend：base 用 lm_eval CLI(hf),init/P1 用 eval_with_piece.py。
# 走流程精简到 3 个英文任务：lambada / piqa / arc_challenge。
set -u
export PYTHONUNBUFFERED=1
# 不设 OFFLINE：lm_eval 需要联网下 lambada/piqa 等数据集（走代理）。
# 第一次跑会下到 ~/.cache/huggingface/datasets/，之后自动 cache。

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv/bin/python
RESULTS=$SUMMER/eval_results/full

TASKS=("lambada_openai:0" "piqa:5" "arc_challenge:25")

# 等 Phase 1 训练结束
while pgrep -f train/finetune_muon.py >/dev/null 2>&1; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] === Phase 1 已结束，启动评估 ==="

run_eval_hf() {
    local tag=$1 path=$2
    [ ! -d "$path" ] && { echo "[$(date +%H:%M:%S)] skip $tag (no path: $path)"; return; }
    local out=$RESULTS/${tag}_tf
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] === EVAL $tag (hf backend) ==="
    for spec in "${TASKS[@]}"; do
        local task=${spec%%:*} shots=${spec##*:}
        echo "[$(date +%H:%M:%S)]   $tag :: $task ($shots-shot)"
        cd $SUMMER && $PY -m lm_eval --model hf \
            --model_args pretrained=$path,dtype=bfloat16,trust_remote_code=True \
            --tasks $task --num_fewshot $shots \
            --output_path $out/$task --batch_size auto \
            >> $out/run.log 2>&1 \
            && echo "[$(date +%H:%M:%S)]   $tag :: $task DONE" \
            || echo "[$(date +%H:%M:%S)]   $tag :: $task FAILED"
    done
}

run_eval_piece() {
    local tag=$1 path=$2
    [ ! -d "$path" ] && { echo "[$(date +%H:%M:%S)] skip $tag (no path: $path)"; return; }
    local out=$RESULTS/${tag}_tf
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] === EVAL $tag (piece + hf backend) ==="
    for spec in "${TASKS[@]}"; do
        local task=${spec%%:*} shots=${spec##*:}
        echo "[$(date +%H:%M:%S)]   $tag :: $task ($shots-shot)"
        cd $SUMMER && $PY $SUMMER/evals/eval_with_piece.py \
            --model_path $path --task $task --num_fewshot $shots \
            --output_path $out/$task \
            >> $out/run.log 2>&1 \
            && echo "[$(date +%H:%M:%S)]   $tag :: $task DONE" \
            || echo "[$(date +%H:%M:%S)]   $tag :: $task FAILED"
    done
}

run_eval_hf    base_1.7b /home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
run_eval_piece init_1.7b /home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
run_eval_piece v17_p1    $SUMMER/output/phase1_ckpt_v17

echo "[$(date +%H:%M:%S)] === eval pipeline 完成 ==="
