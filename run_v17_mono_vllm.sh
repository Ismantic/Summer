#!/usr/bin/env bash
# v17 mono 补漏(vllm 后端,新 venv .venv-eval Python 3.11)：
# base / init / v17_p1 × hellaswag / mmlu / ceval-valid / gsm8k（之前精简版漏的 4 个 task）
#
# eval_with_piece_vllm.py 自动检测 piece vs HF tokenizer,所以 3 个模型同一脚本走。
set -u
export PYTHONUNBUFFERED=1
# 覆盖 eval_with_piece_vllm.py 内部的 setdefault("HF_HUB_OFFLINE", "1")
# —— 让它联网下没 cache 的数据集（hellaswag/mmlu/ceval/gsm8k 还没下过）
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RESULTS=$SUMMER/eval_results/full

# 等翻译评估结束（GPU 让出）
while pgrep -f eval_pretrain_translate.py >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] === GPU 空闲，启动 vllm mono 补漏 ==="

TASKS="hellaswag:10,mmlu:5,ceval-valid:5,gsm8k:5"

run_eval() {
    local tag=$1 path=$2
    [ ! -d "$path" ] && { echo "[$(date +%H:%M:%S)] skip $tag (no path)"; return; }
    local out=$RESULTS/${tag}_vllm
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] === EVAL $tag (vllm) ==="
    cd $SUMMER && $PY $SUMMER/eval_with_piece_vllm.py \
        --model_path $path \
        --tasks "$TASKS" \
        --output_dir $out \
        > $out/run.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] $tag DONE" \
        || echo "[$(date +%H:%M:%S)] $tag FAILED (见 $out/run.log)"
}

run_eval base_1.7b /home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
run_eval init_1.7b /home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
run_eval v17_p1    $SUMMER/output/phase1_ckpt_v17

echo "[$(date +%H:%M:%S)] === vllm mono 补漏全部完成 ==="
