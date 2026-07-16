#!/usr/bin/env bash
# P2 训练完后自动跑：vllm 翻译评估 ×4模型 + vllm mono 补漏(hellaswag/ceval/gsm8k) ×4模型。
# 用 .venv-eval（Python 3.11 + vllm）。
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RESULTS_MONO=$SUMMER/eval_results/full
RESULTS_TRANS=$SUMMER/eval_results/translate_wmt22
mkdir -p $RESULTS_MONO $RESULTS_TRANS

TASKS="hellaswag:10,ceval-valid:5,gsm8k:5"

declare -A MODELS=(
    [base_1.7b]=/home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
    [init_1.7b]=/home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
    [v17_p1]=$SUMMER/output/phase1_ckpt_v17
    [v17_p2]=$SUMMER/output/phase2_ckpt_v17
)

# 等 P2 训练结束
while pgrep -f train/finetune_muon.py >/dev/null 2>&1; do sleep 60; done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
echo "[$(date +%H:%M:%S)] === P2 训练已结束，启动 vllm 翻译 + mono 补漏 ==="

# ---- vllm 翻译评估（base/init/P1/P2 × WMT22 zh-en+en-zh，5-shot，200 samples）----
for tag in base_1.7b init_1.7b v17_p1 v17_p2; do
    path=${MODELS[$tag]}
    [ ! -d "$path" ] && { echo "skip $tag (no path)"; continue; }
    echo "[$(date +%H:%M:%S)] === TRANSLATE $tag (vllm) ==="
    cd $SUMMER && $PY $SUMMER/evals/eval_pretrain_translate_vllm.py \
        --model_path $path \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 \
        --output_path $RESULTS_TRANS/${tag}_vllm.json \
        > $RESULTS_TRANS/${tag}_vllm.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] TRANSLATE $tag DONE" \
        || echo "[$(date +%H:%M:%S)] TRANSLATE $tag FAILED"
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 5
done

# ---- vllm mono 补漏（hellaswag/ceval/gsm8k） ----
for tag in base_1.7b init_1.7b v17_p1 v17_p2; do
    path=${MODELS[$tag]}
    [ ! -d "$path" ] && { echo "skip $tag"; continue; }
    out=$RESULTS_MONO/${tag}_vllm
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] === MONO $tag (vllm) ==="
    cd $SUMMER && $PY $SUMMER/evals/eval_with_piece_vllm.py \
        --model_path $path \
        --tasks "$TASKS" \
        --output_dir $out \
        > $out/run.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] MONO $tag DONE" \
        || echo "[$(date +%H:%M:%S)] MONO $tag FAILED"
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 5
done

echo "[$(date +%H:%M:%S)] === Post-P2 评估全部完成 ==="
