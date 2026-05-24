#!/usr/bin/env bash
# v17 翻译评估 vllm 版（新 venv .venv-eval Python 3.11）。
# base / init / v17_p1 × WMT22 zh-en + en-zh，5-shot，200 samples。
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RESULTS=$SUMMER/eval_results/translate_wmt22

# 等当前 vllm mono 补漏跑完（GPU 让出）
while pgrep -f "eval_with_piece_vllm.py" >/dev/null 2>&1; do
    sleep 30
done
# 再确保没有残留 vllm EngineCore
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
echo "[$(date +%H:%M:%S)] === GPU 空闲，启动 vllm 翻译评估 ==="

run_trans() {
    local tag=$1 path=$2
    [ ! -d "$path" ] && { echo "[$(date +%H:%M:%S)] skip $tag (no path)"; return; }
    echo "[$(date +%H:%M:%S)] === TRANSLATE $tag (vllm) ==="
    cd $SUMMER && $PY $SUMMER/eval_pretrain_translate_vllm.py \
        --model_path $path \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 \
        --compute_comet --save_all_samples \
        --comet_model_path /home/tfbao/a6000/Summer-data/comet-wmt22-da \
        --output_path $RESULTS/${tag}_vllm.json \
        > $RESULTS/${tag}_vllm.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] $tag DONE" \
        || echo "[$(date +%H:%M:%S)] $tag FAILED (见 ${tag}_vllm.log)"
}

run_trans base_1.7b /home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
run_trans init_1.7b /home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
run_trans v17_p1    $SUMMER/output/phase1_ckpt_v17

echo "[$(date +%H:%M:%S)] === vllm 翻译评估全部完成 ==="
