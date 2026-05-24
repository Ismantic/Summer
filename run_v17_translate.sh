#!/usr/bin/env bash
# v17 翻译评估：base / init / v17_p1 三个模型 × WMT22 zh-en + en-zh（5-shot）
# 用 transformers backend（vllm 装不上）。先跑 BLEU 主体，COMET 后补。
set -u
export PYTHONUNBUFFERED=1
SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv/bin/python
RESULTS=$SUMMER/eval_results/translate_wmt22
mkdir -p $RESULTS

run_trans() {
    local tag=$1 path=$2
    [ ! -d "$path" ] && { echo "[$(date +%H:%M:%S)] skip $tag (no path)"; return; }
    echo "[$(date +%H:%M:%S)] === TRANSLATE $tag  ($path) ==="
    cd $SUMMER && $PY $SUMMER/eval_pretrain_translate.py \
        --model_path $path \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 --batch_size 4 \
        --output_path $RESULTS/$tag.json \
        > $RESULTS/$tag.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] $tag DONE" \
        || echo "[$(date +%H:%M:%S)] $tag FAILED (见 $RESULTS/$tag.log)"
}

run_trans base_1.7b /home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
run_trans init_1.7b /home/tfbao/new/Qwen3-1.7B-Base-new-tok-v2
run_trans v17_p1    $SUMMER/output/phase1_ckpt_v17

echo "[$(date +%H:%M:%S)] === 翻译评估全部完成 ==="
