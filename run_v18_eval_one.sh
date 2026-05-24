#!/usr/bin/env bash
# 评估单个 v18 模型:1000-sample BLEU+COMET + 6 mono tasks(vllm)
# 用法: bash run_v18_eval_one.sh <tag> <model_path>
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TAG=$1
MODEL=$2

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RM=$SUMMER/eval_results/full
RT=$SUMMER/eval_results/translate_wmt22
COMET=/home/tfbao/a6000/Summer-data/comet-wmt22-da
ALL_TASKS="${ALL_TASKS:-lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5}"

pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 3
echo "[$(date +%H:%M:%S)] === TRANS+COMET $TAG ==="
cd $SUMMER && $PY $SUMMER/eval_pretrain_translate_vllm.py \
    --model_path $MODEL --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 1000 \
    --compute_comet --save_all_samples --comet_model_path $COMET \
    --output_path $RT/${TAG}_vllm_1000_comet.json \
    > $RT/${TAG}_vllm_1000_comet.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] TRANS DONE" \
    || echo "[$(date +%H:%M:%S)] TRANS FAILED"

pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 3
out=$RM/${TAG}_vllm
mkdir -p $out
echo "[$(date +%H:%M:%S)] === MONO $TAG ==="
cd $SUMMER && $PY $SUMMER/eval_with_piece_vllm.py \
    --model_path $MODEL --tasks "$ALL_TASKS" --output_dir $out \
    > $out/run.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] MONO DONE" \
    || echo "[$(date +%H:%M:%S)] MONO FAILED"
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "[$(date +%H:%M:%S)] === $TAG EVAL ALL DONE ==="
