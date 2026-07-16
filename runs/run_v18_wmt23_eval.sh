#!/usr/bin/env bash
# WMT23 全集 BLEU+COMET 评估:6 模型对比
# 用 wmt22 做 5-shot exemplar(testset/exemplar 不重)
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
R=$SUMMER/eval_results/translate_wmt23
COMET=/home/tfbao/a6000/Summer-data/comet-wmt22-da
mkdir -p $R

declare -A M=(
  [base_1.7b]=/home/tfbao/Shiyu/Interpreter/Translator/Qwen3-1.7B-Base
  [v17_p1]=/home/tfbao/Shiyu/Summer/output/phase1_ckpt_v17
  [v17_p2]=$SUMMER/output/phase2_ckpt_v17
  [v18_p1]=$SUMMER/output/phase1_ckpt_v18
  [v18_p2]=$SUMMER/output/phase2_ckpt_v18
  [v18_p2_tie]=$SUMMER/output/phase2_ckpt_v18_tie
)

for tag in base_1.7b v17_p1 v17_p2 v18_p1 v18_p2 v18_p2_tie; do
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 3
    echo "[$(date +%H:%M:%S)] === WMT23 TRANS+COMET $tag ==="
    cd $SUMMER && $PY $SUMMER/evals/eval_pretrain_translate_vllm.py \
        --model_path ${M[$tag]} --testset wmt23 --exemplar_set wmt22 --direction both \
        --num_fewshot 5 --max_samples 0 \
        --compute_comet --save_all_samples --comet_model_path $COMET \
        --output_path $R/${tag}_wmt23_full.json \
        > $R/${tag}_wmt23_full.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] $tag DONE" \
        || echo "[$(date +%H:%M:%S)] $tag FAILED"
done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "[$(date +%H:%M:%S)] === ALL WMT23 DONE ==="
