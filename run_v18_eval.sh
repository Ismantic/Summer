#!/usr/bin/env bash
# v18 评估:v18_p1 + v18_p2 各跑 1000-sample BLEU+COMET + 6 mono(vllm)
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RM=$SUMMER/eval_results/full
RT=$SUMMER/eval_results/translate_wmt22
COMET=/home/tfbao/a6000/Summer-data/comet-wmt22-da

ALL_TASKS="lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5"

declare -A M=(
  [v18_p1]=$SUMMER/output/phase1_ckpt_v18
  [v18_p2]=$SUMMER/output/phase2_ckpt_v18
)

for tag in v18_p1 v18_p2; do
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 3
    echo "[$(date +%H:%M:%S)] === TRANS+COMET $tag ==="
    cd $SUMMER && $PY $SUMMER/eval_pretrain_translate_vllm.py \
        --model_path ${M[$tag]} --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 1000 \
        --compute_comet --save_all_samples --comet_model_path $COMET \
        --output_path $RT/${tag}_vllm_1000_comet.json \
        > $RT/${tag}_vllm_1000_comet.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] TRANS DONE" \
        || echo "[$(date +%H:%M:%S)] TRANS FAILED"
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 3
    out=$RM/${tag}_vllm
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] === MONO $tag ==="
    cd $SUMMER && $PY $SUMMER/eval_with_piece_vllm.py \
        --model_path ${M[$tag]} --tasks "$ALL_TASKS" --output_dir $out \
        > $out/run.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] MONO DONE" \
        || echo "[$(date +%H:%M:%S)] MONO FAILED"
done
echo "[$(date +%H:%M:%S)] === v18 eval all DONE ==="
