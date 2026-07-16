#!/usr/bin/env bash
# P2 全参数训练完后自动评估 v17_p2_full 一个模型（vllm）：
#  - WMT22 zh-en + en-zh BLEU
#  - hellaswag / ceval-valid / gsm8k mono
#  - lambada_openai / piqa / arc_challenge mono
set -u
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

SUMMER=/home/tfbao/Shiyu/Summer
PY=/home/tfbao/.venv-eval/bin/python
RESULTS_MONO=$SUMMER/eval_results/full
RESULTS_TRANS=$SUMMER/eval_results/translate_wmt22
MODEL=$SUMMER/output/phase2_ckpt_v17_full
TAG=v17_p2_full

ALL_TASKS="lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5"

# 等 P2 full 训练结束
while pgrep -f train/finetune_muon.py >/dev/null 2>&1; do sleep 60; done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
echo "[$(date +%H:%M:%S)] === P2 full 训练已结束，启动 v17_p2_full 评估 ==="

# ---- 翻译 ----
echo "[$(date +%H:%M:%S)] === TRANSLATE $TAG (vllm) ==="
cd $SUMMER && $PY $SUMMER/evals/eval_pretrain_translate_vllm.py \
    --model_path $MODEL \
    --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 200 \
    --output_path $RESULTS_TRANS/${TAG}_vllm.json \
    > $RESULTS_TRANS/${TAG}_vllm.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] TRANSLATE DONE" \
    || echo "[$(date +%H:%M:%S)] TRANSLATE FAILED"
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

# ---- mono 全套 6 task ----
out=$RESULTS_MONO/${TAG}_vllm
mkdir -p $out
echo "[$(date +%H:%M:%S)] === MONO $TAG (vllm, 6 tasks) ==="
cd $SUMMER && $PY $SUMMER/evals/eval_with_piece_vllm.py \
    --model_path $MODEL \
    --tasks "$ALL_TASKS" \
    --output_dir $out \
    > $out/run.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] MONO DONE" \
    || echo "[$(date +%H:%M:%S)] MONO FAILED"

echo "[$(date +%H:%M:%S)] === Post-P2-full 评估完成 ==="
