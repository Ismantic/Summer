#!/usr/bin/env bash
# Re-run BLEU at 1000 samples for v8 and v8_s2_aurora, matching the
# new eval_full.sh convention. Original runs were 200 samples and base
# was also 200; the new full-eval base shows 18.07/32.67 vs the old
# 15.40/33.00, so the prior "aurora -0.7% zh-en" claim was 200-sample
# noise. Need 1000-sample numbers for v8 and aurora to compare with
# muon5e5 / moonshot / aurora_long apples-to-apples.
#
# Designed to run on GPU1 only (GPU0 likely busy with moonshot eval).
# Both ckpts in sequence on GPU1: ~10 min total.
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer

run_bleu() {
    local tag=$1
    local ckpt=$SUMMER/output/$tag
    local out=$SUMMER/eval_results/full/$tag
    mkdir -p $out
    echo "[$(date +%H:%M:%S)] BLEU 1000-sample for $tag (GPU1)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path "$ckpt" \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 1000 --batch_size 4 \
        --output_path "$out/wmt22.json" > "$out/wmt22.log" 2>&1
}

run_bleu phase1_ckpt_v8
run_bleu phase2_ckpt_v8_s2_aurora
echo "[$(date +%H:%M:%S)] BLEU rerun done"
