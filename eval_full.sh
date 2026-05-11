#!/usr/bin/env bash
# Full eval suite — apples-to-apples ground truth across all benchmarks.
# Replaces smoke (200 samples, ±3-5pp noise) for keep/discard decisions.
#
# Runs in parallel rounds: base on GPU0, target on GPU1. Multiple targets
# loop sequentially. Each round produces these artifacts per model:
#
#   eval_results/full/<tag>/
#       lambada_openai/result.json   - LAMBADA acc (0-shot)
#       piqa/result.json             - PIQA acc_norm (5-shot)
#       arc_challenge/result.json    - ARC-C acc_norm (25-shot)
#       hellaswag/result.json        - HellaSwag acc_norm (10-shot)
#       mmlu/result.json             - MMLU acc (5-shot, full 14k)
#       wmt22.json                   - BLEU zh-en + en-zh (1000 samples)
#       ppl.json                     - valid set perplexity
#
# Usage:
#   bash eval_full.sh <ckpt_tag> [<ckpt_tag2> ...]
# where each <ckpt_tag> is the name of a directory under output/.
# Example:
#   bash eval_full.sh phase1_ckpt_v8 phase2_ckpt_v8_s2_aurora
#
# Base (Qwen3-0.6B-Base) is always (re)evaluated as a sanity check
# unless eval_results/full/base/mmlu/__home*/results_*.json already
# exists (then skipped).
set -u
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
FULL=$SUMMER/eval_results/full
VALID_PT=$SUMMER/output/valid_512.pt
QWEN_BASE=/home/tfbao/new/Qwen3-0.6B-Base

mkdir -p $FULL

# mono-LM tasks: name:shots (acc_norm where standard, acc otherwise)
TASKS_MONO="lambada_openai:0 piqa:5 arc_challenge:25 hellaswag:10"
MMLU_SHOTS=5
BLEU_SAMPLES=1000

# -------- base eval (skip if MMLU result already present) --------
eval_base() {
    local out=$FULL/base
    if compgen -G "$out/mmlu/__home*/results_*.json" > /dev/null; then
        echo "  [base] already evaluated, skipping"
        return 0
    fi
    mkdir -p $out
    for spec in $TASKS_MONO; do
        local task=${spec%:*}
        local shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
            --model hf --batch_size auto \
            --model_args "pretrained=$QWEN_BASE,trust_remote_code=True,dtype=bfloat16" \
            --tasks "$task" --num_fewshot "$shots" \
            --output_path "$out/$task" > "$out/$task.log" 2>&1
    done
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m lm_eval \
        --model hf --batch_size auto \
        --model_args "pretrained=$QWEN_BASE,trust_remote_code=True,dtype=bfloat16" \
        --tasks mmlu --num_fewshot $MMLU_SHOTS \
        --output_path "$out/mmlu" > "$out/mmlu.log" 2>&1
    # base has its own tokenizer — translate eval uses HF tokenizer path
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path $QWEN_BASE --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples $BLEU_SAMPLES --batch_size 16 \
        --output_path "$out/wmt22.json" > "$out/wmt22.log" 2>&1 || true
}

# -------- new-tok ckpt eval --------
eval_new() {
    local tag=$1 device=$2
    local ckpt=$SUMMER/output/$tag
    local out=$FULL/$tag
    mkdir -p $out
    [ ! -d "$ckpt" ] && { echo "  [$tag] ckpt dir missing"; return 1; }
    for spec in $TASKS_MONO; do
        local task=${spec%:*}
        local shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=$device $PYTHON $SUMMER/eval_with_piece.py \
            --model_path "$ckpt" \
            --task "$task" --num_fewshot "$shots" \
            --output_path "$out/$task/result.json" > "$out/$task.log" 2>&1 || true
    done
    CUDA_VISIBLE_DEVICES=$device $PYTHON $SUMMER/eval_with_piece.py \
        --model_path "$ckpt" \
        --task mmlu --num_fewshot $MMLU_SHOTS \
        --output_path "$out/mmlu/result.json" > "$out/mmlu.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=$device $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path "$ckpt" --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples $BLEU_SAMPLES --batch_size 16 \
        --output_path "$out/wmt22.json" > "$out/wmt22.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=$device $PYTHON -u $SUMMER/eval_ppl.py \
        --model_path "$ckpt" --valid_pt $VALID_PT --batch_size 16 \
        --output_path "$out/ppl.json" > "$out/ppl.log" 2>&1 || true
}

if [ "$#" -eq 0 ]; then
    echo "usage: eval_full.sh <ckpt_tag> [<ckpt_tag2> ...]"
    exit 1
fi

# Decide whether base needs to run this time. If it does and we have at
# least one target, run base on GPU0 in parallel with the first target
# on GPU1 — saves ~30min the first time eval_full is ever called.
BASE_NEEDS_RUN=true
if compgen -G "$FULL/base/mmlu/__home*/results_*.json" > /dev/null; then
    BASE_NEEDS_RUN=false
fi

TARGETS=("$@")
i=0

if [ "$BASE_NEEDS_RUN" = "true" ] && [ ${#TARGETS[@]} -ge 1 ]; then
    a=${TARGETS[$i]}
    echo "=== [$(date +%H:%M:%S)] base (GPU0) + $a (GPU1) parallel ==="
    eval_base &
    PID_BASE=$!
    eval_new $a 1 &
    PID_A=$!
    wait $PID_BASE $PID_A
    i=$((i+1))
else
    echo "=== [$(date +%H:%M:%S)] base eval (round 0) ==="
    eval_base
fi

# Remaining targets: 2 at a time on GPU0+GPU1.
while [ $i -lt ${#TARGETS[@]} ]; do
    a=${TARGETS[$i]}
    b=${TARGETS[$((i+1))]:-}
    if [ -n "$b" ]; then
        echo "=== [$(date +%H:%M:%S)] eval $a (GPU0) + $b (GPU1) ==="
        eval_new $a 0 &
        PID_A=$!
        eval_new $b 1 &
        PID_B=$!
        wait $PID_A $PID_B
        i=$((i+2))
    else
        echo "=== [$(date +%H:%M:%S)] eval $a (GPU0) ==="
        eval_new $a 0
        i=$((i+1))
    fi
done

echo "=== [$(date +%H:%M:%S)] eval_full ALL DONE ==="
