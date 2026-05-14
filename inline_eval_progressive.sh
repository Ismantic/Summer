#!/usr/bin/env bash
# Dispatched from finetune_muon's --inline_eval_cmd. Receives:
#   $1 = step number
#   $2 = output ckpt root (output_dir from training)
#   $3 = run tag prefix
#
# Strategy: intermediate steps run only the cheap mono tasks (~10 min) so
# we get a fast loss/accuracy heartbeat without bloating wall time. The
# final-step eval runs the full suite (mono+mmlu+bleu+ppl, ~60 min).
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

STEP=$1
CKPT_ROOT=$2
PREFIX=$3
FINAL_STEP=${4:-3000}

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer

CKPT=$CKPT_ROOT/checkpoint-$STEP
TAG=${PREFIX}_step${STEP}
TAG_DIR=$SUMMER/output/$TAG
RESULTS_DIR=$SUMMER/eval_results/full/$TAG

ln -sfn $CKPT $TAG_DIR
mkdir -p $RESULTS_DIR

if [ "$STEP" = "$FINAL_STEP" ]; then
    echo "[inline_eval] FULL eval for final step $STEP — BLEU + COMET FIRST"
    # BLEU+COMET first: saves all translations and runs COMET in same process.
    # COMET model loaded from local /mnt/data/Summer-data/comet-wmt22-da.
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path $TAG_DIR \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 1000 --batch_size 16 \
        --save_all_samples \
        --compute_comet \
        --comet_model_path /mnt/data/Summer-data/comet-wmt22-da \
        --output_path $RESULTS_DIR/wmt22.json \
        > $RESULTS_DIR/wmt22.log 2>&1
    echo "[inline_eval] BLEU+COMET done for step $STEP — see $RESULTS_DIR/wmt22.log"
    grep -E "BLEU|COMET" $RESULTS_DIR/wmt22.log | head -4 || true

    # PPL (cheap, ~3 min)
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_ppl.py \
        --model_path $TAG_DIR --valid_pt $SUMMER/output/valid_512.pt --batch_size 16 \
        --output_path $RESULTS_DIR/ppl.json > $RESULTS_DIR/ppl.log 2>&1 || true

    # Mono tasks sequentially (~25 min total)
    for spec in "lambada_openai:0" "piqa:5" "arc_challenge:25" "hellaswag:10"; do
        task=${spec%:*}; shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
            --model_path $TAG_DIR --task $task --num_fewshot $shots \
            --output_path $RESULTS_DIR/$task/result.json \
            > $RESULTS_DIR/$task.log 2>&1 || true
    done

    # MMLU full (~25 min)
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
        --model_path $TAG_DIR --task mmlu --num_fewshot 5 \
        --output_path $RESULTS_DIR/mmlu/result.json \
        > $RESULTS_DIR/mmlu.log 2>&1 || true
    echo "[inline_eval] FULL eval done for step $STEP"
else
    echo "[inline_eval] LIGHT eval for step $STEP (lambada + piqa + arc_c only)"
    for spec in "lambada_openai:0" "piqa:5" "arc_challenge:25"; do
        task=${spec%:*}
        shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
            --model_path $TAG_DIR \
            --task $task --num_fewshot $shots \
            --output_path $RESULTS_DIR/$task/result.json \
            > $RESULTS_DIR/$task.log 2>&1 || true
    done
    # also do a quick BLEU 200-sample to track translation trend (fast)
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
        --model_path $TAG_DIR \
        --testset wmt22 --exemplar_set wmt21 --direction both \
        --num_fewshot 5 --max_samples 200 --batch_size 16 \
        --output_path $RESULTS_DIR/wmt22_light.json \
        > $RESULTS_DIR/wmt22_light.log 2>&1 || true
    echo "[inline_eval] light eval done for step $STEP"
fi
