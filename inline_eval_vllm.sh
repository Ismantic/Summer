#!/usr/bin/env bash
# vLLM-based inline eval — drop-in replacement for inline_eval_progressive.sh.
# Uses eval_pretrain_translate_vllm.py so BLEU is stable across transformers
# versions (May 14 upgrade broke BLEU reproducibility; vLLM bypasses .generate).
#
# Args:
#   $1 = step number
#   $2 = output ckpt root (output_dir from training)
#   $3 = run tag prefix
#   $4 = final step (default 3000) — when STEP==FINAL_STEP, add mono+PPL
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

# Strip torchrun env so vLLM's torch.distributed doesn't try to rendezvous
# with the parent training group (causes 10-min hang + failure).
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
unset GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE ROLE_NAME
unset TORCHELASTIC_RUN_ID TORCHELASTIC_USE_AGENT_STORE TORCHELASTIC_MAX_RESTARTS
unset TORCHELASTIC_RESTART_COUNT TORCHELASTIC_ERROR_FILE
unset PYTORCH_CUDA_ALLOC_CONF TORCH_NCCL_ASYNC_ERROR_HANDLING

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

echo "[inline_eval_vllm] BLEU+COMET 1000-sample for step $STEP"
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_pretrain_translate_vllm.py \
    --model_path $TAG_DIR \
    --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 1000 \
    --save_all_samples \
    --compute_comet \
    --comet_model_path /mnt/data/Summer-data/comet-wmt22-da \
    --output_path $RESULTS_DIR/wmt22.json \
    > $RESULTS_DIR/wmt22.log 2>&1
echo "[inline_eval_vllm] BLEU+COMET done for step $STEP"
grep -E "BLEU|COMET" $RESULTS_DIR/wmt22.log | head -4 || true

if [ "$STEP" = "$FINAL_STEP" ]; then
    echo "[inline_eval_vllm] FINAL: add PPL + mono + MMLU"

    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_ppl.py \
        --model_path $TAG_DIR --valid_pt $SUMMER/output/valid_512.pt --batch_size 16 \
        --output_path $RESULTS_DIR/ppl.json > $RESULTS_DIR/ppl.log 2>&1 || true

    for spec in "lambada_openai:0" "piqa:5" "arc_challenge:25" "hellaswag:10"; do
        task=${spec%:*}; shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
            --model_path $TAG_DIR --task $task --num_fewshot $shots \
            --output_path $RESULTS_DIR/$task/result.json \
            > $RESULTS_DIR/$task.log 2>&1 || true
    done

    CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
        --model_path $TAG_DIR --task mmlu --num_fewshot 5 \
        --output_path $RESULTS_DIR/mmlu/result.json \
        > $RESULTS_DIR/mmlu.log 2>&1 || true
    echo "[inline_eval_vllm] FINAL FULL eval done for step $STEP"
else
    echo "[inline_eval_vllm] intermediate done for step $STEP (translation only)"
fi
