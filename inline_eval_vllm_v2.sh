#!/usr/bin/env bash
# vLLM-end-to-end inline eval — translation + mono all on vLLM.
#
# Replaces inline_eval_vllm.sh (which used eval_with_piece.py / transformers
# for mono tasks). Mono loglikelihood now runs ~5-10x faster via vLLM's
# prompt_logprobs feature.
#
# Args mirror v1:
#   $1 = step number
#   $2 = output ckpt root
#   $3 = run tag prefix
#   $4 = final step (mono only triggered when STEP==FINAL_STEP)
set -e
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com

# Strip torchrun env so vLLM's torch.distributed doesn't rendezvous with
# the parent training group.
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

echo "[inline_eval_v2] BLEU+COMET 1000-sample for step $STEP"
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_pretrain_translate_vllm.py \
    --model_path $TAG_DIR \
    --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 1000 \
    --save_all_samples \
    --compute_comet \
    --comet_model_path /mnt/data/Summer-data/comet-wmt22-da \
    --output_path $RESULTS_DIR/wmt22.json \
    > $RESULTS_DIR/wmt22.log 2>&1
echo "[inline_eval_v2] BLEU+COMET done for step $STEP"
grep -E "BLEU|COMET" $RESULTS_DIR/wmt22.log | head -4 || true

if [ "$STEP" = "$FINAL_STEP" ]; then
    echo "[inline_eval_v2] FINAL: add PPL + vLLM-mono"

    # PPL (transformers-backed; cheap and stable on loglikelihood)
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_ppl.py \
        --model_path $TAG_DIR --valid_pt $SUMMER/output/valid_512.pt --batch_size 16 \
        --output_path $RESULTS_DIR/ppl.json > $RESULTS_DIR/ppl.log 2>&1 || true

    # Mono tasks via vLLM (5-10x faster than transformers backend)
    for spec in "lambada_openai:0" "piqa:5" "arc_challenge:25" "hellaswag:10" "mmlu:5"; do
        task=${spec%:*}; shots=${spec#*:}
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece_vllm.py \
            --model_path $TAG_DIR --task $task --num_fewshot $shots \
            --max_model_len 4096 \
            --output_path $RESULTS_DIR/$task/result.json \
            > $RESULTS_DIR/$task.log 2>&1 || true
    done
    echo "[inline_eval_v2] FINAL FULL eval done for step $STEP"
else
    echo "[inline_eval_v2] intermediate done for step $STEP (translation only)"
fi
