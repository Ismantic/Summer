#!/usr/bin/env bash
# v8_s2 with Aurora: stress-test the Phase 2 LR ceiling.
#
# v5_s2 with muon_lr=5e-5 (Muon) destroyed translation. v7_s2 with
# muon_lr=1e-5 (Muon) was a no-op. v7_s2_v2 with muon_lr=2e-5 (Muon)
# moved BLEU slightly but stayed within noise.
# Aurora's row-uniform update should let us push muon_lr back up to
# 5e-5 without the catastrophic drift, because tall MLP matrices
# don't lose rows to neuron starvation.
#
# 500 steps from v8 ckpt, single-variable change: Muon -> Aurora.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
VALID_PT=$SUMMER/output/valid_512.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/phase2_ckpt_v8_s2_aurora

echo "=== [$(date +%H:%M:%S)] PHASE 2 v8_s2_aurora: muon_lr=5e-5 + AURORA ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V8 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --warmup_steps 100 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 50

TAG=phase2_v8_s2_aurora
echo "=== [$(date +%H:%M:%S)] eval $TAG ==="
QWEN_NEW=$OUT OUT=$RESULTS/smoke_$TAG bash $SUMMER/runs/smoke_test.sh \
    > $RESULTS/smoke_${TAG}.log 2>&1 || true
CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SUMMER/evals/eval_pretrain_translate.py \
    --model_path $OUT --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 200 --batch_size 4 \
    --output_path $RESULTS/translate_wmt22/$TAG.json \
    > $RESULTS/translate_wmt22/$TAG.log 2>&1 || true
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/evals/eval_ppl.py \
    --model_path $OUT --valid_pt $VALID_PT --batch_size 16 \
    --output_path $RESULTS/ppl/$TAG.json \
    > $RESULTS/ppl/$TAG.log 2>&1 || true

# Full MMLU — this is the big one. Aurora paper claims +10.8 pp here.
echo "=== [$(date +%H:%M:%S)] full MMLU on $TAG ==="
mkdir -p $RESULTS/mmlu_full/v8_s2_aurora
unset http_proxy https_proxy 2>/dev/null
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/evals/eval_with_piece.py \
    --model_path $OUT \
    --task mmlu --num_fewshot 5 \
    --output_path $RESULTS/mmlu_full/v8_s2_aurora/result.json \
    > $RESULTS/mmlu_full/v8_s2_aurora.log 2>&1 || true

echo "=== [$(date +%H:%M:%S)] v8_s2_aurora ALL DONE ==="
