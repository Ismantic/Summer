#!/usr/bin/env bash
# v16 = v15 P1 (sota Phase 1) + Phase 2 annealing on v12 anneal mix.
#
# Background:
# - v15 P1 set new COMET sota (zh-en -7.9%, en-zh -6.5% vs base, via vLLM eval)
# - v15 has no Phase 2 yet. v10/v11/v12 P2 added +0.01–0.02 COMET on average.
# - Goal: push COMET losses inside -5% (stretch goal) — BLEU expected to lag.
#
# Recipe choices:
# - data: v12_anneal_512.pt (HQ-concentrated EN:CN=60:40, Cosmopedia + CN_FineWeb_Edu).
#         Already pretokenized, proven on v10/v11/v12.
# - schedule: cosine + min_lr_ratio 0.01 (matches v15 P1, "Reuse Don't Retrain")
# - 2000 steps, save every 500 (4 saves) — matches v10/v11/v12 P2 for direct compare
# - warmup 200 (vs v14's 100) — v15 ended at lr=1e-6, jump to 5e-5 is 50x, give
#   a bit more runway before peak
# - Aurora on (per v10+ practice)
# - inline_eval uses vLLM (transformers 4.57.6 broke BLEU repro; vLLM is engine-stable)
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V15_P1=$SUMMER/output/phase1_ckpt_v15
ANNEAL_PT=$SUMMER/output/v12_anneal_512.pt
V16_P2=$SUMMER/output/phase2_ckpt_v16_aurora_anneal

echo "=== [$(date +%H:%M:%S)] PHASE 2 v16: 2000 steps unfreeze Aurora anneal, cosine min_lr 0.01 ==="
INLINE_EVAL_P2="bash $SUMMER/inline_eval_vllm.sh {step} $V16_P2 v16_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29516 finetune_muon.py \
    --model_path $V15_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V16_P2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] v16 P2 DONE ==="
