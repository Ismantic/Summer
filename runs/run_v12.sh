#!/usr/bin/env bash
# v12 = clean general data mix with HQ upgrades, no code/math.
#
# Phase 1 (frozen, broad HQ): 9 sources, EN:CN = 60:40
#   New: Cosmopedia v2 (synthetic textbooks) + Chinese-FineWeb-Edu V2.2 (CN edu-filtered)
#   Removed vs v8: PeopleDaily, CnnDailyMail, C4_CN (low-signal)
#
# Phase 2 (unfreeze + Aurora, concentrated HQ): 6 sources
#   Drop SkyPile/Gutenberg/C4_EN — keep only top-tier filtered/encyclopedic/synthetic.
#
# Hypothesis: Cosmopedia + CN-FineWeb-Edu break the -6.4% mono / -7.7% BLEU ceiling
# that v8/v10/v11 all converged to. If v12 also plateaus, ceiling is structural.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
MAIN_PT=$SUMMER/output/v12_main_512.pt
ANNEAL_PT=$SUMMER/output/v12_anneal_512.pt
V12_P1=$SUMMER/output/phase1_ckpt_v12
V12_P2=$SUMMER/output/phase2_ckpt_v12_aurora_anneal

# Step 1: pretokenize main mix (2B pool target)
if [ ! -f "$MAIN_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v12 MAIN: 2B tokens ==="
  cd $SUMMER && $PYTHON data_prep/pretokenize_v12.py \
    --mix main \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $MAIN_PT \
    --total_tokens 2000000000 \
    --num_workers 28
fi

# Step 2: pretokenize anneal mix (0.4B pool target)
if [ ! -f "$ANNEAL_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v12 ANNEAL: 0.4B tokens ==="
  cd $SUMMER && $PYTHON data_prep/pretokenize_v12.py \
    --mix anneal \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $ANNEAL_PT \
    --total_tokens 400000000 \
    --num_workers 16
fi

# Step 3: Phase 1 — frozen transformer, broad mix
echo "=== [$(date +%H:%M:%S)] PHASE 1 v12: 5000 steps frozen, broad mix, lr=1e-4 ==="
INLINE_EVAL_P1="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V12_P1 v12_p1 5000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29505 train/finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $MAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V12_P1 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P1"

# Step 4: Phase 2 — unfreeze + Aurora on annealed mix
echo "=== [$(date +%H:%M:%S)] PHASE 2 v12: 2000 steps unfreeze Aurora anneal mix, muon/adam=5e-5 ==="
INLINE_EVAL_P2="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V12_P2 v12_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29505 train/finetune_muon.py \
    --model_path $V12_P1/checkpoint-5000 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V12_P2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] v12 DONE ==="
