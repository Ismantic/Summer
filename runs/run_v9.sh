#!/usr/bin/env bash
# v9 = v8 mix + STEM data + scale-up.
#
# Goal: hit -5% loss on BOTH mono-LM and translation vs Qwen3-0.6B-Base.
# v8 sits at -6 to -8% mono and -7 to -12% BLEU. v9 plans:
#   - 4B-token pool (vs v8's 2B), 60/40 EN/CN (vs 65/35), +OpenWebMath 20%
#   - 1.5B training tokens (vs v8's 1B), 11500 steps eff_bs=256
#   - Phase 2: Aurora 500 steps from v9 ckpt (Phase 2 known-saturated at this scale)
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
TRAIN_PT=$SUMMER/output/phase1_train_512_v9.pt
VALID_PT=$SUMMER/output/valid_512.pt
V9=$SUMMER/output/phase1_ckpt_v9
V9_S2=$SUMMER/output/phase2_ckpt_v9_s2_aurora

# Step 1: pretokenize 4B-token pool (~90 min)
if [ ! -f "$TRAIN_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v9: 4B tokens ==="
  cd $SUMMER && $PYTHON data_prep/pretokenize_v9.py \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $TRAIN_PT \
    --total_tokens 4000000000
fi

# Step 2: Phase 1 training
#   11500 steps × eff_bs 256 × seq 512 = ~1.5B tokens
echo "=== [$(date +%H:%M:%S)] PHASE 1 v9: 11500 steps, eff bs=256, lr=1e-4 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V9 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 11500 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 3000 \
    --logging_steps 100

echo "=== [$(date +%H:%M:%S)] eval phase1_v9 (full suite) ==="
bash $SUMMER/runs/eval_full.sh phase1_ckpt_v9

# Step 3: Phase 2 (Aurora 500 steps) — known-saturated regime
echo "=== [$(date +%H:%M:%S)] PHASE 2 v9_s2_aurora: 500 steps, Aurora muon=5e-5 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V9 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $V9_S2 \
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

echo "=== [$(date +%H:%M:%S)] eval phase2_v9_s2_aurora (full suite) ==="
bash $SUMMER/runs/eval_full.sh phase2_ckpt_v9_s2_aurora

echo "=== [$(date +%H:%M:%S)] v9 ALL DONE ==="
