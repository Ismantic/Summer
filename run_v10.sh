#!/usr/bin/env bash
# v10 = ReTok proper: web-heavy P1 (RegMix-informed) + annealed P2 (high-quality narrow).
#
# Phase 1: frozen transformer, broad mix (FineWebEdu+SkyPile bulk + Wiki + Code 9% + Math 7%)
# Phase 2: unfreeze + Aurora, anneal mix (Code 20%, Math 15%, Wiki 20%, FineWebEdu 45%)
#
# Targets: ~0.65B P1 + ~0.13B P2 = 0.78B total. ~7.8h training + eval.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
MAIN_PT=$SUMMER/output/v10_main_512.pt
ANNEAL_PT=$SUMMER/output/v10_anneal_512.pt
V10_P1=$SUMMER/output/phase1_ckpt_v10
V10_P2=$SUMMER/output/phase2_ckpt_v10_aurora_anneal

# Step 1: pretokenize main mix (~2B pool, plenty)
if [ ! -f "$MAIN_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v10 MAIN: 2B tokens ==="
  cd $SUMMER && $PYTHON pretokenize_v10.py \
    --mix main \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $MAIN_PT \
    --total_tokens 2000000000 \
    --num_workers 28
fi

# Step 2: pretokenize anneal mix (~0.4B pool, enough for 1000 steps × 128K/step = 0.13B)
if [ ! -f "$ANNEAL_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v10 ANNEAL: 0.4B tokens ==="
  cd $SUMMER && $PYTHON pretokenize_v10.py \
    --mix anneal \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $ANNEAL_PT \
    --total_tokens 400000000 \
    --num_workers 16
fi

# Step 3: Phase 1 — frozen transformer, broad mix
#   5000 steps × eff_bs 256 × seq 512 = ~0.65B tokens
echo "=== [$(date +%H:%M:%S)] PHASE 1 v10: 5000 steps frozen, broad mix, lr=1e-4 ==="
INLINE_EVAL_P1="bash $SUMMER/inline_eval_progressive.sh {step} $V10_P1 v10_p1 5000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29503 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $MAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V10_P1 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 5000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P1"

echo "=== [$(date +%H:%M:%S)] eval phase1_v10 (full) ==="
ln -sfn $V10_P1/checkpoint-5000 $SUMMER/output/phase1_ckpt_v10_final
bash $SUMMER/eval_full.sh phase1_ckpt_v10_final

# Step 4: Phase 2 — unfreeze, Aurora on annealed mix
#   1000 steps × eff_bs 128 (bs16 × accum 8 × 2 GPUs / 2 = wait recalc) — use accum 4 like v8
#   1000 × 16 × 4 × 512 × 2 = 65M-ish? Hmm. Let me redo: bs 16 × accum 4 × 2 GPUs = 128 eff bs, × seq 512 = 65K tok/step. × 1000 = 65M. Want ~0.13B → 2000 steps. Use 2000.
echo "=== [$(date +%H:%M:%S)] PHASE 2 v10: 2000 steps unfreeze Aurora annealed mix, muon/adam=5e-5 ==="
INLINE_EVAL_P2="bash $SUMMER/inline_eval_progressive.sh {step} $V10_P2 v10_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29503 finetune_muon.py \
    --model_path $V10_P1/checkpoint-5000 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V10_P2 \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 2000 \
    --logging_steps 100 \
    --inline_eval_cmd "$INLINE_EVAL_P2"

echo "=== [$(date +%H:%M:%S)] eval phase2_v10 (full) ==="
ln -sfn $V10_P2/checkpoint-2000 $SUMMER/output/phase2_ckpt_v10_final
bash $SUMMER/eval_full.sh phase2_ckpt_v10_final

echo "=== [$(date +%H:%M:%S)] v10 ALL DONE ==="
