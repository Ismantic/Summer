#!/usr/bin/env bash
# v11 = v8 P1 ckpt + narrow-high-quality anneal Phase 2.
#
# Hypothesis: v10's Phase 2 annealing (+10% BLEU lift) was the real gain.
# v10's P1 mix (RegMix-style web+code+math) actually hurt BLEU baseline.
# v11 keeps v8 P1 unchanged (reuse existing ckpt-7500) and only tests
# whether the anneal Phase 2 lifts BLEU above v8 P2 ceiling.
#
# Anneal mix is v8-style narrow: 6 highest-quality EN+CN sources, NO code/math.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V8_P1=$SUMMER/output/phase1_ckpt_v8
ANNEAL_PT=$SUMMER/output/v11_anneal_512.pt
V11_P2=$SUMMER/output/phase2_ckpt_v11_aurora_anneal

# Step 1: pretokenize v11 anneal mix (0.4B target)
if [ ! -f "$ANNEAL_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v11 ANNEAL: 0.4B tokens ==="
  cd $SUMMER && $PYTHON data_prep/pretokenize_v10.py \
    --mix v11_anneal \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $ANNEAL_PT \
    --total_tokens 400000000 \
    --num_workers 16
fi

# Step 2: Phase 2 — unfreeze + Aurora on v8 P1 ckpt
#   2000 steps × eff_bs 128 × seq 512 ≈ 0.13B tokens (same as v10 P2 budget)
echo "=== [$(date +%H:%M:%S)] PHASE 2 v11: 2000 steps unfreeze Aurora narrow mix, muon/adam=5e-5 ==="
INLINE_EVAL="bash $SUMMER/runs/inline_eval_progressive.sh {step} $V11_P2 v11_p2 2000"
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29504 train/finetune_muon.py \
    --model_path $V8_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V11_P2 \
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
    --inline_eval_cmd "$INLINE_EVAL"

echo "=== [$(date +%H:%M:%S)] v11 DONE ==="
