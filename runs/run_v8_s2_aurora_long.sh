#!/usr/bin/env bash
# v8_s2_aurora_long: scale Aurora Phase 2 to 3000 steps to test the
# "token efficiency" hypothesis from the Aurora blog. v8_s2_aurora (500
# steps) gave +0.023 score. If Aurora has any of the paper's claimed
# token-efficiency scaling, longer training should keep moving MMLU
# (currently the saturated metric, -8.1% vs base).
#
# Same nominal hyperparameters as v8_s2_aurora, only max_steps and
# warmup adjusted for the longer run. Architecture untouched.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/phase2_ckpt_v8_s2_aurora_long

echo "=== [$(date +%H:%M:%S)] PHASE 2 v8_s2_aurora_long: 3000 steps, Aurora muon_lr=5e-5 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V8 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 3000 \
    --warmup_steps 300 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 1000 \
    --logging_steps 100

# Eval all intermediate ckpts (1000/2000) + final (3000) — gives us
# the Aurora scaling curve in step count for free. eval_full.sh runs
# 2 at a time on GPU0+GPU1.
echo "=== [$(date +%H:%M:%S)] symlink intermediate ckpts as top-level tags ==="
for step in 1000 2000; do
    src=$OUT/checkpoint-$step
    dst=$SUMMER/output/phase2_ckpt_v8_s2_aurora_long_step${step}
    [ -d "$src" ] && ln -sfn "$src" "$dst"
done

echo "=== [$(date +%H:%M:%S)] full eval: aurora_long step1000 / step2000 / final ==="
bash $SUMMER/runs/eval_full.sh \
    phase2_ckpt_v8_s2_aurora_long_step1000 \
    phase2_ckpt_v8_s2_aurora_long_step2000 \
    phase2_ckpt_v8_s2_aurora_long
echo "=== [$(date +%H:%M:%S)] v8_s2_aurora_long ALL DONE ==="
