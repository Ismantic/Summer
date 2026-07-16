#!/usr/bin/env bash
# Smoke test for --inline_eval_cmd CPU-offload mechanism.
# 100-step Aurora training with save_steps=50 → triggers 2 inline-eval
# cycles. Eval cmd is just `echo + ls` — verifies the offload/resume
# machinery works without running real eval (which takes 25+ min).
#
# Pass criteria:
#   1. training reaches step 100 without OOM / crash
#   2. checkpoint-50 and checkpoint-100 both saved
#   3. "[inline_eval] running" and "[inline_eval] done" both appear twice
#   4. step 51-100 training loss continues sensibly (no NaN, no explosion)
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/test_inline_eval

rm -rf $OUT
INLINE_CMD="echo '[test] inline_eval at step {step}'; ls -la $OUT/checkpoint-{step} | head -5; echo '[test] subprocess done at step {step}'"

echo "=== [$(date +%H:%M:%S)] smoke test inline_eval ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 train/finetune_muon.py \
    --model_path $V8 \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 100 \
    --warmup_steps 20 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 50 \
    --logging_steps 25 \
    --inline_eval_cmd "$INLINE_CMD"

echo "=== [$(date +%H:%M:%S)] smoke test DONE ==="
echo "expected: ckpts at step 50 and 100, 2 inline_eval cycles"
ls $OUT/ 2>/dev/null
