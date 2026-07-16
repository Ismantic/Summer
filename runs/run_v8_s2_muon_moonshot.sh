#!/usr/bin/env bash
# v8_s2_muon_moonshot: Muon + Moonshot's per-param LR scaling + weight decay.
#
# Moonshot paper: lr is multiplied by 0.2 * sqrt(max(A, B)) per parameter,
# to keep update RMS consistent across heterogeneous matrix shapes.
# For Qwen3-0.6B-Base matrices:
#   attention 1536x1536 -> 7.84x
#   MLP 1536x8960       -> 18.93x
# So nominal muon_lr=5e-6 gives effective:
#   attention ~3.9e-5,  MLP ~9.5e-5
# Roughly bracketing our v8_s2_aurora (5e-5 uniform).
# Weight decay set to 0.1 per Moonshot paper.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
V8=$SUMMER/output/phase1_ckpt_v8
OUT=$SUMMER/output/phase2_ckpt_v8_s2_muon_moonshot

echo "=== [$(date +%H:%M:%S)] PHASE 2 v8_s2_muon_moonshot: muon_lr=5e-6 + MOONSHOT scaling + wd=0.1 ==="
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
    --muon_lr 5e-6 \
    --adam_lr 5e-5 \
    --weight_decay 0.1 \
    --moonshot_scaling \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] full eval on v8_s2_muon_moonshot ==="
bash $SUMMER/runs/eval_full.sh phase2_ckpt_v8_s2_muon_moonshot
echo "=== [$(date +%H:%M:%S)] v8_s2_muon_moonshot ALL DONE ==="
