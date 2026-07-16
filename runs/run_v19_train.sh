#!/usr/bin/env bash
# v19 from-scratch 训练:Qwen3-0.6B 架构 + v18 piece + 随机 init + 10B token
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

INIT=/home/tfbao/new/Qwen3-0.6B-fromscratch-v19
TRAIN_PT=$SUMMER/output/v19_main_1024.pt
OUT=$SUMMER/output/v19_train_ckpt

# 10B token / (eff_bs 256 × 1024) = 38,147 steps
# 实际取 38,000 步整,留 anneal 余地
echo "=== [$(date +%H:%M:%S)] v19 FROM-SCRATCH: Qwen3-0.6B arch + v18 piece + 10B token ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29523 train/finetune_muon.py \
    --model_path $INIT \
    --train_data $TRAIN_PT \
    --mode clm \
    --output_dir $OUT \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --compile \
    --max_steps 38000 \
    --warmup_steps 2000 \
    --muon_lr 6e-4 \
    --adam_lr 6e-4 \
    --min_lr_ratio 0.1 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 2500 \
    --logging_steps 50 \
    --inline_eval_cmd "bash $SUMMER/runs/run_v19_eval_callback.sh {step} $OUT v19"

echo "=== [$(date +%H:%M:%S)] v19 TRAIN DONE ==="
