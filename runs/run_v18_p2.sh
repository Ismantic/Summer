#!/usr/bin/env bash
# v18 Phase 2: LoRA + Aurora + v12 anneal mix (200M token)
# 继承 v17_p2 配方(LoRA 这次最佳):r=16 q/v + modules_to_save embed/lm_head
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V18_P1=$SUMMER/output/phase1_ckpt_v18
ANNEAL_PT=$SUMMER/output/v18_anneal_1024.pt
V18_P2=$SUMMER/output/phase2_ckpt_v18

echo "=== [$(date +%H:%M:%S)] PHASE 2 v18: LoRA Aurora anneal seqlen 1024 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29521 train/finetune_muon.py \
    --model_path $V18_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V18_P2 \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_target q_proj,v_proj \
    --max_steps 1500 \
    --warmup_steps 200 \
    --muon_lr 5e-5 \
    --adam_lr 5e-5 \
    --min_lr_ratio 0.01 \
    --lr_schedule cosine \
    --max_grad_norm 1.0 \
    --use_aurora \
    --save_steps 500 \
    --logging_steps 50

echo "=== [$(date +%H:%M:%S)] v18 P2 DONE ==="
