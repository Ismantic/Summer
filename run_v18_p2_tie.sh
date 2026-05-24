#!/usr/bin/env bash
# v18 Phase 2 - tie-safe 版本:LoRA q/v + 手动 unfreeze embed_tokens
# (lm_head 通过 tie 自动同步,保持 tie_word_embeddings=true)
# 输出新目录 phase2_ckpt_v18_tie,不覆盖现有 phase2_ckpt_v18(Interpreter 仍在用)
set -e
export PYTHONUNBUFFERED=1

TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer

V18_P1=$SUMMER/output/phase1_ckpt_v18
ANNEAL_PT=$SUMMER/output/v18_anneal_1024.pt
V18_P2_TIE=$SUMMER/output/phase2_ckpt_v18_tie

echo "=== [$(date +%H:%M:%S)] PHASE 2 v18 TIE-SAFE: LoRA + tied embed/lm_head ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=1 --master_port=29522 finetune_muon.py \
    --model_path $V18_P1 \
    --train_data $ANNEAL_PT \
    --mode clm \
    --output_dir $V18_P2_TIE \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --compile \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_target q_proj,v_proj \
    --lora_tie_embed_head \
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

echo "=== [$(date +%H:%M:%S)] v18 P2 TIE DONE ==="
