#!/usr/bin/env bash
# v18 自动链:P1 训完 → 立即 P2 训 → 立即评估,GPU 全程不空
set -u
SUMMER=/home/tfbao/Shiyu/Summer
P1_CKPT=$SUMMER/output/phase1_ckpt_v18/checkpoint-3815/config.json
P2_CKPT=$SUMMER/output/phase2_ckpt_v18/checkpoint-1500
P2_PRETOK=$SUMMER/output/v18_anneal_1024.pt

echo "[$(date +%F\ %H:%M:%S)] CHAIN START — 等 P1 ckpt-3815"

# 等 P1 训练完(轮询 3815 final ckpt 出现)
until [ -f "$P1_CKPT" ]; do sleep 60; done
echo "[$(date +%F\ %H:%M:%S)] P1 DONE"

# 等 P2 pretok(几乎一定早完了)
until [ -f "$P2_PRETOK" ]; do sleep 30; done
echo "[$(date +%F\ %H:%M:%S)] P2 pretok 就绪"

# 清残留 vllm,让出 GPU
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

# 立即启 P2(同步等完)
echo "[$(date +%F\ %H:%M:%S)] 启 P2 训练"
bash $SUMMER/run_v18_p2.sh > $SUMMER/output/v18_p2_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] P2 训练结束"

# 双保险:轮询 ckpt 文件
until [ -d "$P2_CKPT" ]; do sleep 30; done

pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

# 立即启评估
echo "[$(date +%F\ %H:%M:%S)] 启评估"
bash $SUMMER/run_v18_eval.sh > $SUMMER/output/v18_eval.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] CHAIN ALL DONE"
