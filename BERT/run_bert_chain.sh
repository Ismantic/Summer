#!/usr/bin/env bash
# 等 v19 pretok 完成 → 立即启 BERT-base MLM 训练
set -u
SUMMER=/home/tfbao/Shiyu/Summer
V19_PT=$SUMMER/output/v19_main_1024.pt

echo "[$(date +%F\ %H:%M:%S)] BERT CHAIN START — 等 v19 pretok 输出"
until [ -f "$V19_PT" ]; do sleep 60; done
echo "[$(date +%F\ %H:%M:%S)] v19 pretok 完 ($(du -h $V19_PT 2>/dev/null | awk '{print $1}')) — 启 BERT 训练"

# BERT 用的是 v18_main_1024.pt(头 1B token),不依赖 v19 输出,但 v19 跑完释放 CPU
bash $SUMMER/BERT/run_bert_train.sh > $SUMMER/BERT/bert_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] BERT TRAIN END"
