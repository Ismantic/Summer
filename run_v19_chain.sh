#!/usr/bin/env bash
# v19 chain:等 pretok 完 → 启动训练(含 inline_eval 5 个中间评估)
set -u
SUMMER=/home/tfbao/Shiyu/Summer
PRETOK_PT=$SUMMER/output/v19_main_1024.pt

echo "[$(date +%F\ %H:%M:%S)] CHAIN v19 START - 等 pretok 输出"
until [ -f "$PRETOK_PT" ]; do sleep 60; done
echo "[$(date +%F\ %H:%M:%S)] pretok 就绪 ($(du -h $PRETOK_PT | awk '{print $1}'))"

pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

# 启动训练(inline_eval 在 step 5000/10000/20000/30000/38000 自动评估)
bash $SUMMER/run_v19_train.sh > $SUMMER/output/v19_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] v19 TRAIN END"
