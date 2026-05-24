#!/usr/bin/env bash
# 等 GPU 空闲(< 5GB)→ 启动 WMT23 全集评估
set -u
SUMMER=/home/tfbao/Shiyu/Summer

echo "[$(date +%F\ %H:%M:%S)] LAUNCHER START — 等到下午 11:00"

# 1. 等到 11:00
TARGET=$(date -d '11:00' +%s)
NOW=$(date +%s)
if [ "$NOW" -lt "$TARGET" ]; then
    SLEEP=$((TARGET - NOW))
    echo "[$(date +%F\ %H:%M:%S)] 等 $SLEEP 秒到 11:00"
    sleep $SLEEP
fi

# 2. 11:00 之后才轮询 GPU
echo "[$(date +%F\ %H:%M:%S)] 11:00 到,开始检查 GPU"
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$USED" -lt 5000 ]; then
        echo "[$(date +%F\ %H:%M:%S)] GPU 空 (${USED}MiB),启动 WMT23 评估"
        break
    fi
    echo "[$(date +%F\ %H:%M:%S)] GPU 还在用 (${USED}MiB),等 10 分钟"
    sleep 600
done

bash $SUMMER/run_v18_wmt23_eval.sh > $SUMMER/output/v18_wmt23_eval.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] WMT23 EVAL DONE"
