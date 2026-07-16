#!/usr/bin/env bash
# 23:55 之后启动:等 GPU 空闲 → P2 tie-safe 训练 → 评估
set -u
SUMMER=/home/tfbao/Shiyu/Summer

# 1. 等到 23:55 之后(用户指定时间)
TARGET=$(date -d '23:55' +%s)
NOW=$(date +%s)
if [ "$NOW" -lt "$TARGET" ]; then
    SLEEP=$((TARGET - NOW))
    echo "[$(date +%F\ %H:%M:%S)] 等 $SLEEP 秒到 23:55"
    sleep $SLEEP
fi

# 2. 等 GPU 空闲(显存 < 5GB)
echo "[$(date +%F\ %H:%M:%S)] 检查 GPU 空闲"
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$USED" -lt 5000 ]; then
        echo "[$(date +%F\ %H:%M:%S)] GPU 空 (${USED}MiB),启动 P2 tie-safe"
        break
    fi
    echo "[$(date +%F\ %H:%M:%S)] GPU 还在用 (${USED}MiB),等 5 分钟"
    sleep 300
done

# 3. 启 P2 tie-safe 训练
bash $SUMMER/runs/run_v18_p2_tie.sh > $SUMMER/output/v18_p2_tie_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] P2 tie 训练结束"

# 4. ckpt-1500 出现后启评估(到新 tag)
until [ -d "$SUMMER/output/phase2_ckpt_v18_tie/checkpoint-1500" ]; do sleep 30; done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
echo "[$(date +%F\ %H:%M:%S)] 启 v18_p2_tie eval"
bash $SUMMER/runs/run_v18_eval_one.sh v18_p2_tie $SUMMER/output/phase2_ckpt_v18_tie \
    > $SUMMER/output/v18_eval_p2_tie.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] CHAIN-tie ALL DONE"
