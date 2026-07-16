#!/usr/bin/env bash
# v18 chain v2:eval P1 → train P2 → eval P2,GPU 全程不空
set -u
SUMMER=/home/tfbao/Shiyu/Summer
P2_CKPT=$SUMMER/output/phase2_ckpt_v18/checkpoint-1500

echo "[$(date +%F\ %H:%M:%S)] CHAIN-v2 START — eval v18_p1"
bash $SUMMER/runs/run_v18_eval_one.sh v18_p1 $SUMMER/output/phase1_ckpt_v18 \
    > $SUMMER/output/v18_eval_p1.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] v18_p1 eval DONE → 启 P2 训练"

# 删除之前的 P2 ckpt(被 kill 留下的不完整 state)
rm -rf $SUMMER/output/phase2_ckpt_v18
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

bash $SUMMER/runs/run_v18_p2.sh > $SUMMER/output/v18_p2_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] P2 训练结束"

until [ -d "$P2_CKPT" ]; do sleep 30; done

pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

echo "[$(date +%F\ %H:%M:%S)] 启 v18_p2 eval"
bash $SUMMER/runs/run_v18_eval_one.sh v18_p2 $SUMMER/output/phase2_ckpt_v18 \
    > $SUMMER/output/v18_eval_p2.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] CHAIN-v2 ALL DONE"
