#!/usr/bin/env bash
set -u
SUMMER=/home/tfbao/Shiyu/Summer
P2_CKPT=$SUMMER/output/phase2_ckpt_v18/checkpoint-1500

echo "[$(date +%F\ %H:%M:%S)] CHAIN-v3 START: eval v18_p1 (offline COMET)"
bash $SUMMER/run_v18_eval_one.sh v18_p1 $SUMMER/output/phase1_ckpt_v18 \
    > $SUMMER/output/v18_eval_p1.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] v18_p1 eval DONE → 启 P2 训练"

rm -rf $SUMMER/output/phase2_ckpt_v18
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

bash $SUMMER/run_v18_p2.sh > $SUMMER/output/v18_p2_train.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] P2 训练结束"

until [ -d "$P2_CKPT" ]; do sleep 30; done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5

echo "[$(date +%F\ %H:%M:%S)] 启 v18_p2 eval(含 MMLU)"
ALL_TASKS="lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5,mmlu:5" \
    bash $SUMMER/run_v18_eval_one.sh v18_p2 $SUMMER/output/phase2_ckpt_v18 \
    > $SUMMER/output/v18_eval_p2.log 2>&1
echo "[$(date +%F\ %H:%M:%S)] CHAIN-v3 ALL DONE"
