#!/usr/bin/env bash
# S1 退火的看门人。和 s0_autoresume.sh 同一套逻辑,只是接的是 S1 那条线:
# 第一次从 S0 的 checkpoint-$S1_FROM 起,之后从 S1 自己最新的 checkpoint 续。
#
#   make -C prepare s1-service S1_FROM=40000
set -uo pipefail
cd "$(dirname "$0")/.."

S1=output/summer05b_s1
FROM=${S1_FROM:-40000}

# 最新的、且带 train_state.pt 的 —— 没有优化器状态就续不了(Muon 的动量会
# 从零重建,轨迹会断)。S1 自己的优先,没有才回到 S0 的起点。
STEP=""
for n in $(ls -d "$S1"/checkpoint-* 2>/dev/null |
           sed -E 's|.*/checkpoint-||' | sort -rn); do
  if [ -f "$S1/checkpoint-$n/train_state.pt" ]; then STEP=$n; break; fi
done

if [ -n "$STEP" ]; then
  echo "[autoresume] S1 从自己的 checkpoint-$STEP 续"
  exec make -C prepare s1-resume STEP="$STEP" >> "output/s1_train_r$STEP.log" 2>&1
fi

echo "[autoresume] S1 从 S0 的 checkpoint-$FROM 起"
exec make -C prepare s1 STEP="$FROM" >> "output/s1_train.log" 2>&1
