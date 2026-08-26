#!/usr/bin/env bash
# S1B(S0B 的退火段)长跑的看门人 —— 同 s0b_autoresume.sh 的理由,复制一份
# 而不是参数化,避免硬重启后接错目录。
set -uo pipefail
cd "$(dirname "$0")/.."

S1B=output/summer05b_s1b

STEP=""
for n in $(ls -d "$S1B"/checkpoint-* 2>/dev/null |
           sed -E 's|.*/checkpoint-||' | sort -rn); do
  if [ -f "$S1B/checkpoint-$n/train_state.pt" ]; then STEP=$n; break; fi
done

if [ -z "$STEP" ]; then
  echo "[autoresume] 没有可续的 checkpoint,从头开始(从 S0B 分叉)"
  exec make -C prepare s1b >> output/s1b_train.log 2>&1
fi

echo "[autoresume] 从 checkpoint-$STEP 续"
exec make -C prepare s1b-resume STEP="$STEP" >> "output/s1b_train_r$STEP.log" 2>&1
