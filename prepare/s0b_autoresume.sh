#!/usr/bin/env bash
# S0B(重做预训练)长跑的看门人 —— 同 s0_autoresume.sh 的理由,复制一份而不是
# 参数化,是因为两条线的 output 目录、日志前缀都不同,共用脚本反而容易在
# 硬重启后接错目录。
set -uo pipefail
cd "$(dirname "$0")/.."

S0B=output/summer05b_s0b

STEP=""
for n in $(ls -d "$S0B"/checkpoint-* 2>/dev/null |
           sed -E 's|.*/checkpoint-||' | sort -rn); do
  if [ -f "$S0B/checkpoint-$n/train_state.pt" ]; then STEP=$n; break; fi
done

if [ -z "$STEP" ]; then
  echo "[autoresume] 没有可续的 checkpoint,从头开始"
  exec make -C prepare s0b >> output/s0b_train.log 2>&1
fi

echo "[autoresume] 从 checkpoint-$STEP 续"
exec make -C prepare s0b-resume STEP="$STEP" >> "output/s0b_train_r$STEP.log" 2>&1
