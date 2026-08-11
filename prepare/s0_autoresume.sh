#!/usr/bin/env bash
# S0 长跑的看门人:找到最新的可续 checkpoint 接上,一个都没有就从头开。
#
# 给 systemd 用户服务调用(`make -C prepare s0-service` 装),自己手跑也行。
#
# 为什么要有这个:2026-08-02 05:36 本机硬重启(掉电签名 —— wtmp 记 crash、
# journal 无 shutdown 序列、无 panic),训练没了,而当时挂的监控是绑在会话
# 上的,会话一没监控也没,结果 GPU 空转 3.5 小时才被发现。让开机自己把训练
# 拉回来,比让人记得去拉可靠。
set -uo pipefail
cd "$(dirname "$0")/.."

S0=output/summer05b_s0

# 最新的、并且带 train_state.pt 的那个 —— 没有优化器状态的 checkpoint 续不了,
# 续了也是从零重建动量,轨迹会断。
STEP=""
for n in $(ls -d "$S0"/checkpoint-* 2>/dev/null |
           sed -E 's|.*/checkpoint-||' | sort -rn); do
  if [ -f "$S0/checkpoint-$n/train_state.pt" ]; then STEP=$n; break; fi
done

if [ -z "$STEP" ]; then
  echo "[autoresume] 没有可续的 checkpoint,从头开始"
  exec make -C prepare s0 >> output/s0_train.log 2>&1
fi

echo "[autoresume] 从 checkpoint-$STEP 续"
exec make -C prepare s0-resume STEP="$STEP" >> "output/s0_train_r$STEP.log" 2>&1
