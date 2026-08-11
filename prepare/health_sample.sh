#!/usr/bin/env bash
# 每 30 秒记一行机器状态到 output/health.csv,给下一次硬重启留现场。
#
# 2026-08-02 05:36 那次重启什么都没留下:pstore 空、没有 panic、journal 因为
# SyncIntervalSec=5m 断在崩溃前 8 分钟。连是掉电、硬锁还是 MCE 都判断不了。
# 这个采样器用 O_APPEND 逐行写并 sync,掉电时最多丢最后一行。
#
#   make -C prepare health-service   装成用户服务
#   tail -5 output/health.csv        看
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=output/health.csv
[ -f "$CSV" ] || echo "ts,cpu_c,gpu_c,gpu_w,gpu_mhz,gpu_util,gpu_mem_mib,throttle,load1,mem_avail_mib" > "$CSV"

while true; do
  ts=$(date -Is)
  cpu=$(sensors -u k10temp-pci-00c3 2>/dev/null |
        awk '/temp1_input/{printf "%.1f", $2; exit}')
  read -r gt gw gc gu gm < <(
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used \
               --format=csv,noheader,nounits 2>/dev/null | tr -d ',')
  # 降频原因:HW Slowdown / Power Brake 出现就是供电或散热到极限了
  thr=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null |
        awk -F: '/Slowdown|Power Brake/{gsub(/ /,"",$2); if($2=="Active") print $1}' |
        tr -d ' ' | paste -sd'+' -)
  load1=$(awk '{print $1}' /proc/loadavg)
  memav=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  echo "$ts,${cpu:-},${gt:-},${gw:-},${gc:-},${gu:-},${gm:-},${thr:-none},$load1,$memav" >> "$CSV"
  sync -d "$CSV" 2>/dev/null
  sleep 30
done
