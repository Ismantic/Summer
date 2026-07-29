#!/usr/bin/env bash
# Summer-0.5B 长跑的状态速查。自己跑,不用问 Claude。
#
#   bash prepare/s0_status.sh
#
# 跑六天的任务,状态该是一条命令就能看全的东西 —— 每次都去问一遍,对话
# 上下文要整段重发,那比训练本身还贵。
set -u
cd "$(dirname "$0")/.."

LOG=$(ls -t output/s0_train*.log 2>/dev/null | head -1)
CURVE=output/summer05b_s0/bleu_curve.jsonl
TOTAL=45149
SEC_PER_STEP=12.66

echo "日志  $LOG"

LAST=$(grep -E '^step' "$LOG" 2>/dev/null | tail -1)
if [ -z "$LAST" ]; then
  echo "还没有 step 行(torch.compile 要几分钟)"
else
  STEP=$(sed -E 's|^step ([0-9]+)/.*|\1|' <<<"$LAST")
  PCT=$(awk -v s="$STEP" -v t="$TOTAL" 'BEGIN{printf "%.1f", s/t*100}')
  ETA=$(awk -v s="$STEP" -v t="$TOTAL" -v r="$SEC_PER_STEP" \
        'BEGIN{printf "%.1f", (t-s)*r/86400}')
  echo "进度  $LAST"
  echo "      $PCT%  剩余约 $ETA 天"
fi

if pgrep -f 'src/train.py.*summer05b_s0' >/dev/null; then
  echo "进程  在跑 (pid $(pgrep -f 'src/train.py.*summer05b_s0' | head -1))"
else
  echo "进程  **不在跑** —— 续训:make -C prepare s0-resume STEP=<最后的 checkpoint>"
fi

echo
echo "BLEU(WMT22 5-shot / 200 条 / vLLM)"
if [ -f "$CURVE" ]; then
  awk -F'"' '{
    for (i=1;i<=NF;i++) {
      if ($i=="step") s=$(i+1)
      if ($i=="bleu_zh-en") z=$(i+1)
      if ($i=="bleu_en-zh") e=$(i+1)
    }
    gsub(/[^0-9.]/,"",s); gsub(/[^0-9.]/,"",z); gsub(/[^0-9.]/,"",e)
    printf "  step %-7s zh-en %-6s en-zh %s\n", s, z, e
  }' "$CURVE"
else
  echo "  (还没有点)"
fi

echo
echo "对照  Qwen3-0.6B-Base 17.66 / 32.31   ReTok-1.7B 20.46 / 36.03"
echo -n "GPU   "; nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader 2>/dev/null || echo "(取不到)"
echo -n "磁盘  "; df -h /home | tail -1 | awk '{print $4" 可用 ("$5" 已用)"}'
echo -n "ckpt  "; ls -d output/summer05b_s0/checkpoint-* 2>/dev/null | tr '\n' ' '; echo
