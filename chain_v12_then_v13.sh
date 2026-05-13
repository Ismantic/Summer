#!/usr/bin/env bash
# Wait for v12 to fully complete (P2 eval done), then auto-launch v13.
#
# Detection signal: v12 P2 mmlu eval result exists (last task in inline_eval).
set -e
export PYTHONUNBUFFERED=1

SUMMER=/home/tfbao/Shiyu/Summer
SIGNAL=$SUMMER/eval_results/full/v12_p2_step2000/mmlu/result.json

echo "[$(date +%H:%M:%S)] Watching for v12 P2 completion signal: $SIGNAL"
until [ -f "$SIGNAL" ]; do
  sleep 60
done

echo "[$(date +%H:%M:%S)] v12 P2 mmlu result detected — v12 complete. Launching v13..."
sleep 30   # let any tail processes settle (model offload back, etc)

bash $SUMMER/run_v13.sh
