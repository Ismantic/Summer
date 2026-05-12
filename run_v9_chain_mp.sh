#!/usr/bin/env bash
# v9 chain: parallel pretokenize → Phase 1 (with inline eval) → Phase 2 (with inline eval).
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
SUMMER=/home/tfbao/Shiyu/Summer
TRAIN_PT=$SUMMER/output/phase1_train_512_v9.pt

if [ ! -f "$TRAIN_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v9 (mp, 28 workers): 4B tokens ==="
  cd $SUMMER && $PYTHON pretokenize_v9_mp.py \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $TRAIN_PT \
    --total_tokens 4000000000 \
    --num_workers 28
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE DONE ==="
else
  echo "[$(date +%H:%M:%S)] $TRAIN_PT exists, skipping pretokenize"
fi

echo "=== [$(date +%H:%M:%S)] HANDOFF -> run_v9_phase1.sh ==="
bash $SUMMER/run_v9_phase1.sh
