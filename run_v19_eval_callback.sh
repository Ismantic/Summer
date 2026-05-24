#!/usr/bin/env bash
# v19 inline_eval callback - 由 finetune_muon.py 每次 save 后触发。
# 只在 5 个关键步评估,其他 step skip(避免每 2500 步都评估的开销)。
# 触发时:模型/优化器已被 _inline_eval offload 到 CPU,GPU 显存可用。
set -u
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE 2>/dev/null

STEP=$1
CKPT_ROOT=$2
PREFIX=$3

# 目标评估步(精确等于)
case "$STEP" in
    5000|10000|20000|30000|38000)
        echo "[$(date +%H:%M:%S)] [v19_eval] step $STEP — 跑完整评估"
        ;;
    *)
        echo "[$(date +%H:%M:%S)] [v19_eval] step $STEP — skip (非评估点)"
        exit 0
        ;;
esac

CKPT=$CKPT_ROOT/checkpoint-$STEP
TAG=${PREFIX}_step${STEP}
SUMMER=/home/tfbao/Shiyu/Summer

if [ ! -d "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] [v19_eval] WARN: $CKPT not found, skip"
    exit 0
fi

# 拷贝 tokenizer artifacts 到 ckpt(_copy_tokenizer_artifacts 在 save 时已经做了,以防万一)
for f in piece.model dict.txt token_mapping.json special_tokens_map.json tokenizer_config.json; do
    if [ -f "$CKPT_ROOT/$f" ] && [ ! -f "$CKPT/$f" ]; then
        cp "$CKPT_ROOT/$f" "$CKPT/$f"
    fi
done

bash $SUMMER/run_v18_eval_one.sh $TAG $CKPT
