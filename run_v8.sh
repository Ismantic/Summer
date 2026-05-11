#!/usr/bin/env bash
# v8 = scale Phase 1 to 1B training tokens on 2B-token diverse pool.
# Same v7 mix (65/35 EN/CN, 10 sources). Skipping Phase 2 since v7
# showed it's a no-op at this scale.
#
# Pretokenize 2B (~3h) -> train 7500 steps × eff_bs 256 × 512 = 1B tokens
# (~10.5h) -> eval suite + full MMLU (~50min). Total ~14h.
set -e
export PYTHONUNBUFFERED=1

PYTHON=/home/tfbao/.venv/bin/python
TORCHRUN=/home/tfbao/.venv/bin/torchrun
SUMMER=/home/tfbao/Shiyu/Summer
RESULTS=$SUMMER/eval_results

NEW_TOK=/home/tfbao/new/Qwen3-0.6B-Base-new-tok
TRAIN_PT=$SUMMER/output/phase1_train_512_v8.pt
VALID_PT=$SUMMER/output/valid_512.pt
V8=$SUMMER/output/phase1_ckpt_v8

# ------------------------------------------------------------------------
# Step 1: pretokenize 2B-token pool (idempotent; skip if .pt exists)
# ------------------------------------------------------------------------
if [ ! -f "$TRAIN_PT" ]; then
  echo "=== [$(date +%H:%M:%S)] PRETOKENIZE v8: 2B tokens ==="
  cd $SUMMER && $PYTHON pretokenize_v7.py \
    --tokenizer_model ./piece_mt.model \
    --cn_dict ./dict.txt \
    --output $TRAIN_PT \
    --total_tokens 2000000000
else
  echo "=== [$(date +%H:%M:%S)] $TRAIN_PT exists, skipping pretokenize ==="
fi

# ------------------------------------------------------------------------
# Step 2: Phase 1 training
#   7500 steps × eff_bs 256 × 512 = ~1B tokens (≈ 0.5 epoch on 2B pool)
#   warmup 1000 (ReTok-style, longer than v7's 500 since 2.5x total steps)
# ------------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] PHASE 1 v8: 7500 steps, eff bs=256, lr=1e-4 ==="
cd $SUMMER && $TORCHRUN --nproc_per_node=2 --master_port=29501 finetune_muon.py \
    --model_path $NEW_TOK \
    --train_data $TRAIN_PT \
    --mode clm \
    --freeze_transformer \
    --output_dir $V8 \
    --max_seq_length 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_steps 7500 \
    --warmup_steps 1000 \
    --adam_lr 1e-4 \
    --max_grad_norm 1.0 \
    --save_steps 2500 \
    --logging_steps 100

# ------------------------------------------------------------------------
# Step 3: standard eval (smoke + WMT22 BLEU + valid PPL)
# ------------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] eval phase1_v8 (standard suite) ==="
TAG=phase1_v8
QWEN_NEW=$V8 OUT=$RESULTS/smoke_$TAG bash $SUMMER/smoke_test.sh \
    > $RESULTS/smoke_${TAG}.log 2>&1 || true
CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SUMMER/eval_pretrain_translate.py \
    --model_path $V8 --testset wmt22 --exemplar_set wmt21 --direction both \
    --num_fewshot 5 --max_samples 200 --batch_size 4 \
    --output_path $RESULTS/translate_wmt22/$TAG.json \
    > $RESULTS/translate_wmt22/$TAG.log 2>&1 || true
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SUMMER/eval_ppl.py \
    --model_path $V8 --valid_pt $VALID_PT --batch_size 16 \
    --output_path $RESULTS/ppl/$TAG.json \
    > $RESULTS/ppl/$TAG.log 2>&1 || true

# ------------------------------------------------------------------------
# Step 4: full MMLU (57 subjects, ±0.004 stderr — definitive answer)
# ------------------------------------------------------------------------
echo "=== [$(date +%H:%M:%S)] full MMLU on phase1_v8 ==="
mkdir -p $RESULTS/mmlu_full/v8
unset http_proxy https_proxy 2>/dev/null
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 $PYTHON $SUMMER/eval_with_piece.py \
    --model_path $V8 \
    --task mmlu --num_fewshot 5 \
    --output_path $RESULTS/mmlu_full/v8/result.json \
    > $RESULTS/mmlu_full/v8.log 2>&1 || true

echo "=== [$(date +%H:%M:%S)] v8 ALL DONE ==="
