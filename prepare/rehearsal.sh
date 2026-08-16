#!/usr/bin/env bash
# 流程预演:用 depth-10(114M)走一遍设计 B 的全链路,再上 0.5B。
#
#     bash prepare/rehearsal.sh [mix名]        默认 chat
#
# 目的是**验链路和筛配方**,不是出成绩。词表 81903 固定,114M 的嵌入占 46%、
# 0.5B 占 16%,**不是同一种模型 —— 这里的绝对值不能推到 0.5B**。
# 能推的是方向:中文条数占比那条曲线在 114M 和 0.5B 上四个点重合过。
#
# ## 为什么在仓库里而不是临时目录
#
# 这个脚本被 `/tmp` 清掉过**三次**(机器硬复位一次、会话结束两次),每次都要
# 重写。`prepare/stoprate.py` 也是同样的原因搬进来的。**反复用的编排和判据
# 不能住 /tmp。**
#
# ## 每一步幂等 —— 重跑这个脚本 = 自动续跑
#
# 机器会无预警硬复位(2026-08-13 那次在 step 850/1150 丢了整条链)。所以每步
# 先看产物在不在,训练则从最新 checkpoint 续。找 checkpoint 用**数字排序**,
# 字典序会选中 checkpoint-500 而不是 1000,**从更早的点续训而且不报错**。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venv-e/bin/python}
MIX=${1:-chat}
R=output/rehearsal.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$R"; }
die() { say "  !! $*"; exit 1; }
latest_ckpt() {
    ls -d "$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 \
        | while read -r n; do [ -n "$n" ] && echo "$1/checkpoint-$n"; done
}
TM=$($PY -c "import sys;sys.path.insert(0,'.');from prepare.tokenizer import resolve_assets;print(resolve_assets()[0])")
CD=$($PY -c "import sys;sys.path.insert(0,'.');from prepare.tokenizer import resolve_assets;print(resolve_assets()[1])")

INIT=$PWD/output/init_summer114m
BASE=$PWD/output/summer114m_base
CHAT=$PWD/output/summer114m_chat
PRE=$PWD/output/rehearsal_pre_1025
POST=$PWD/output/rehearsal_chat_1025

say "=== 预演 mix=$MIX ==="

say "=== 1/6 init depth 10 ==="
if [ -f $INIT/config.json ]; then say "  已有,跳过"; else
  make -C prepare init-scratch INIT_S=$INIT INIT_DEPTH=10 > output/rehearsal_init.log 2>&1
  grep -E "^参数" output/rehearsal_init.log | tee -a "$R"
  [ -f $INIT/config.json ] || die "init 没产出"
fi

say "=== 2/6 编预训练数据(BOS + best-fit,seq 1025,310M)==="
if [ -f $PRE.pt ]; then say "  已有,跳过"; else
  $PY prepare/encode_corpus.py --mix scratch --pack bos_bestfit --seq_length 1025 \
      --total_tokens 310000000 --num_workers 8 \
      --tokenizer_model "$TM" --cn_dict "$CD" --output $PRE.pt \
      > output/rehearsal_encode_pre.log 2>&1
  grep -E "^Total:" output/rehearsal_encode_pre.log | tee -a "$R"
  [ -f $PRE.pt ] || die "预训练数据没产出"
fi

say "=== 3/6 预训练 1150 步 ==="
if [ -f $BASE/config.json ]; then say "  已有最终模型,跳过"; else
  CK=$(latest_ckpt $BASE); [ -n "$CK" ] && say "  从 $(basename $CK) 续"
  # 动量按 nanochat 的 base_train:前 400 步 0.85 → 0.97(chat 阶段才是 0.95/300)。
  # **偏离一处**:它在退火段还会把动量降回 0.90,我们没实现那一半。
  $PY src/train.py \
      --model_path $INIT --train_data $PRE.pt --mode clm \
      --output_dir $BASE ${CK:+--resume_from $CK} \
      --max_seq_length 1025 --batch_size 16 --gradient_accumulation_steps 16 \
      --gradient_checkpointing --compile --param_dtype float32 --ce_chunk 4096 \
      --max_steps 1150 --warmup_steps 50 \
      --lr_schedule wsd --lr_decay_steps 115 --min_lr_ratio 0 \
      --muon_lr 0.02 --adam_lr 3e-4 \
      --muon_momentum 0.97 --muon_momentum_warmup 400 --muon_momentum_start 0.85 \
      --adam_beta1 0.8 --adam_beta2 0.95 --adam_weight_decay 0 --dmodel_lr_scale \
      --val_rows 512 --eval_steps 200 \
      --max_grad_norm 1.0 --seed 42 --num_workers 0 \
      --save_steps 200 --logging_steps 50 --keep_ckpt 3 \
      >> output/rehearsal_base.log 2>&1
  [ -f $BASE/config.json ] || die "预训练没产出,看 output/rehearsal_base.log"
fi
say "  预训练就绪"

say "=== 4/6 编后训练数据(mix=$MIX,seq 1025)==="
if [ -f $POST.pt ]; then say "  已有,跳过"; else
  $PY -m prepare.chat --mix $MIX --output $POST --seq_length 1025 \
      > output/rehearsal_encode_chat.log 2>&1
  grep -E "被监督|写出" output/rehearsal_encode_chat.log | tail -3 | tee -a "$R"
  [ -f $POST.pt ] || die "后训练数据没产出"
fi

say "=== 5/6 单阶段后训练(lr = 预训练的 0.8)==="
if [ -f $CHAT/config.json ]; then say "  已有,跳过"; else
  STEPS=$($PY -c "
import torch
print(max(50, torch.load('$POST.pt', weights_only=True, mmap=True).shape[0] // 64))")
  CK=$(latest_ckpt $CHAT)
  say "  $STEPS 步$([ -n "$CK" ] && echo ",从 $(basename $CK) 续")"
  $PY src/train.py \
      --model_path $BASE --train_data $POST.pt --loss_mask $POST.mask.pt --mode clm \
      --output_dir $CHAT ${CK:+--resume_from $CK} \
      --max_seq_length 1025 --batch_size 16 --gradient_accumulation_steps 4 \
      --gradient_checkpointing --compile --param_dtype float32 --ce_chunk 4096 \
      --max_steps $STEPS --warmup_steps 0 \
      --lr_schedule linear --min_lr_ratio 0 \
      --muon_lr 0.016 --adam_lr 2.4e-4 \
      --muon_momentum 0.95 --muon_momentum_warmup 300 --muon_momentum_start 0.85 \
      --adam_beta1 0.8 --adam_beta2 0.95 --adam_weight_decay 0 --dmodel_lr_scale \
      --val_rows 512 --eval_steps 150 \
      --max_grad_norm 1.0 --seed 42 --num_workers 0 \
      --save_steps 300 --logging_steps 25 --keep_ckpt 2 \
      >> output/rehearsal_chat.log 2>&1
  [ -f $CHAT/config.json ] || die "后训练没产出,看 output/rehearsal_chat.log"
fi
say "  后训练就绪"

say "=== 6/6 评测 ==="
# **主判据是 --render nanochat**(混比里的多选就是用它训的);ours 只为和 v2~v5 对口径
for rd in nanochat ours; do
  $PY prepare/mc_eval.py --model_path $CHAT --tasks arc_easy,ceval --render $rd \
      --output_path eval_results/mc/summer114m_$MIX.$rd.json 2>/dev/null \
      | grep -E --line-buffered "acc " | tee -a "$R"
done
$PY prepare/stoprate.py $CHAT 200 600 2>/dev/null \
    | grep -E --line-buffered "自然停止|长回答|不一致" | tee -a "$R"
say "=== 预演结束(mix=$MIX)==="
