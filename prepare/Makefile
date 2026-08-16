# Summer 编排层:词表手术、预编码、两阶段训练、评测
#
#   make deps       clone + 编译 PieceTokenizer(词表也在它仓库里)
#   make status     每一步的产物在不在
#
#   make retok      词表手术:Qwen3-1.7B-Base → 换成 81903 piece 词表
#   make encode     预编码语料:main 1B token + anneal 200M token
#   make p1         Phase 1:冻结 transformer,只训 embed + lm_head(25 小时)
#   make p2         Phase 2:LoRA tie-safe(v18 SOTA 的配方,5 小时)
#
#   ---- Summer-0.5B:从零预训练,与上面 ReTok 是独立的一条线 ----
#   make init-scratch    随机初始化的起点(Qwen3-0.6B-Base 架构 + 81903 词表)
#   make encode-scratch  预编码 12B token / 中英各半 / 分 shard(3 小时)
#   make s0              长跑约 6 天,每 2000 步自动测一次 WMT22 5-shot BLEU
#   make encode-anneal-mt  编带平行语料的退火集(CPU,可与训练并行)
#   make s1 STEP=40000     换退火集跑最后 4514 步(学习率同时衰到 10%)
#   make encode-scratch3   重做预训练用的语料:EN 单源对齐 nanochat + CN 筛过
#                          简繁/质量,14.6B token,bos_bestfit 打包
#   make s0b               同一个 init 重跑一遍(唯一变量是数据),约 7 天
#
#   24GB 卡上要降 batch:make p1 BATCH=4 ACCUM=64 / make p2 BATCH=4 ACCUM=32
#
#   make bench      mono 六任务(vLLM + lm_eval)
#   make trans      WMT22 翻译 BLEU + COMET
#   make sweep      多个 ckpt 批量评测 + 出对照表
#   make mc-eval    nanochat 的字母 MC 协议 —— 量指令跟随,和上面的似然打分不可比
#
# 全流程:deps → (make -C data download) → retok → encode → p1 → p2
#
# ## 这些配方是从哪来的
#
# v18 是当前 SOTA(WMT22 zh-en BLEU 20.46)。下面 p1 / p2 的超参与
# `_attic/runs/run_v18_p1.sh` 和 `run_v18_p2_tie.sh` 逐项一致 —— 那 56 个
# 一次性启动脚本收编进这里之后就移走了。
#
# ## 没有多卡代码路径
#
# 本机一张 RTX 4090。旧脚本里的 `torchrun --nproc_per_node=2` 和
# `retro_eval --gpus 0 1` 是 2× A6000 时代的遗留,已删。
#
# ## 两个解释器
#
# PY       训练(src/ 只要 torch)
# PY  解释器。训练和评测同一个 venv,所以只有这一个变量。
# 都写在仓库根的 local.mk(gitignore)里。

HERE  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT  := $(HERE)..
-include $(ROOT)/local.mk
PY      ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)

OUT       := $(ROOT)/output
# 基座:注册表里下好了就用本地目录,没下就退回 HF repo id(在线拉)。
# 不要写相对路径 —— `make -C prepare` 会切目录,相对路径会被 transformers
# 当成 repo id,报 HFValidationError。
BASE      ?= $(shell $(PY) -c "import sys;sys.path.insert(0,'$(ROOT)');\
             from data import source as s;d=s.get('qwen_base').dir();\
             print(d if (d/'config.json').exists() else s.get('qwen_base').repo_id)" 2>/dev/null)
INIT      ?= $(OUT)/init_retok
MAIN_PT   ?= $(OUT)/main_1024.pt
ANNEAL_PT ?= $(OUT)/anneal_1024.pt
P1        ?= $(OUT)/phase1
P2        ?= $(OUT)/phase2
CKPT      ?= $(OUT)/phase2_ckpt_v18_tie
TAG       ?= dev

# 每卡 batch 和梯度累积。**默认值就是 v18 的配方**(A6000 48GB 上训的)。
# 24GB 的 4090 装不下 —— loss 里的 logits.float() 要 16×1023×81903×4 = 5.4GB。
# 按等效 batch 不变换算即可:
#     make p1 BATCH=4 ACCUM=64     # 4×64 = 256 = v18 的 16×16
#     make p2 BATCH=4 ACCUM=32     # 4×32 = 128 = v18 的 16×8
BATCH     ?= 16
ACCUM_P1  ?= 16
ACCUM_P2  ?= 8
# 单独给 p1/p2 传 ACCUM 时覆盖上面两个
ifdef ACCUM
ACCUM_P1  := $(ACCUM)
ACCUM_P2  := $(ACCUM)
endif

# 词表从 clone 的 PieceTokenizer 反查,本仓库不留副本
PIECE     := $(shell $(PY) -c "import sys;sys.path.insert(0,'$(ROOT)');\
             from prepare.tokenizer import resolve_assets;print(resolve_assets()[0])" 2>/dev/null)
CN_DICT   := $(shell $(PY) -c "import sys;sys.path.insert(0,'$(ROOT)');\
             from prepare.tokenizer import resolve_assets;print(resolve_assets()[1])" 2>/dev/null)

.PHONY: help status deps retok encode encode-main encode-anneal p1 p2 \
        bench trans sweep

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

status:
	@echo "词表     $(if $(PIECE),$(PIECE),**没装 —— make deps**)"
	@echo "cn dict  $(if $(CN_DICT),$(CN_DICT),—)"
	@for f in "$(INIT)/config.json:换词表后的初始模型" \
	          "$(MAIN_PT):main 预编码 1B token" \
	          "$(ANNEAL_PT):anneal 预编码 200M token" \
	          "$(P1)/config.json:Phase 1 产物" \
	          "$(P2)/config.json:Phase 2 产物"; do \
	  p=$${f%%:*}; d=$${f##*:}; \
	  if [ -e "$$p" ]; then echo "  [有] $$d"; else echo "  [无] $$d  ($$p)"; fi; \
	done

deps:
	bash $(HERE)install_deps.sh

# ---------------------------------------------------------------- 手术

# 不可逆:嵌入映射一改,所有已有 checkpoint 全部失配。
# 顺带写出 frozen_ids.json(一对一映射的行,Phase 1 可冻结)。
retok:
	$(PY) $(HERE)retok.py \
	    --old_model_path $(BASE) \
	    --new_tokenizer_path $(PIECE) \
	    --cn_dict $(CN_DICT) \
	    --output_path $(INIT)

# ---------------------------------------------------------------- 预编码

encode: encode-main encode-anneal

encode-main:
	$(PY) $(HERE)encode_corpus.py --mix main \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(MAIN_PT) --total_tokens 1000000000 --seq_length 1024

encode-anneal:
	$(PY) $(HERE)encode_corpus.py --mix anneal \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(ANNEAL_PT) --total_tokens 200000000 --seq_length 1024

# ---------------------------------------------------------------- 训练

# Phase 1:冻结 transformer,只训 embed_tokens + lm_head。
# 换词表后约 26% 的嵌入行是多对一平均或 fallback 初始化的,先把它们学出来,
# 不动原始 Qwen 权重。1B token / 3815 步。
p1:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT) --train_data $(MAIN_PT) --mode clm \
	    --output_dir $(P1) \
	    --freeze_transformer \
	    --max_seq_length 1024 --batch_size $(BATCH) \
	    --gradient_accumulation_steps $(ACCUM_P1) \
	    --gradient_checkpointing --compile \
	    --max_steps 3815 --warmup_steps 250 \
	    --adam_lr 1e-4 --min_lr_ratio 0.01 --lr_schedule cosine \
	    --max_grad_norm 1.0 --save_steps 1000 --logging_steps 50

# Phase 2:LoRA tie-safe —— transformer 走低秩旁路,embed_tokens 全参训
# (lm_head 因 tie 自动同步)。200M 高质量 token 退火 / 1500 步。
#
# 全参数解冻在 0.6B 时代(v7–v16)是破坏性的,LoRA 才是跑通的配方。
# min_lr_ratio 不能是 0 —— 衰减到 0 是已知的病态。
p2:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(P1) --train_data $(ANNEAL_PT) --mode clm \
	    --output_dir $(P2) \
	    --use_lora --lora_r 16 --lora_alpha 32 --lora_target q_proj,v_proj \
	    --lora_tie_embed_head \
	    --max_seq_length 1024 --batch_size $(BATCH) \
	    --gradient_accumulation_steps $(ACCUM_P2) \
	    --gradient_checkpointing --compile \
	    --max_steps 1500 --warmup_steps 200 \
	    --muon_lr 5e-5 --adam_lr 5e-5 --min_lr_ratio 0.01 --lr_schedule cosine \
	    --max_grad_norm 1.0 --use_aurora --save_steps 500 --logging_steps 50

# ---------------------------------------------------------------- Summer-0.5B
#
# 从零预训练。与上面 ReTok 的 p1/p2 **是两条独立的线**,共用的只有词表、
# 语料池和 src/train.py。
#
#   make init-scratch     造随机初始化的起点(读 Qwen3-0.6B-Base 的 config)
#   make encode-scratch   预编码 12B token,分 shard(约 3 小时,48GB)
#   make s0               长跑(约 6 天),中途每 2000 步自动测 WMT22 5-shot BLEU
#   make s0-resume STEP=N 从 checkpoint-N 续
#
# ## 超参是怎么定的
#
# batch 16 × accum 16 × 1024 = 262,144 token/步。实际编出来 11,558,371 chunk
# (Gutenberg 被 max_line_chars 截断,少了 164M),除以 256 = 45,149 步 ——
# **正好一个 epoch,不重复喂**。
# 4090 实测(fp32 参数 + bf16 autocast + 梯度检查点 + compile)20,200 tok/s,
# 也就是 1.75B token/天 —— 12B 要 6.9 天。
#
# **batch 16 必须配梯度检查点。** 不开的话 batch 只能到 8,而且峰值 21.2GiB,
# 24GB 卡上只剩 2.8G 余量,跑一周碰上碎片化就前功尽弃。开检查点后 17.8GiB
# (fp32 参数再加 2.7G),换来 20% 的吞吐损失,值。
#
# **--param_dtype float32 不能省。** 见 src/train.py 里那条帮助。
# **--ce_chunk 4096 不能省。** 词表 81903 时 logits.float() 是显存大头,
# 不分块的话 batch 16 直接 OOM。
#
# weight_decay 保持 0(与 v18、与 modded-nanogpt 一致)。想开的话注意
# build_optimizer 把 RMSNorm(ndim==1)也归进了 Adam 组 —— 直接加 wd 会把
# norm 权重从 1 往 0 拽,那是有害的,得先给它单开一组。

INIT_S   ?= $(OUT)/init_summer05b
SCRATCH  ?= $(OUT)/scratch_1024.pt
S0       ?= $(OUT)/summer05b_s0
S0_STEPS ?= 45149
S0_BATCH ?= 16
S0_ACCUM ?= 16
# **DataLoader 用 0 个 worker。**
#
# 2026-08-01 第二次挂:worker 被外部 SIGTERM 杀掉,主进程随后被 torch 的
# `_error_if_any_worker_fails()` 硬崩 —— 走不到 train.py 的优雅退出,
# 那一轮的进度全丢(1050 步 / 3.7 小时)。
#
# 数据是 mmap 的,取一条只是一次页缺失 + int32→int64:实测 20,800 tok/s 下
# 每秒才 25 次随机 4KB 读,主进程自己做完全够。多开 worker 换不来吞吐,
# 只多出一条「worker 死了就硬崩」的路径。
S0_WORKERS ?= 0

# 训练时把所有 shard 逗号拼起来喂进去(train.py 会 mmap 它们)
SCRATCH_SHARDS = $(shell ls $(OUT)/scratch_1024.s*.pt 2>/dev/null | paste -sd,)

.PHONY: init-scratch encode-scratch s0 s0-resume

# 缩小架构做流程预演:INIT_DEPTH=10 → 10L/640H/114M(hidden = depth×64)。
# 空 = 原尺寸(28L/1024H/524M)。**小模型只验链路,不定配方** ——
# 词表 81903 固定,depth 10 的嵌入占 46% 而 0.5B 占 16%,不是同一种模型。
INIT_DEPTH ?=
# 训练中途的 BLEU 评测钩子。预演时置空(`S0_EVAL_HOOK=`),省掉每次存档后的评测。
S0_EVAL_HOOK ?= --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S0)/bleu_curve.jsonl'

init-scratch:
	$(PY) $(HERE)init_scratch.py --output_path $(INIT_S) \
	    $(if $(INIT_DEPTH),--depth $(INIT_DEPTH),)

encode-scratch:
	$(PY) $(HERE)encode_corpus.py --mix scratch \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(SCRATCH) --total_tokens 12000000000 --seq_length 1024 \
	    --shard_output

s0:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --train_data "$(SCRATCH_SHARDS)" --mode clm \
	    --output_dir $(S0) \
	    --max_seq_length 1024 --batch_size $(S0_BATCH) \
	    --gradient_accumulation_steps $(S0_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	--save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 10000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S0)/bleu_curve.jsonl'

# 断了之后接着跑:make s0-resume STEP=12000
s0-resume:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --resume_from $(S0)/checkpoint-$(STEP) \
	    --train_data "$(SCRATCH_SHARDS)" --mode clm \
	    --output_dir $(S0) \
	    --max_seq_length 1024 --batch_size $(S0_BATCH) \
	    --gradient_accumulation_steps $(S0_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	--save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 10000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S0)/bleu_curve.jsonl'

# 装成 systemd 用户服务:开机自动从最新 checkpoint 续上。
#
# 跑六天的东西不能指望「有人记得去拉起来」。2026-08-02 那次硬重启,人不在,
# GPU 空转了 3.5 小时。装完还要开 linger,否则没登录时用户服务不启动。
S0_UNIT := $(HOME)/.config/systemd/user/summer-s0.service

s0-service:
	@mkdir -p $(dir $(S0_UNIT))
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=Summer-0.5B S0 pretraining (auto-resume)' \
	  'StartLimitIntervalSec=3600' \
	  'StartLimitBurst=5' \
	  '' \
	  '[Service]' \
	  'Type=simple' \
	  'WorkingDirectory=$(abspath $(ROOT))' \
	  'Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
	  'ExecStart=/usr/bin/env bash $(HERE)s0_autoresume.sh' \
	  'Restart=on-failure' \
	  'RestartSec=120' \
	  '' \
	  '[Install]' \
	  'WantedBy=default.target' \
	  > $(S0_UNIT)
	systemctl --user daemon-reload
	systemctl --user enable summer-s0.service
	@echo
	@echo "已装。开机自启还需要一次:  loginctl enable-linger \$$USER"
	@echo "启动:  systemctl --user start summer-s0"

# 机器状态采样,给下一次硬重启留现场(见 health_sample.sh 的注释)
HEALTH_UNIT := $(HOME)/.config/systemd/user/summer-health.service

health-service:
	@mkdir -p $(dir $(HEALTH_UNIT))
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=Summer machine health sampler' \
	  '' \
	  '[Service]' \
	  'Type=simple' \
	  'WorkingDirectory=$(abspath $(ROOT))' \
	  'ExecStart=/usr/bin/env bash $(HERE)health_sample.sh' \
	  'Restart=always' \
	  'RestartSec=30' \
	  'Nice=10' \
	  '' \
	  '[Install]' \
	  'WantedBy=default.target' \
	  > $(HEALTH_UNIT)
	systemctl --user daemon-reload
	systemctl --user enable --now summer-health.service
	@echo "已装并启动。看:  tail -5 output/health.csv"

# ---------------------------------------------------------------- S0B:重做预训练
#
# **不是续训,是从同一个 init 重新跑一遍 S0**,唯一变量是数据(SCRATCH3_WEIGHTS
# + bos_bestfit 打包,对齐 nanochat)。复用 $(INIT_S)(architecture/词表都没变,
# init_scratch.py 是确定性的),这样「底座好不好」这个问题里只有数据配方在变,
# 别的都跟 S0 一样 —— 这是刻意的对照设计,不是偷懒。
#
# 2026-08-16 决定,原因见 encode_corpus.py 里 SCRATCH3_WEIGHTS 的大段注释:
# 英文单源对齐 nanochat(FineWebEdu),中文筛掉繁体混杂的 CC_CN/Wikipedia_CN,
# 打包方式换成 bos_bestfit(算法对齐 nanochat 的 dataloader,S0 用的是 stream)。
#
# encode-scratch3 实跑(2026-08-17):第一次跑完对账报了中英比例偏 18.1 个点
# ——bos_bestfit 对 FineWebEdu 的实测丢弃率约 51%(nanochat 自述约 35%,
# 我们的更高),sample/10BT 那 14 个文件(这个子集的全部)打包后只剩 4.7B,
# 喂不满 70% 的 10.22B 配额。改 data/source.py 把 FineWebEdu 换到
# sample/100BT 取前 40 个之后重编,实得 14,244,304 chunk(14.59B token,
# en 70.1% / zh 29.9%,在容差内)。
#
# **随后又把 seq_len 从 1024 换成 2048**(同日,训到 step 738 就停了重来,
# 沉没成本可忽略):查了 nanochat 从初始提交起 `max_seq_len` 就是 2048,
# `miniseries.sh`/`scaling_laws.sh` 扫 d4~d26+ 全系列都没覆盖过这个值——
# 2048 不是某个模型的特例,是它的统一选择(唯一例外 `runcpu.sh` 的
# `--max-seq-len=512` 是 CPU 冒烟测试,不是真实训练配置)。既然要对齐,
# 索性把这个也对上。
#
# 显存上可行:`src/model.py` 的注意力是 `F.scaled_dot_product_attention`
# (flash 算法,O(seq) 显存,不是物化 O(seq²) 矩阵),`batch 16→8、accum
# 16→16` 之后每个 micro-batch 还是 8×2048=16384 token,和原来 16×1024
# 一样,4090 24G 装得下。RoPE `max_position_embeddings=32768` 早留够余量。
# 副作用应验了:2048 下实测丢弃率 32%~36%(CN_FineWeb_Edu 41.8% 偏高,
# FineWebEdu 32%~34%、CCI3-HQ 36%、SkyPile 20.7%~34%),和 nanochat 自述的
# 35% 基本吻合,比 1024 下的约 51% 好得多。
#
# encode-scratch3(seq_len=2048)实跑(2026-08-17):7,128,855 chunk = 14.60B
# token,en 70.0% / zh 30.0%,零偏差。/ 128(= batch8×accum16)= 55,694 步,
# **这是实数**。
S0B        ?= $(OUT)/summer05b_s0b
S0B_DATA   ?= $(OUT)/scratch3_2048
S0B_SEQ    ?= 2048
S0B_BATCH  ?= 8
S0B_ACCUM  ?= 16
S0B_STEPS  ?= 55694
S0B_SHARDS  = $(shell ls $(S0B_DATA).s*.pt 2>/dev/null | paste -sd,)

.PHONY: encode-scratch3 s0b s0b-resume s0b-service

encode-scratch3:
	$(PY) $(HERE)encode_corpus.py --mix scratch3 \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(S0B_DATA) --total_tokens 14600000000 --seq_length $(S0B_SEQ) \
	    --pack bos_bestfit --shard_output --num_workers 6 --max_tokens_per_shard 250000000

s0b:
	@test -n "$(S0B_SHARDS)" || { echo "没有 $(S0B_DATA).s*.pt —— 先 make encode-scratch3"; exit 2; }
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --train_data "$(S0B_SHARDS)" --mode clm \
	    --output_dir $(S0B) \
	    --max_seq_length $(S0B_SEQ) --batch_size $(S0B_BATCH) \
	    --gradient_accumulation_steps $(S0B_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0B_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0B_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 10000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S0B)/bleu_curve.jsonl'

s0b-resume:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --resume_from $(S0B)/checkpoint-$(STEP) \
	    --train_data "$(S0B_SHARDS)" --mode clm \
	    --output_dir $(S0B) \
	    --max_seq_length $(S0B_SEQ) --batch_size $(S0B_BATCH) \
	    --gradient_accumulation_steps $(S0B_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0B_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0B_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 10000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S0B)/bleu_curve.jsonl'

# 开机自动续 —— 同 s0-service 的理由(2026-08-02 那次硬重启,GPU 空转 3.5 小时)
S0B_UNIT := $(HOME)/.config/systemd/user/summer-s0b.service

s0b-service:
	@mkdir -p $(dir $(S0B_UNIT))
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=Summer-0.5B S0B pretraining redo (auto-resume)' \
	  'StartLimitIntervalSec=3600' \
	  'StartLimitBurst=5' \
	  '' \
	  '[Service]' \
	  'Type=simple' \
	  'WorkingDirectory=$(abspath $(ROOT))' \
	  'Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
	  'ExecStart=/usr/bin/env bash $(HERE)s0b_autoresume.sh' \
	  'Restart=on-failure' \
	  'RestartSec=120' \
	  '' \
	  '[Install]' \
	  'WantedBy=default.target' \
	  > $(S0B_UNIT)
	systemctl --user daemon-reload
	systemctl --user enable --now summer-s0b.service
	@echo
	@echo "已装并启动。开机自启还需要一次:  loginctl enable-linger \$$USER"

encode-scratch2:
	$(PY) $(HERE)encode_corpus.py --mix scratch2 \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(S2_DATA) --total_tokens 10000000000 --seq_length 1024 \
	    --shard_output --num_workers 6 --max_tokens_per_shard 250000000

# ---------------------------------------------------------------- S2 续训
#
# **在 S0 的 45,149 步之后继续训,不是从头。**
#
# 目标:英文 token 追上 nanochat d20 的 11.2B。我们 13B 里只有一半是英文
# (约 6.5B),所以英文基准上实际是「6.5B 英文模型」对它的「11.2B」——
# MMLU 0.2594(随机)对它 0.3151 就是这么来的。
#
# **WSD 让这件事不用改学习率。** decay_start = max_steps - lr_decay_steps;
# 把 max_steps 从 45,149 拉到 S2_STEPS,decay_start 跟着后移,从 45,149 续上时
# step < decay_start,lr 自动回到稳定段的 0.02。这正是当初选 WSD 而不是 cosine
# 的理由(见 docs/WHY.md 第四点五)—— 总步数在开跑时不可知。
#
# **--reset_data_position 不能省。** 新语料是另一批 shard,不清零的话会从旧的
# 数据偏移开始读,前面一大段直接跳过而且不报错。
S2         ?= $(OUT)/summer05b_s2
S2_DATA    ?= $(OUT)/scratch2_1024
S2_FROM    ?= $(S0)/checkpoint-45000
S2_STEPS   ?= 79000
# **S2 只用纯文本,对话格式不掺进来。**
#
# 一度打算把对话掺进预训练(让 <bos>/<user>/<assistant>/<end> 跟着 10B token
# 一起训),但决定保持阶段职责分明:预训练只学语言,对话格式统一在 midtrain 学。
#
# 依据是实测:midtrain 跑 200 步就把这几个 token 的余弦从 +1.0000 推到 0.985,
# 而 midtrain 现在有 103M token(旧版 34.8M)—— 掺进预训练能多给的曝光有限,
# 换来的却是「预训练数据不纯」这个长期的理解成本。
#
# 对话数据仍然编在 output/chat_pretrain_1024.pt,给 midtrain 用(见下面 MIDTRAIN)。
S2_SHARDS   = $(shell ls $(S2_DATA).s*.pt 2>/dev/null | paste -sd,)

encode-chat-pretrain:
	cd $(ROOT) && $(PY) -m prepare.chat --mix chat_pretrain \
	    --output $(S2_CHAT) --total_tokens 250000000 --seq_length 1024

s2:
	@test -n "$(S2_SHARDS)" || { echo "没有 $(S2_DATA).s*.pt —— 先 make encode-scratch2"; exit 2; }
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --resume_from $(S2_FROM) \
	    --reset_data_position \
	    --train_data "$(S2_SHARDS)" --mode clm \
	    --output_dir $(S2) \
	    --max_seq_length 1024 --batch_size $(S0_BATCH) \
	    --gradient_accumulation_steps $(S0_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S2_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S2_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 --num_workers $(S0_WORKERS) \
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 10000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S2)/bleu_curve.jsonl'

# ---------------------------------------------------------------- midtrain
#
# **让 <user> / <assistant> 不再是死向量。**
#
# 预训练语料全是纯文本,这三个特殊 token 从没作为**输入**出现过。实测 S1 权重
# 里它们和 <pad> 的余弦相似度全是 +1.0000 —— 在模型眼里是同一个向量(普通
# token 之间平均 0.285)。直接上 SFT 等于让模型在一个 epoch 里同时学任务和学
# 这三个分隔符。nanochat 把这件事拆成 midtrain(整段算 loss,喂熟格式)+ SFT
# (只在助手回复上算 loss),这里照它分。
#
# 学习率取 S0 末端的量级再往下一档:这是**继续预训练**不是从头训,大步长会把
# 已有能力冲掉。cosine 衰到 10%,不用 WSD —— 步数是知道的(34.8M / 262144 ≈ 133
# 步太少,所以把 accum 降到 4,变成 532 步)。
MIDTRAIN    ?= $(OUT)/chat_pretrain_1024
MID         ?= $(OUT)/summer05b_mid
MID_FROM    ?= $(OUT)/summer05b_s0
MID_STEPS   ?= 532
MID_BATCH   ?= 16
MID_ACCUM   ?= 4
# 额外参数透传,默认空。主要用途是断点续训:
#   MID_EXTRA="--resume_from output/summer05b_mid4/checkpoint-1400"
# 24GB 卡上 batch 16 的峰值是 23.1G / 24.5G,只剩 1.4G 余量,**碎片化就会 OOM**
# (v4 在 step 1520 挂过一次,报错里 3.94G 是 reserved 但用不上的)。所以跑长任务
# 建议带上 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
MID_EXTRA   ?=

# ---- 对齐 nanochat 的后训练配方(docs/POSTTRAIN.md 的 A 组)----------------
#
# **这些开关在 src/train.py 里默认全关。** 那个脚本同时在跑预训练和 ReTok
# 两条线,而那两条线**已经发布了权重** —— 改默认等于悄悄换掉它们的配方,
# 而且不会报错。所以只在这两个 target 里显式打开。
#
# 想 bisect(哪一项带来了变化)就把对应的变量置空,不用改代码:
#   make midtrain NANOCHAT_OPT=              # 只留调度
#   make midtrain MID_SCHED="--warmup_steps 50 --lr_schedule cosine --min_lr_ratio 0.1"
#
# midtrain 的调度:前 80% 不衰减,后 20% 线性到 0,无 warmup。wsd 的恒定段
# 加 min_lr_ratio 0 正好就是这个形状 —— 不用新写调度。
MID_SCHED   ?= --warmup_steps 0 --lr_schedule wsd --min_lr_ratio 0 \
               --lr_decay_steps $(shell echo $$(( $(MID_STEPS) / 5 )))
# SFT 的调度:线性 1.0 → 0,无 warmup。
SFT_SCHED   ?= --warmup_steps 0 --lr_schedule linear --min_lr_ratio 0
# 优化器:Muon 动量前 300 步 0.85→0.95;AdamW betas (0.8,0.95);AdamW 的 wd
# 硬编码 0(wd 只给 Muon);AdamW 的 lr × (d_model/768)^-0.5 = 0.866。
# **wd 那一项今天是空操作** —— 我们两个阶段的 --weight_decay 本来就是 0,
# 写上是为了以后真给 wd 时不会误伤 AdamW 组。
NANOCHAT_OPT ?= --muon_momentum_warmup 300 --muon_momentum_start 0.85 \
                --adam_beta1 0.8 --adam_beta2 0.95 \
                --adam_weight_decay 0 --dmodel_lr_scale

# 验证集:留出尾部若干行(chat.py 是全局 shuffle 后写盘的,尾部天然分层)。
# **它不是后训练的主判据** —— v4 实测行为侧 +40 点而 loss 侧看不出改进。
# 它防的是训崩和过拟合。主判据是 mc-eval 和停止率。
VAL_ROWS    ?= 512
EVAL_STEPS  ?= 150

midtrain:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(MID_FROM) --train_data $(MIDTRAIN).pt \
	    --loss_mask $(MIDTRAIN).mask.pt --mode clm \
	    --output_dir $(MID) \
	    --max_seq_length 1024 --batch_size $(MID_BATCH) \
	    --gradient_accumulation_steps $(MID_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(MID_STEPS) $(MID_SCHED) \
	    --muon_lr 2e-3 --adam_lr 1e-4 $(NANOCHAT_OPT) \
	    --val_rows $(VAL_ROWS) --eval_steps $(EVAL_STEPS) \
	    --max_grad_norm 1.0 --seed 42 --num_workers 0 \
	    --save_steps 200 --logging_steps 20 --keep_ckpt 3 $(MID_EXTRA)

# SFT:**从 midtrain 的产物起,不是从 S0 起。**
#
# midtrain 已经把对话格式喂熟(特殊 token 的余弦从 1.0000 分化到 0.97),SFT 这一
# 步才专心学「答得好」。跳过 midtrain 直接 SFT 就是 Interpreter 那边踩过的坑:
# 一个 epoch 里同时学任务和学分隔符,两头落空。
#
# 只在助手回复上算 loss(--loss_mask,监督比例 88.3%)—— 用户那段不算,否则模型
# 会去学「怎么生成问题」,那不是要的能力,还会稀释真正的监督信号。
#
# **lr 取 midtrain 的 2%,对齐 nanochat 的 init_lr_frac。** 它 midtrain 跑
# init_lr_frac=1.0(全速)、SFT 跑 0.02,差 50 倍;我们原来只差 4 倍(5e-4/2e-3),
# 高了一个数量级。怀疑这正是 SFT 把 ARC-Easy 从 0.5564 打到 0.4941 的原因 ——
# 步子太大,把 midtrain 学到的东西冲掉了。
#
# **含义是格式和题型要在 midtrain 学完,SFT 只做轻微收尾。** nanochat 就是这么
# 分工的:midtrain 全速学格式,SFT 用 2% 的步长精调答案质量。<end> 也一样 ——
# midtrain 数据里每条回答都以它结尾,不必等到 SFT 才学。
SFT_DATA   ?= $(OUT)/sft_1024
SFT_OUT    ?= $(OUT)/summer05b_sft
SFT_FROM   ?= $(OUT)/summer05b_mid
SFT_STEPS  ?= 366
SFT_BATCH  ?= 16
SFT_ACCUM  ?= 2
SFT_EXTRA  ?=

sft-chat:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(SFT_FROM) --train_data $(SFT_DATA).pt \
	    --loss_mask $(SFT_DATA).mask.pt --mode clm \
	    --output_dir $(SFT_OUT) \
	    --max_seq_length 1024 --batch_size $(SFT_BATCH) \
	    --gradient_accumulation_steps $(SFT_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(SFT_STEPS) $(SFT_SCHED) \
	    --muon_lr 4e-5 --adam_lr 2e-6 $(NANOCHAT_OPT) \
	    --val_rows $(VAL_ROWS) --eval_steps $(EVAL_STEPS) \
	    --max_grad_norm 1.0 --seed 42 --num_workers 0 \
	    --save_steps 150 --logging_steps 20 --keep_ckpt 2 $(SFT_EXTRA)

# ---------------------------------------------------------------- S1 退火
#
# 单语跑到 S0_STEPS - 退火窗口(默认 45149 - 4514 = 40635)之后,换成带中英
# 平行语料的数据集跑最后那 4514 步,学习率同时从 0.02 衰到 0.002。
#
#   make encode-anneal-mt          编 1.2B token 的退火集(CPU,可与训练并行)
#   make s1 STEP=40000             从 S0 的 checkpoint 接着退火
#
# ## 为什么退火要换数据
#
# 2026-07-30 在 S0 的 step 8000(2.5B token)看实际输出:中文已经流利,但
# 5-shot 的例子对模型没有任何约束力 —— 让它把中文译成英文,它输出一段无关的
# 中文。**这是质的缺口,不是量的缺口**,剩下 9B 单语 token 不会改变它。
# 平行文本当普通文档喂进来是唯一的杠杆(TowerBase / ALMA 的路子),模型仍是 base。
#
# ## --reset_data_position 不能省
#
# train_state 里的 consumed 是**旧数据集的索引**。拿 1024 万去切一个 460 万
# chunk 的新数据集,切出来是空的 —— DataLoader 一条都不吐,for 循环直接结束、
# epoch 自增,然后才从头开始。不报错,只是白转一圈还把 epoch 记歪。

ANNEAL_MT ?= $(OUT)/anneal_mt_1024.pt
S1        ?= $(OUT)/summer05b_s1
ANNEAL_MT_SHARDS = $(shell ls $(OUT)/anneal_mt_1024.s*.pt 2>/dev/null | paste -sd,)

.PHONY: encode-anneal-mt s1 s1-resume s1-service

encode-anneal-mt:
	$(PY) $(HERE)encode_corpus.py --mix anneal_mt \
	    --tokenizer_model $(PIECE) --cn_dict $(CN_DICT) \
	    --output $(ANNEAL_MT) --total_tokens 1200000000 --seq_length 1024 \
	    --shard_output --num_workers 6 --max_tokens_per_shard 150000000

s1:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --resume_from $(S0)/checkpoint-$(STEP) \
	    --reset_data_position \
	    --train_data "$(ANNEAL_MT_SHARDS)" --mode clm \
	    --output_dir $(S1) \
	    --max_seq_length 1024 --batch_size $(S0_BATCH) \
	    --gradient_accumulation_steps $(S0_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 2000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S1)/bleu_curve.jsonl'

# 断了之后接着跑 —— 起点在 S1 自己的目录里,不是 S0 的
s1-resume:
	$(PY) $(ROOT)/src/train.py \
	    --model_path $(INIT_S) --resume_from $(S1)/checkpoint-$(STEP) \
	    --train_data "$(ANNEAL_MT_SHARDS)" --mode clm \
	    --output_dir $(S1) \
	    --max_seq_length 1024 --batch_size $(S0_BATCH) \
	    --gradient_accumulation_steps $(S0_ACCUM) \
	    --gradient_checkpointing --compile \
	    --param_dtype float32 --ce_chunk 4096 \
	    --max_steps $(S0_STEPS) --warmup_steps 2000 \
	    --lr_schedule wsd --lr_decay_steps $$(( $(S0_STEPS) / 10 )) \
	    --muon_lr 0.02 --adam_lr 3e-4 --min_lr_ratio 0.1 \
	    --max_grad_norm 1.0 --seed 42 \
	    --num_workers $(S0_WORKERS) \
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 2000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S1)/bleu_curve.jsonl'

# 同 s0-service,但只跑退火那 4514 步。装之前记得先停掉 summer-s0,
# 否则重启后两个服务会一起起来抢 GPU。
S1_UNIT := $(HOME)/.config/systemd/user/summer-s1.service
S1_FROM ?= 40000

s1-service:
	@mkdir -p $(dir $(S1_UNIT))
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=Summer-0.5B S1 anneal (auto-resume)' \
	  'StartLimitIntervalSec=3600' \
	  'StartLimitBurst=5' \
	  '' \
	  '[Service]' \
	  'Type=simple' \
	  'WorkingDirectory=$(abspath $(ROOT))' \
	  'Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
	  'Environment=S1_FROM=$(S1_FROM)' \
	  'ExecStart=/usr/bin/env bash $(HERE)s1_autoresume.sh' \
	  'Restart=on-failure' \
	  'RestartSec=120' \
	  '' \
	  '[Install]' \
	  'WantedBy=default.target' \
	  > $(S1_UNIT)
	systemctl --user daemon-reload
	systemctl --user enable summer-s1.service
	@echo "已装。启动:  systemctl --user start summer-s1"

# ---------------------------------------------------------------- 评测

bench:
	$(PY) $(HERE)benchmark.py --model_path $(CKPT) \
	    --tasks "lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5" \
	    --output_dir $(ROOT)/eval_results/full/$(TAG)_vllm

# ---------------------------------------------------------------- nanochat 口径
#
# **与 nanochat 的 report card 对齐**,这样 Summer-0.5B 有个同量级的外部靶子:
# 它的 d20 是 560M 参数 / 约 11B token,我们是 524M / 13B —— 同一档。
#
#   nanochat d20  base: CORE 0.2219
#                 SFT 后: ARC-E 0.3876  ARC-C 0.2807  MMLU 0.3151
#                        GSM8K 0.0455  HumanEval 0.0854  ChatCORE 0.0884
#
# few-shot 数按 nanochat:多选题 0-shot(它的 base 评测走 CORE 的 continuation
# 口径),GSM8K 用 8-shot(它 SFT 后是 0-shot,但我们的底座没经过指令微调,
# 0-shot 下连格式都不会,给 8 个示例才测得出能力而不是格式)。
#
# **上面那个 bench 的 shot 数不动** —— eval_results/full/ 里 v18 的历史数字是
# 按那套跑的,改了就没法和自己的过去比。两套并存,各自内部可比。
#
# 中文那两项(ceval / cmmlu)是 nanochat 没有的:它是纯英文模型,我们是中英
# 50:50。只报英文会把一半能力藏起来。
# piqa 和 cmmlu **不在这套里**:它们在 HF 上是老式的 loading script 数据集,
# 新版 datasets 已经不支持("Dataset scripts are no longer supported")。为了凑
# 这两个指标去降级 datasets 不值得 —— 上面 `make bench` 那套里还有 piqa,
# 那是历史缓存能读到的。
# **只留关键的五项。** 0.5B 这个量级上,hellaswag / winogrande / boolq /
# openbookqa / humaneval 基本都贴着随机线,跑它们只是把噪声记下来。留下的是:
#   ARC-E / ARC-C  常识推理,nanochat 报的主项,难度阶梯清楚
#   MMLU           知识面,nanochat 报的主项
#   GSM8K          数学,能看出 midtrain 掺 GSM8K 有没有用
#   ceval-valid    中文 —— nanochat 没有,但我们是中英 50:50,只报英文会藏掉一半
BENCH_NC_TASKS ?= arc_easy:0,arc_challenge:0,mmlu:0,gsm8k:8,ceval-valid:0

bench-nanochat:
	$(PY) $(HERE)benchmark.py --model_path $(CKPT) \
	    --tasks "$(BENCH_NC_TASKS)" \
	    --output_dir $(ROOT)/eval_results/nanochat/$(TAG)

# nanochat 的**字母 MC** 协议:题面渲染成 Multiple Choice、模型输出一个字母。
#
# **和 bench-nanochat 量的不是一回事,两个都要留。** lm_eval 的 acc/acc_norm 是
# 比各选项正文的对数似然,量底座知识;字母 MC 要求模型听懂「只回答字母」,量的是
# 指令跟随 —— 那正是 midtrain + SFT 要加的能力。同一个 S0、同一批 ARC-Easy 题,
# 似然协议 0.5564、字母协议 0.2563,差 0.30。**别把两边的数混着比。**
#
# 协议实现校验过:拿 nanochat 官方 d20 跑同一套打分得 0.3994,官方 meta 是
# 0.4033,差 -0.0039。校验脚本见 docs/reports/summer-0.5b-pretrain.md 的说明。
MC_TASKS ?= arc_easy,arc_challenge,mmlu,ceval

mc-eval:
	$(PY) $(HERE)mc_eval.py --model_path $(CKPT) --tasks "$(MC_TASKS)" 	    --output_path $(ROOT)/eval_results/mc/$(TAG).json

# 把 eval_results/nanochat/ 下的结果排成一张对照表(含 nanochat d20 的数)
bench-card:
	$(PY) $(HERE)report_card.py

trans:
	$(PY) $(HERE)translate.py --model_path $(CKPT) \
	    --testset wmt22 --exemplar_set wmt21 --direction both \
	    --num_fewshot 5 --max_samples 1000 --compute_comet --save_all_samples \
	    --output_path $(ROOT)/eval_results/translate_wmt22/$(TAG)_vllm_1000_comet.json

# 多 ckpt:make sweep CKPTS="base=/path/a v18=output/phase2_ckpt_v18_tie"
sweep:
	$(PY) $(HERE)sweep.py run --python $(PY) --ckpt $(CKPTS)
