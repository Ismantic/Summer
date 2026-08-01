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
#
#   24GB 卡上要降 batch:make p1 BATCH=4 ACCUM=64 / make p2 BATCH=4 ACCUM=32
#
#   make bench      mono 六任务(vLLM + lm_eval)
#   make trans      WMT22 翻译 BLEU + COMET
#   make sweep      多个 ckpt 批量评测 + 出对照表
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

init-scratch:
	$(PY) $(HERE)init_scratch.py --output_path $(INIT_S)

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

.PHONY: encode-anneal-mt s1

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
	    --save_steps 1000 --logging_steps 50 --keep_ckpt 3 --ckpt_milestone 2000 \
	    --inline_eval_cmd '$(PY) $(HERE)eval_hook.py --ckpt {ckpt} --step {step} --curve $(S1)/bleu_curve.jsonl'

# ---------------------------------------------------------------- 评测

bench:
	$(PY) $(HERE)benchmark.py --model_path $(CKPT) \
	    --tasks "lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5" \
	    --output_dir $(ROOT)/eval_results/full/$(TAG)_vllm

trans:
	$(PY) $(HERE)translate.py --model_path $(CKPT) \
	    --testset wmt22 --exemplar_set wmt21 --direction both \
	    --num_fewshot 5 --max_samples 1000 --compute_comet --save_all_samples \
	    --output_path $(ROOT)/eval_results/translate_wmt22/$(TAG)_vllm_1000_comet.json

# 多 ckpt:make sweep CKPTS="base=/path/a v18=output/phase2_ckpt_v18_tie"
sweep:
	$(PY) $(HERE)sweep.py run --python $(PY) --ckpt $(CKPTS)
