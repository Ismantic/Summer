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
