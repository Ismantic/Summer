# Summer —— 对 Qwen3-1.7B-Base 做 ReTok,产出自建的中英双语底座
#
#   make help          这份说明
#   make status        四层各自的就绪情况
#   make deps          clone + 编译 PieceTokenizer
#   make test          回归防线(改了 src/ 或 prepare/ 之后必跑)
#
# 分层跑,每层自己的 Makefile 里有更细的 target:
#
#   make -C data       下载语料与评测资产
#   make -C prepare    词表手术、预编码、调训练、评测
#
# 本机的解释器路径写在 local.mk(gitignore),不要写回 Makefile ——
# 那样别的机器就跑不了。需要两个:
#
#   PY       训练 / src(只要 torch)
#   PY_EVAL  评测(vllm + comet,上不了 Python 3.14,得单开一个 3.11 的 venv)

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
-include $(HERE)local.mk
PY      ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)
PY_EVAL ?= $(PY)

.PHONY: help status deps test test-equiv test-lora test-ppl test-full clean-pyc

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

status:
	@echo "== data/ =="   && $(PY) $(HERE)data/download.py --list
	@echo && echo "== deps/ ==" && \
	  ($(PY) -c "import piece_tokenizer,os;print('  piece_tokenizer',os.path.dirname(piece_tokenizer.__file__))" \
	   2>/dev/null || echo "  piece_tokenizer 没装 —— make deps")

deps:
	bash $(HERE)prepare/install_deps.sh

# ---------------------------------------------------------------- 回归防线
#
# 三项各防一段链路,不能互相替代:
#   ppl    走自己的 forward —— 唯一能锚住 src/model.py 的(阶段 3 之后尤其重要)
#   trans  走 vLLM —— 词表/数据/导出链路;BLEU 是这个项目的头号指标
#   mono   走 vLLM + lm_eval —— 六个 benchmark 的整体退化
#
# ppl 几秒,trans 约 5 分钟,mono 约 36 分钟。日常改完跑 test-ppl 就够。

# 阶段 3 之后必跑的三件套:
#   equiv  自写 Qwen3 vs transformers,逐层对齐(结构判据)
#   lora   自写 LoRA vs peft(数学判据)
#   ppl    固定切片 next-token loss(端到端锚点)
test: test-equiv test-lora test-ppl

test-equiv:
	$(PY) $(HERE)test/test_model_equiv.py

test-lora:
	$(PY) $(HERE)test/test_lora.py

test-ppl:
	$(PY) $(HERE)test/test_reproduce_sota.py --only ppl

test-full: test
	$(PY) $(HERE)test/test_reproduce_sota.py --only ppl
	$(PY) $(HERE)test/test_reproduce_sota.py --only trans --python $(PY_EVAL)
	$(PY) $(HERE)test/test_reproduce_sota.py --only mono  --python $(PY_EVAL)

clean-pyc:
	find $(HERE) -name __pycache__ -type d -not -path "*/_attic/*" -exec rm -rf {} + 2>/dev/null || true
