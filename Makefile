# Summer —— 对 Qwen3-1.7B-Base 做 ReTok,产出自建的中英双语底座
#
#   make help          这份说明
#   make status        四层各自的就绪情况
#   make deps          clone + 编译 PieceTokenizer
#   make test          回归防线(改了 src/ 或 prepare/ 之后必跑)
#
# 重建 PieceTokenizer 之前先 `python test/capture_baseline.py` 抓基线 ——
# 顺序反了就失去意义,基线是用来发现「重建把行为改了」的。
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
#   PY  解释器。训练和评测同一个 venv,所以只有这一个变量。

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
-include $(HERE)local.mk
PY      ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)

.PHONY: help status deps test test-equiv test-lora test-tok test-retok test-init \
        test-ppl test-full clean-pyc

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
#   noleak 被跟踪的文件里没有本机绝对路径(防的是不可逆的信息泄露)
#   equiv  自写 Qwen3 vs transformers,逐层对齐(结构判据)
#   lora   自写 LoRA vs peft(数学判据)
#   ppl    固定切片 next-token loss(端到端锚点)
#   init   从零初始化的统计量(**上面每一条都是「加载权重之后」比对,
#          所以 2026-07-28 之前「根本没有初始化代码」这件事没有一条拦得住**)
test: test-noleak test-equiv test-lora test-tok test-retok test-init test-ppl

# 被跟踪的文件里不许有本机绝对路径(会把用户名推到 GitHub / HF 上,
# 而且 git 历史里删不掉)。放在最前面 —— 它几毫秒,而且拦的是不可逆的事故。
test-noleak:
	$(PY) $(HERE)test/test_no_local_paths.py

# 分词器行为没变(需要先有基线,见 test/capture_baseline.py)
test-tok:
	$(PY) $(HERE)test/test_tokenizer.py

# 换词表后的模型自洽:特殊 token、round-trip、forward 数值
test-retok:
	$(PY) $(HERE)test/test_retok.py

test-equiv:
	$(PY) $(HERE)test/test_model_equiv.py

test-lora:
	$(PY) $(HERE)test/test_lora.py

# 从零预训练的起点:嵌入/残差出口的 std、RMSNorm 全 1、初始 loss ≈ ln(V)
test-init:
	$(PY) $(HERE)test/test_init_scratch.py

test-ppl:
	$(PY) $(HERE)test/test_reproduce_sota.py --only ppl

test-full: test
	$(PY) $(HERE)test/test_reproduce_sota.py --only ppl
	$(PY) $(HERE)test/test_reproduce_sota.py --only trans --python $(PY)
	$(PY) $(HERE)test/test_reproduce_sota.py --only mono  --python $(PY)

clean-pyc:
	find $(HERE) -name __pycache__ -type d -not -path "*/_attic/*" -exec rm -rf {} + 2>/dev/null || true
