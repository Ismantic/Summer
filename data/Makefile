# Summer 数据获取
#
#   make status        每个源下了没
#   make probe         **每源只下第一个文件并真读一遍** —— 验注册表写对没
#   make download      下全部(预训练语料 + COMET 模型,100GB+)
#   make pretrain      只下预训练语料
#   make eval-assets   只下评测要的(COMET 模型 + 六个 benchmark 数据集)
#   make dry           只看要下什么,不动手
#
# 改了 source.py 之后跑 `make probe`,不要跑 `make download`:会写错的是
# glob / fmt / text_field 这三样,一个文件就能暴露,不需要下满。全量只有真要
# 重建语料时才需要。
#
# 单个源:make download-CC_EN / download-Gutenberg …(源名见 make status)
#
# 落地位置由 SUMMER_DATA_ROOT 控制,默认 data/downloads/(gitignore)。
# 本机若已有语料,用符号链接指过去,**不要把路径写回代码**:
#
#   ln -s /path/to/existing/fineweb data/downloads/fineweb
#
# 下载走 hf-mirror(HF_ENDPOINT),download.py 内部会清代理 —— hf-mirror 走不通
# 代理。走官方源:make download HF_ENDPOINT=
#
# 数据源的登记在 source.py,那是唯一的真相来源。每个源都必须有公开出处;
# 溯源依据(哪些是 sha256 确证、哪些是替换)写在那个文件的开头。

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
-include $(HERE)../local.mk
PY ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)

.PHONY: help status probe download pretrain eval-assets dry

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

status:
	@$(PY) $(HERE)download.py --list

probe:
	$(PY) $(HERE)download.py --probe

download:
	$(PY) $(HERE)download.py --all

pretrain:
	$(PY) $(HERE)download.py --pretrain

# COMET 打分模型 + lm-eval 那六个 benchmark 的数据集
eval-assets:
	$(PY) $(HERE)download.py comet
	$(PY) $(HERE)prefetch_eval_datasets.py

dry:
	$(PY) $(HERE)download.py --all --dry

# 单个源:make download-SkyPile
download-%:
	$(PY) $(HERE)download.py $*
