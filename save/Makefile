# Summer 发布层:导出 HF 包、上传、核对
#
#   make export     从 SOTA checkpoint 导出干净的 HF 上传目录
#   make releases   列已发布的东西
#   make verify     与 HF 线上核对 sha256(登记值 / 线上 / 本地 三方比对)
#   make demo       交互式续写(base 模型,不是 chat)
#
#   make upload-dry     先看要传什么(**默认先跑这个**)
#   make upload-code    只传代码和模型卡,不碰 3.0GB 权重
#   make upload         全部(含权重)
#
# 发布是不可逆的 —— 别人可能已经下走了。所以 releases.py 记着每一项的
# sha256,verify 用来确认本地和线上没有一边被动过。

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(HERE)..
-include $(ROOT)/local.mk
PY ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)

CKPT ?= $(ROOT)/output/phase2_ckpt_v18_tie
OUT  ?= $(HERE)releases/Qwen3-1.7B-Base-ReTok

.PHONY: help export releases verify demo upload-dry upload-code upload

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

export:
	$(PY) $(HERE)export.py --source $(CKPT) --out $(OUT)

releases:
	@$(PY) $(HERE)releases.py

verify:
	$(PY) $(HERE)releases.py --verify

demo:
	$(PY) -m save.generate

# 权重没变时用 upload-code —— 重传 3.0GB 纯属浪费,而且覆盖已发布权重
# 本身就是不必要的风险(别人可能已经下走了)。
# 先 `make verify` 确认权重与线上一致,再决定用哪个。
upload-dry:
	$(PY) -m save.upload --dry-run

upload-code:
	$(PY) -m save.upload --code-only

upload:
	$(PY) -m save.upload
