# Repository Guidelines

面向自动化 agent 的简版说明。完整版见 `CLAUDE.md`,设计理由见 `docs/WHY.md`。

## 结构

四层,按数据流切:

```
data/       下载。source.py 是数据源注册表,唯一的真相来源
prepare/    编排:依赖、词表手术、预编码、调训练、评测
src/        模型 + 训练。**只依赖 torch**
save/       导出 HF 发布包 + 上传 + 核对
```

外加 `deps/`(gitignore)、`docs/`、`test/`、`papers/`。产物落 `output/` 和
`eval_results/`,都 gitignore。

两条不能破坏的分层约束:**`src/` 只依赖 torch**;**`src/` 不碰文本**
(分词在 `prepare/`,`src/` 只读预编码好的 id)。

## 命令

每层一个 Makefile,`make help` 有说明。

```bash
make deps                  # clone + 编译 PieceTokenizer
make -C data probe         # 验数据源注册表(每源只下一个文件)
make -C prepare retok      # 词表手术(不可逆)
make -C prepare encode     # 预编码
make -C prepare p1 / p2    # 两阶段训练
make test                  # 回归防线
```

## 代码风格

4 空格缩进,函数和文件名 snake_case,共享常量 UPPER_CASE。新入口用 `argparse`。
**机器相关的路径不要写进代码** —— 数据走 `data/source.py` 注册表,解释器路径
走 gitignore 的 `local.mk`,词表用 `prepare.tokenizer.resolve_assets()` 反查。

不要再新增 `run_vNN.sh` 这类一次性脚本 —— 配方进 `prepare/Makefile`。

## 测试

改了 `src/` 或 `prepare/` 之后跑 `make test`(五项,几分钟)。涉及评测口径的
改动再跑 `make test-full`(加 trans 5 分钟 + mono 36 分钟,要 `PY_EVAL`)。

**`trans` 和 `mono` 走 vLLM,测不到 `src/model.py`** —— 能锚住自写模型的只有
`test_model_equiv.py` 和 `--only ppl`。

判断数字变化是否算数之前先看 `docs/WHY.md` 第二节:vLLM 贪心解码不可复现,
BLEU 跑间 range 约 0.1;两个后端的数字不能混比。

## 提交

commit message 不要带 `Co-Authored-By: Claude ...` 或任何 AI 署名。

删东西之前先 `du -sh` 看有没有 gitignored 的数据 —— `git ls-files` 看不到的
才是危险的。
