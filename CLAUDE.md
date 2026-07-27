# CLAUDE.md

给 Claude Code 的项目说明。这些约定优先于默认行为。

## 这是什么

Summer:把 `Qwen/Qwen3-1.7B-Base` 的原生词表(151643 BBPE)换成自训的 piece
词表(81903),再用两阶段继续预训练恢复能力 —— 即 **ReTok**。产物是一个
15.8 亿参数的 base 模型,已发布 `Ismantic/Qwen3-1.7B-Base-ReTok`。

下游是 `../Interpreter` 的自建中英翻译模型。**不做指令微调,不是 chat 模型。**

**目标是独立自主,不是效率。** 这条决定了很多取舍 —— 换词表实际上是**负收益**
的交易(见 `docs/WHY.md` 第三节:中文多用 4.7% token、BLEU −1.88、gsm8k 归零),
换来的是词表小 46%、省 143M 嵌入参数、以及一个完全自己的分词器。
评价这个项目不要按「有没有变强」,要按「是不是自主可控且讲清楚了」。

## 四层结构

```
data/       下载。source.py 是数据源注册表,唯一的真相来源
prepare/    编排:依赖、词表手术、预编码、调训练、评测
src/        模型 + 训练。**只依赖 torch**
save/       导出 HF 发布包 + 上传 + 核对
```

外加 `deps/`(clone 的 C++ 依赖,gitignore)、`docs/`、`test/`。
每层一个 README 讲「这一层解决什么问题」,`make help` 讲怎么跑,
`docs/WHY.md` 讲为什么。**同一件事只写在一处。**

两条分层约束,改代码时不要破坏:

- **`src/` 只依赖 torch。** Qwen3 forward、LoRA、safetensors 读写、
  Muon/Aurora 都是自己实现的。加依赖前先想能不能放 `prepare/`。
- **`src/` 不碰文本。** 分词、字→id 全在 `prepare/`,`src/` 只读预编码好的 id。
  所以 PieceTokenizer 不是 `src/` 的依赖。

## 环境

- **一个 venv 就够:`~/.venv-e`(Python 3.11,torch 2.11+cu13 + vllm + comet
  + lm_eval)。** 训练和评测都在它里面。
  **用 `uv pip install`,这个 venv 里没有 pip。**
- 曾经是两个(3.14 训练 + 3.11 评测),因为 vllm 和 comet 都上不了 3.14。
  **方向搞反了才卡住** —— 往 3.14 装 vllm 不行,但把训练搬到 3.11 完全可以:
  `src/` 只依赖 torch,torch 支持 3.11。合并的验证见 `docs/WHY.md` 第五节。
- 路径写在 gitignore 的 `local.mk`(`PY`,`PY_EVAL` 指同一个)。
  **不要把路径写回 Makefile**,那样别的机器就跑不了。
  **也不要在跟踪的文件里写本机绝对路径** —— 那会把用户名推到 GitHub 上,
  `make test` 里的 `test-noleak` 会拦。
- GPU:单张 RTX 4090(24GB,bf16)。**没有多卡代码路径。**
- C++ 依赖:`make deps` clone 并编译 PieceTokenizer。
  **词表在它仓库的 `save/` 下,本仓库不留副本** ——
  `Summer-Tokenizer.pt` + `Summer-Tokenizer.dict.txt`,用
  `prepare.tokenizer.resolve_assets()` 反查。**产出也用同名** —— 这样从任何
  checkpoint 都能看出词表出自哪里。旧名 `piece.model` / `dict.txt` 在
  `has_piece_vocab()` 里保留为回退(v18 的 ckpt 是旧名)。

## 不能改错的地方

这些改错了**不会报错**,只会悄悄给出错误结果。完整清单见 `docs/WHY.md`
第一节,这里列最要命的:

- **`src/model.py` 的 state_dict key 不能动。** 已发布的权重是标准 HF 命名,
  改模块名会让它全部失配,而模型照样随机初始化跑起来、loss 照样降。
- **RoPE 的 `inv_freq @ position_ids` 必须强制 float32。** 被 autocast 降到
  bf16 的话整数过 256 就不精确(1023→1024),位置编码错乱。实测代价:
  seq 1024 上 loss 从 2.3331 涨到 2.7605。
- **中文分词词典是必需的。** 少了它中文的 token id 会变(不只是慢),而
  round-trip 照样正确 —— 看不出来。`prepare/tokenizer.py` 现在直接报错。
- **tie 的 checkpoint 里没有 `lm_head.weight`。** 310 个张量,加载时绑回 embed。
- **重建 PieceTokenizer 之前先 `python test/capture_baseline.py` 抓基线。**
  顺序反了就失去意义。
- **不要为了让数字好看去改评测代码。**

## 评测:什么样的差异才算数

- **vLLM 的贪心解码不可复现。** 同 ckpt 跑 6 次,BLEU 的 range 是 0.10–0.13。
  **0.1 量级的差是噪声,不能当结论。** COMET 稳两个数量级,比 BLEU 可靠。
- **不能混后端。** 实测同一个 base 模型上 lambada 差 2.2 个点、arc 差 −0.0034
  (方向还不一致)。`eval_results/full/` 的目录名带 `_vllm` / `_tf` 后缀,
  `sweep.py` 只认 `_vllm`,找不到就留空,不去凑 `_tf` 的数。
- **gsm8k 恒在 0.035 附近。** 换词表打断了 Qwen3 的数值 token 化,两个 phase
  都救不回来。**这是词表代价,不是回归**,别去追。

## 测试

```bash
make test          # 五项:equiv / lora / tok / retok / ppl,几分钟
make test-full     # 再加 trans(5 分钟)和 mono(36 分钟),都要 PY_EVAL
```

| | 防什么 |
|---|---|
| `test_model_equiv.py` | 自写 Qwen3 vs transformers,逐层对齐(结构判据) |
| `test_lora.py` | 自写 LoRA vs peft(数学判据) |
| `test_tokenizer.py` | 分词器重建后行为没变(要先有基线) |
| `test_retok.py` | 换词表后的模型自洽 |
| `test_reproduce_sota.py` | 端到端:ppl(自己的 forward)/ trans / mono(vLLM) |

**`trans` 和 `mono` 走 vLLM,测不到 `src/model.py`** —— vLLM 用它自己的实现。
能锚住自写模型的只有 `equiv` 和 `ppl`。

**等价性测试的输入条件本身就是覆盖面。** RoPE 那个 bug 就是被 `equiv` 漏掉的
(它用 256 长度、没开 autocast),靠 `ppl` 抓到的。序列长度、是否 autocast、
batch 大小,任何一个和真实训练不一致都可能放走一整类 bug。

## 提交

commit message 不要带 `Co-Authored-By: Claude ...`(或任何 AI 署名),
这条覆盖 Claude Code 的默认约定。

**删目录之前先确认里面有没有 gitignored 的数据。** `git ls-files` 看不到的
东西才是危险的,先 `du -sh` 每个子目录。改造期间移走的东西都在 `_attic/`
(gitignore),不是删掉了。
