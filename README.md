# Summer

自建分词器的中英双语底座模型。把 `Qwen/Qwen3-1.7B-Base` 的原生词表(151643
BBPE)换成自己训的 piece 词表(81903),再用继续预训练把能力恢复回来 ——
这类工作叫 **ReTok**([arXiv:2410.04335](https://arxiv.org/abs/2410.04335))。

产物是一个 **15.8 亿参数**的 base 模型,规模上与 GPT-2 XL 相当,但用的是
2025 年的架构。已发布:
[`Ismantic/Qwen3-1.7B-Base-ReTok`](https://huggingface.co/Ismantic/Qwen3-1.7B-Base-ReTok)。

下游是 [`Interpreter`](https://github.com/Ismantic/Interpreter) 的自建中英
翻译模型。这里不做指令微调,不是 chat 模型。

## 这是在换什么

|  | Qwen3-1.7B-Base | 本项目 |
|---|---|---|
| 词表 | 151643(BBPE) | **81903**(piece,自训) |
| 参数量 | 1,720,574,976 | **1,577,147,392** |
| 其中嵌入 | 310.6M | **167.7M** |
| 架构 | 28L / 2048H / GQA 16:8 / RoPE / SwiGLU | 不变 |

对照:GPT-2 XL 是 1,557,611,200 参数 —— 相差 1.25%,基本同规模。
但形状完全不同,GPT-2 XL 是 48 层 ×1600 深而窄。

## 换到了什么,又付了什么

**这不是一次「换个词表就变强」的改进。** 完整的账:

| | |
|---|---|
| 代价 | WMT22 zh-en BLEU −1.88、en-zh −2.31 |
| 代价 | gsm8k 从正常水平掉到 ~0.035,两个阶段都救不回来 |
| 代价 | 中文**多用 4.7% token** —— 推理更贵,不是更省 |
| 收益 | 词表小 46%,嵌入参数省 142.9M |
| 收益 | 分词器和底座完全自有,不依赖 Qwen 的词表 |

ReTok 常见的动机是「换词表提高压缩率」。这个项目**没换到,反而略降** ——
Qwen 的 151K 词表里中文多词合并更多(`在北京` 是一个 token,piece 拆成两个)。
真正换到的是:用一半的词表槽位做到接近的压缩,省下 143M 嵌入参数,
以及一个完全自己的分词器。

**目标是独立自主,不是效率。** 这些数字和推导过程见
[`docs/WHY.md`](docs/WHY.md)。

## 表现

WMT22,1000 样本 5-shot,vLLM 后端:

| | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 22.34 | 0.8122 | 38.34 | 0.8597 |
| Phase 1(冻结 transformer) | 20.26 | 0.7821 | 35.16 | 0.8276 |
| **Phase 2(LoRA tie-safe)** | **20.46** | **0.7933** | **36.03** | **0.8444** |

通用 benchmark:

| | LAMBADA | PIQA | ARC-C | HellaSwag | CEVAL | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 0.6513 | 0.7731 | 0.5512 | 0.6705 | 0.6560 | 0.6710 |
| Phase 1 | 0.5674 | 0.7301 | 0.5137 | 0.6375 | 0.6263 | 0.0341 |
| **Phase 2** | 0.5768 | 0.7367 | 0.5145 | 0.6389 | 0.6204 | 0.0349 |

**vLLM 的贪心解码不可复现**:同一 ckpt 跑 6 次,BLEU 的 range 是 0.10–0.13。
所以 0.1 量级的差是噪声,不是结论;COMET 稳两个数量级,比 BLEU 可靠。
两个后端的数字也不能混着比 —— 实测 lambada 上差 2.2 个点。详见
[`docs/WHY.md`](docs/WHY.md)。

## 怎么跑

```bash
make deps                  # clone + 编译 PieceTokenizer(词表也在它仓库里)
make -C data probe         # 验数据源注册表写对没(每源只下一个文件)
make -C data download      # 下全部语料(100GB+,只有要重建语料时才需要)

make -C prepare retok      # 词表手术:Qwen3-1.7B-Base → 81903 piece 词表
make -C prepare encode     # 预编码:main 1B token + anneal 200M token
make -C prepare p1         # Phase 1:冻结 transformer,只训 embed + lm_head
make -C prepare p2         # Phase 2:LoRA tie-safe

make test                  # 回归防线
```

`make help` 和 `make -C <层> help` 有更细的说明。

## 四层结构

```
data/       下载 + 加工。source.py 是数据源注册表,唯一的真相来源
prepare/    编排:依赖、词表手术、预编码、调训练、评测
src/        模型 + 训练。**只依赖 torch**
save/       导出 HF 发布包 + 上传
```

外加 `deps/`(clone 的 C++ 依赖,gitignore)、`docs/`、`test/`。

两条分层约束,改代码时不要破坏:

- **`src/` 只依赖 torch。** Qwen3 的 forward、LoRA、safetensors 读写、
  Muon/Aurora 都是自己实现的(替掉 transformers、peft、safetensors 库)。
  这条的意义是让读者**看得见每一步在做什么**。
- **`src/` 不碰文本。** 分词、字→id 全在 `prepare/`,`src/` 只读预编码好的
  id。所以 PieceTokenizer 不是 `src/` 的依赖。

自写实现的验证:与 transformers 逐层对齐 float32 最大 **1.66e-6**、
argmax **100% 一致**;与 peft 对拍 logits **逐位相同**。

## 环境

- 训练:Python 3.14 + torch 2.11,单张 RTX 4090(24GB,bf16)。
  **没有多卡代码路径。**
- 评测:需要 vllm + comet,它们上不了 3.14 —— 得单开一个 3.11 的 venv。
  两个解释器路径写在 gitignore 的 `local.mk` 里(`PY` / `PY_EVAL`)。
  为什么合并不了见 [`docs/WHY.md`](docs/WHY.md)。
- C++ 依赖:[PieceTokenizer](https://github.com/Ismantic/PieceTokenizer),
  `make deps` 自动 clone 并编译。**81903 词表和中文分词词典都在它仓库里,
  本仓库不留副本。**

装依赖:

```bash
uv pip install -r requirements.txt                    # 训练侧(3.14)
uv pip install -r requirements-eval.txt               # 评测侧(单独的 3.11 venv)
```

语料、词表、基座权重全部从 Hugging Face 和 GitHub 获取,不需要任何本机既有
文件。唯一的手动步骤:`BAAI/CCI3-HQ` 是 gated 数据集,要先在 HF 页面接受条款
并 `huggingface-cli login`。

## 文档

| | |
|---|---|
| [`docs/WHY.md`](docs/WHY.md) | 各处设计选择的理由,以及**改错了不报错**的地方 |
| [`docs/PRETRAIN.md`](docs/PRETRAIN.md) | 从零重做一遍的全流程 |
| [`docs/eval/pipeline.md`](docs/eval/pipeline.md) | 评测栈:三个入口、数字怎么读 |
| [`data/source.py`](data/source.py) | 数据源注册表 + 逐个溯源的依据 |

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
