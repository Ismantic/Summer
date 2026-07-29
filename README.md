# Summer

中英双语 GPT-2 1.5B 级别底座模型，自建分词器，纯 PyTorch 实现。

把 `Qwen3-1.7B-Base` 的原生词表(151643 BBPE)换成自己训的 Piece 词表
(81903)，再用两阶段继续预训练把能力恢复回来 —— 这类工作叫 **ReTok**
([arXiv:2410.04335](https://arxiv.org/abs/2410.04335))。包含从下载语料到发布上
Hugging Face 的完整流程:词表手术、预编码、两阶段预训练、评测、导出发布。
模型与训练代码(`src/`)只依赖 torch —— Qwen3 的 forward、LoRA、Muon/Aurora、
safetensors 读写都是独立实现，方便理解。

产物是一个 **15.8 亿参数**的 Base 模型,规模上与 GPT-2 XL 相当。
**不做指令微调,不是 Chat 模型** —— 下游是
[`Interpreter`](https://github.com/Ismantic/Interpreter) 的自建中英翻译模型。

取名 Summer ，源于三年前 LLama-2 出来的时候不支持中文，做过类似的工作，
三年后的今年夏天，旧事重提，再把这个项目做一次。

```python
import torch
from model import Qwen3ForCausalLM          # 自写的 Qwen3,只依赖 torch
from tokenizer import PieceTokenizerWrapper

tok = PieceTokenizerWrapper(".")
model = Qwen3ForCausalLM.from_pretrained(".", device="cuda", dtype=torch.bfloat16)
ids = tok.encode("机器翻译的基本任务是")
logits = model(torch.tensor([ids], device="cuda"))
```

贪心续写的完整循环在发布包的 `example_load.py` 里。**这是 Base 模型,只会续写,
不会对话** ——「█」之后是它接出来的,各截到第一个句号:

```
机器翻译的基本任务是█将一种语言的文本翻译成另一种语言的文本。
中文分词的目标是█将一个句子分割成若干个词语，这些词语是句子的基本单位。
深度学习之所以有效,是因为█它能从大量的数据中学习到有用的特征,而这些特征是人类无法通过传统的方法发现的。
```

## 模型

| Hugging Face | 参数 | 词表 | 阶段 |
|---|---|---|---|
| [`Ismantic/Qwen3-1.7B-Base-ReTok`](https://huggingface.co/Ismantic/Qwen3-1.7B-Base-ReTok) | 1,577,147,392 | 81903 | Phase 2(LoRA tie-safe) |

```bash
huggingface-cli download Ismantic/Qwen3-1.7B-Base-ReTok --local-dir ReTok
pip install git+https://github.com/Ismantic/PieceTokenizer
cd ReTok && python example_load.py     # 自己的 forward,只要 torch
cd ReTok && python example_vllm.py     # vLLM 后端
```

发布包自带推理代码和两个示例,除 PyTorch 和 PieceTokenizer 外无其他依赖 ——
**不需要 transformers,也不需要 safetensors 库**。

**分词器不是标准格式**,`AutoTokenizer` 走不通,所以 HF 在线推理和 `vllm serve`
开箱用不了。vLLM 能加载权重(它用自己那份 Qwen3 实现),但要
`skip_tokenizer_init=True` 并自己编码 id,见 `example_vllm.py`。

## 表现

WMT22,1000 样本 5-shot,vLLM 后端:

| | zh-en BLEU | zh-en COMET | en-zh BLEU | en-zh COMET |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 22.34 | 0.8122 | 38.34 | 0.8597 |
| Phase 1(冻结 transformer) | 20.26 | 0.7821 | 35.16 | 0.8276 |
| **Phase 2(LoRA tie-safe)** | **20.46** | **0.7933** | **36.03** | **0.8444** |

| | LAMBADA | PIQA | ARC-C | HellaSwag | CEVAL | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 0.6513 | 0.7731 | 0.5512 | 0.6705 | 0.6560 | 0.6710 |
| Phase 1 | 0.5674 | 0.7301 | 0.5137 | 0.6375 | 0.6263 | 0.0341 |
| **Phase 2** | 0.5768 | 0.7367 | 0.5145 | 0.6389 | 0.6204 | 0.0349 |

指标:LAMBADA / CEVAL 用 `acc`,PIQA / ARC-C / HellaSwag 用 `acc_norm`,
GSM8K 用 `exact_match,strict-match`;shots 依次 0 / 5 / 25 / 10 / 5 / 5。
**gsm8k 两个指标差得不小**(Phase 1 的 flexible-extract 是 0.0417),所以
必须写明用的哪个 —— 口径以 [`prepare/sweep.py`](prepare/sweep.py) 的
`PREFER` 为准。

**vLLM 的贪心解码不可复现** —— 同一 ckpt 跑 6 次,BLEU 的 range 是 0.10–0.13。
0.1 量级的差是噪声不是结论;COMET 稳两个数量级,更可靠。两个后端的数字也不能
混着比(实测 lambada 上差 2.2 个点)。口径见
[`docs/WHY.md`](docs/WHY.md#二评测什么样的差异才算数)。

## 换到了什么

**这不是一次「换个词表就变强」的改进。** 完整的账:

| | |
|---|---|
| 代价 | WMT22 zh-en BLEU −1.88、en-zh −2.31 |
| 代价 | gsm8k 从正常水平掉到 ~0.035,两个阶段都救不回来 |
| 代价 | 中文**多用 4.7% token** —— 推理更贵,不是更省 |
| 收益 | 词表 151643 → 81903,小 46%;嵌入参数省 142.9M |
| 收益 | 分词器和底座完全自有,不依赖 Qwen 的词表 |

ReTok 常见的动机是「换词表提高压缩率」,这个项目**没换到,反而略降** ——
Qwen 的 151K 词表里中文多词合并更多(`在北京` 是一个 token,piece 拆成两个)。
真正换到的是:用一半的词表槽位做到接近的压缩,省下 143M 嵌入参数,以及一个
完全自己的分词器。**目标是独立自主,不是效率。**

## 架构

28L / 2048H / 6144I / GQA 16:8 / head_dim 128,与 Qwen3-1.7B-Base 完全一致 ——
**这个项目只换词表,不改架构**。

| | Qwen3-1.7B-Base | 本项目 |
|---|---|---|
| 词表 | 151643(BBPE) | **81903**(piece,自训) |
| 参数量 | 1,720,574,976 | **1,577,147,392** |
| 其中嵌入 | 310.6M | **167.7M** |

对照 GPT-2 XL 的 1,557,611,200 —— 相差 1.25%,基本同规模,但形状完全不同
(GPT-2 XL 是 48 层 ×1600,深而窄)。相当于用 2025 年的架构重新分配那笔参数预算。

两阶段配方,1.2B token / 约 30 小时(单张 4090):

- **Phase 1** 冻结 transformer,只训 embed + lm_head。3815 步,1B token,25.1 小时
- **Phase 2** LoRA tie-safe 退火:q/v 走 r=16 低秩旁路,embed 全参训,lm_head
  因 tie 自动同步。1500 步,200M token,5.2 小时

嵌入初始化用 mean-of-BBPE:新词表每个 piece 过一遍 Qwen 的原生分词器,取那些
旧嵌入的均值。实测 66.4% 是一对一映射(等价于 ReTok 论文的「两边都有就直接
复制」),32.5% 多对一,1.1% 落回全局均值。

每条选择的理由见 [`docs/WHY.md`](docs/WHY.md)。

## 代码

四层,按数据流切:

```
data/       下载语料。source.py 是数据源注册表,唯一的真相来源   make -C data
prepare/    编排:依赖、词表手术、预编码、调训练、评测            make -C prepare
src/        模型定义 + 训练循环
save/       导出 HF 发布包、上传、核对
```

两条分层约束:

- **`src/` 只依赖 torch。** Qwen3 的 forward、LoRA、safetensors 读写、
  Muon/Aurora 都是自己实现的(替掉 transformers、peft、safetensors 库)。
  这条的意义是让读者**看得见每一步在做什么**
- **`src/` 不碰文本。** 分词、字→id 全在 `prepare/`,`src/` 只读预编码好的 id。
  所以 PieceTokenizer 不是 `src/` 的依赖

自写实现的验证:与 transformers 逐层对齐 float32 最大 **1.66e-6**、argmax
**100% 一致**;与 peft 对拍 logits **逐位相同**。

每层一个 README 说明这层在做什么。另有 `deps/`(clone 的 C++ 依赖,gitignore)、
`docs/`、`test/`。

## 重做一遍

语料、词表、基座权重全部从 Hugging Face 和 GitHub 获取,不需要任何本机既有文件。

```bash
make deps                  # clone + 编译 PieceTokenizer(词表也在它仓库里)
make -C data probe         # 验数据源注册表写对没(每源只下一个文件)
make -C prepare status     # 每一步产物在不在
```

| | 时间 | 从哪开始 | 教程 |
|---|---|---|---|
| 评测已发布模型 | 分钟级 | HF 上的 ReTok 模型 | [`docs/eval/pipeline.md`](docs/eval/pipeline.md) |
| 两阶段继续预训练 | 约 30 小时 + 预编码 | Qwen3-1.7B-Base | [`docs/PRETRAIN.md`](docs/PRETRAIN.md) |

建议先跑评测 —— 分钟级就有反馈,而且能验证整条链路(词表、模型、数据、指标)
是通的。`make test` 是回归防线,几分钟。

唯一的手动步骤:`BAAI/CCI3-HQ` 是 gated 数据集,要先在 HF 页面接受条款并
`huggingface-cli login`。

## 文档

| | |
|---|---|
| [`docs/WHY.md`](docs/WHY.md) | 各处设计选择的理由,以及**改错了不报错**的地方 |
| [`docs/PRETRAIN.md`](docs/PRETRAIN.md) | 从零重做一遍的全流程,含每步耗时和磁盘占用 |
| [`docs/eval/pipeline.md`](docs/eval/pipeline.md) | 评测栈:三个入口、数字怎么读 |
| [`data/source.py`](data/source.py) | 数据源注册表 + 逐个溯源的依据 |

## 环境

Python 3.11 + torch 2.11,单张 RTX 4090(24GB,bf16),没有多卡代码路径。

```bash
uv pip install -r requirements.txt          # 训练 + 评测,一条就够
```

**一个 venv 就够**,训练和评测都在里面。之所以是 3.11 而不是更新的版本:
vllm 和 comet 上不了 3.14。解释器路径写在 gitignore 的 `local.mk` 里。

C++ 依赖 [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer) 由
`make deps` 自动 clone 并编译(需要 `cmake` 和 C++17 编译器)。
**81903 词表和中文分词词典都在它仓库的 `save/` 下,本仓库不留副本。**

## 许可

Apache-2.0。训练语料各自的许可见对应数据集卡。
