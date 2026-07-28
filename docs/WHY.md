# 为什么这么做

各处设计选择的理由,以及**改错了不报错**的地方。否定结果也写 —— 论文不发
失败尝试,教材最该有。

---

## 一、会静默出错的地方

排在最前面。这些改错了不会报错、不会崩、loss 照样是个正常数字,只是结果全错。

### 1. `dict.txt` 缺失会改变中文的 token id

不只是慢。2026-07-27 实测(同一 `piece.model`,带 / 不带 dict):

| 输入 | 带 dict | 不带 dict | 倍数 | id 序列 |
|---|---|---|---|---|
| 中文 22 字 | 0.169 ms | 1.060 ms | 6.3× | **不同**(长度都是 22) |
| 中文 ×20(1780 字) | 3.289 ms | 21.125 ms | 6.4× | **不同**(长度都是 440) |
| 英文 | 0.110 ms | 0.110 ms | 1.0× | 相同 |

**round-trip 正确会掩盖这个问题** —— 两种模式 decode 都能还原原文,看着没事,
但喂给模型的 id 已经不是训练时那套了。英文不受影响,dict 只管中文切分。

`prepare/tokenizer.py` 现在缺 dict 直接报错。旧版本是
`else: self._tok.load(model_file)`,静默降级。

顺带纠正:老文档说「没有 dict 会慢 100–1500×」,实测在这些长度上只有 6×。
那个量级要更长的输入才出得来,不该当成一般情况描述。

### 2. RoPE 被 autocast 降到 bf16,位置编码错乱

bf16 尾数只有 8 位,整数超过 256 就不精确:

    255 → 255    256 → 256    257 → **256**    1023 → **1024**

`RotaryEmbedding.forward` 里的 `inv_freq @ position_ids` 如果不强制 float32,
外层的 `torch.amp.autocast(bfloat16)` 会把它降精度。代价:seq_len=1024 上
next-token loss 从 **2.3331 涨到 2.7605**,不报错。

`src/model.py` 里有 `with torch.autocast(device_type=dev, enabled=False)`,
HF 的 `Qwen3RotaryEmbedding` 里也有同样的防护(注释写着 `Force float32`)。

**这个 bug 是 PPL 锚点抓到的,逐层等价性测试漏了** —— 那两轮用的序列只有
256 长、也没开 autocast,两个条件都不满足就复现不出来。教训:**等价性测试的
输入条件本身就是覆盖面**。现在 `test_model_equiv.py` 里有
`check_long_autocast()` 专盯这个组合。

### 3. state_dict 的 key 对不上,模型照样跑

已发布的 `Ismantic/Qwen3-1.7B-Base-ReTok` 用标准 HF 命名。改模块名或嵌套层级
会让那份权重全部失配,而 `load_state_dict(strict=False)` 会**静默跳过** ——
模型随机初始化照样跑、loss 照样降。

`src/model.py` 的 `from_pretrained` 手动检查 missing/unexpected 并报错。
防线是 `test/test_model_equiv.py`。

同理 `src/lora.py` 的 `load_adapter`:adapter 没加载上,模型照样跑,只是等于
没微调。

### 4. `tie_word_embeddings` 的 checkpoint 里没有 `lm_head.weight`

v18 的 checkpoint 是 310 个张量 —— 28 层 ×11 + `embed_tokens` + `norm`,
**没有 lm_head**。加载时要把它绑回 embed,不能指望文件里有。

peft 的 `ModulesToSave` 会 deepcopy embed/lm_head,那正是 tie 被破坏的根源,
也是当初要单开 `--lora_tie_embed_head` 的原因。自写的 `src/lora.py` 不存在
这个问题:根本不去动 embed 模块,只是把它解冻。

### 5. `*.pt` 的 gitignore 规则会吞掉回归 fixture

`test/fixtures/ppl_slice.pt` 被 `*.pt` 匹配上,新 clone 出来缺这个文件,
`test_reproduce_sota.py --only ppl` 会**静默跳过**(而不是失败)—— 等于没有防线。
已加 `!test/fixtures/*.pt`。

### 6. 上游 API 改名,悄悄弄坏了一条没人走的路径

`prepare/encode_corpus.py` 调 `tok.load(model, cn_dict=...)`,而 PieceTokenizer
在 commit `20d55e0`(*Refactor tokenizer dispatch and harden model I/O*)把这个
关键字从 `cn_dict` 改成了 `dict`:

    - py::arg("model_file"), py::arg("cn_dict") = ""
    + py::arg("model_file"), py::arg("dict") = ""

**预编码从那次改名起就是坏的,一直没人发现** —— v18 之后就没再跑过预编码,
直到 2026-07-27 才撞上。现在改成位置参数 `tok.load(model, cn_dict)`,不依赖
关键字名。

同类的还有 `save/export.py` 里写死的 `core/tokenizer_wrapper.py`(阶段 1 搬到
`prepare/tokenizer.py` 了),以及 `prepare/Makefile` 里用相对路径传 `BASE`
(`make -C` 会切目录,相对路径被 transformers 当成 repo id)。

教训:**改完的代码路径必须真跑一遍。** 语法检查、dry-run、import 检查都发现
不了这类问题。改造期间三个 bug 都是这么抓到的。

### 7. 手术产物必须带 `dict.txt`

`prepare/retok.py` 原来只拷 `piece.model`。手术产物缺 dict 的话,后面
`make p1` 加载它会被第 1 条的检查拦住 —— 整条链断在这里。现在一起拷。

### 8. PieceTokenizer 重建后行为可能变,而且没有提示

分词器是 C++ 编的,换 commit、换编译器、换 CMake 选项都可能改变编码,
不会有任何提示。所以:

    1. python test/capture_baseline.py     ← 重建**之前**抓基线
    2. bash prepare/install_deps.sh        ← 重建
    3. python test/test_tokenizer.py       ← 比对

顺序反了就失去意义。

**本机已经出现过版本漂移,而且它真的弄坏了东西。** `.venv` 装的
piece_tokenizer 来自 `BERTc/deps/PieceTokenizer`(commit `7331d09`),不是老
文档说的 `Shiyu/PieceTokenizer`(`5c3f081`)。这两个 clone 的 `src/` 逐字节
相同,所以**编码行为**一致 —— 但真正要紧的漂移不在这两者之间,而在

    v18 训练时用的版本  ←→  现在两个 clone 都在的版本

之间:中间隔着 commit `20d55e0`,那次把 `load()` 的关键字从 `cn_dict` 改成
`dict`(见第 6 条)。**只比 `src/` 是不够的 —— python binding 的签名也是行为
的一部分。**

`test_tokenizer.py` 比的是编码行为,抓不到 API 改名(它自己用位置参数)。
抓到那个的是「真跑一遍」。两种检查都需要。

---

## 二、评测:什么样的差异才算数

### vLLM 的贪心解码不可复现

`temperature=0.0` 挡不住。连续批处理 + chunked prefill + 异步调度让每次的
batch 组成不同 → bf16 归约顺序不同 → 接近平局的 argmax 偶尔翻转。

同一 ckpt、同一条命令跑 6 次:

| | 最小 | 最大 | range | sd |
|---|---|---|---|---|
| zh-en BLEU | 20.4081 | 20.5114 | **0.1033** | 0.022 |
| en-zh BLEU | 36.0314 | 36.1651 | **0.1337** | 0.056 |
| zh-en COMET | 0.7932 | 0.7942 | 0.0010 | 0.0005 |
| en-zh COMET | 0.8441 | 0.8447 | 0.0005 | 0.0002 |

**checkpoint 之间 0.1 量级的 BLEU 差是噪声,不能当结论。** COMET 稳两个
数量级 —— 它是连续打分,不被个别 token 翻转放大,比 BLEU 可靠。

头两次重跑曾与历史数字对到小数点后四位,据此一度写下「这套 eval 是确定性的」。
**那是碰巧撞上了相同的批次调度。一次相等不等于确定性。**

mono 六任务里,样本量大的(lambada 5153 / hellaswag 10042)能精确命中;
ceval 的子任务只有 18–31 题,翻一两道就看得见 —— 聚合值实测漂 +0.0022。

### 不能混后端

`transformers` 和 vLLM 是两套注意力内核。在**训练充分**的 Qwen3-1.7B-Base 上
实测:

| 任务 | transformers | vLLM | 差 |
|---|---|---|---|
| lambada_openai acc | 0.6290 | 0.6513 | **+0.0223** |
| piqa acc_norm | 0.7720 | 0.7731 | +0.0011 |
| arc_challenge acc_norm | 0.5546 | 0.5512 | −0.0034 |

**lambada 差 2.2 个点,方向还不一致**(arc 是负的)—— 这比 v18 P2 与 P2-tie
之间的全部差异都大。老文档说「under-trained ckpt 上可达 ±10%」,实测表明
**训练充分的模型上照样能差这么多**,不是欠训练特有的现象。

所以 `eval_results/full/` 的目录名带后端后缀(`_vllm` / `_tf`),
`prepare/sweep.py` 的 `read_mono()` **只认 `_vllm`,找不到就留空,不去凑 `_tf`
的数**。空着比填错好。

这张表本身是补出来的:base 原先只在 vLLM 上跑过 ceval / gsm8k / hellaswag,
另外三个任务只有 transformers 的数。也就是说任何「相对 base 的损失」表在那
三项上都是跨后端比较。2026-07-27 重下 base 补齐了 vLLM 侧。

### 已知不会恢复的:gsm8k

换词表后 gsm8k 从 base 的正常水平掉到 ~0.035,两个 phase 都救不回来。
81903 的 piece 词表打断了 Qwen3 的数值 token 化方式。**这是词表代价,
不是回归**,不要当成 bug 去追。

---

## 三、这个项目换到了什么

ReTok 常见的动机是「换个词表提高压缩率」。**这个项目没换到,反而略降。**

tokens_per_char(越低越好),两边 encode 都不加特殊 token:

| 样本 | Qwen BBPE | Piece | Piece/Qwen |
|---|---|---|---|
| WMT22 中文(1000 句) | 0.6359 | 0.6658 | **1.047** |
| WMT22 英文(1000 句) | 0.2159 | 0.2180 | 1.010 |
| 训练语料切片(混合) | 0.2750 | 0.2787 | 1.013 |

例子:`中国科学院在北京` → Qwen 3 个 token(`中国`/`科学院`/`在北京`),
Piece 4 个(`中国`/`科学院`/`在`/`北京`)。Qwen 的 151K 词表有更多中文多词合并。

### 完整的账

| | |
|---|---|
| 代价 | WMT22 zh-en BLEU −1.88、en-zh −2.31 |
| 代价 | gsm8k 归零 |
| 代价 | 中文多用 4.7% token —— **推理更贵,不是更省** |
| 收益 | 词表 151643 → 81903,小 46% |
| 收益 | 嵌入参数 310.6M → 167.7M,**省 142.9M** |
| 收益 | 分词器自有,不依赖 Qwen |

**目标是独立自主,不是效率。** 用一半的词表槽位做到接近的压缩,省下 143M
嵌入参数,换来一个完全自己的分词器和底座 —— 性能降一些是认了的代价。

顺带:整模型 1,577,147,392 参数,与 GPT-2 XL(1,557,611,200)差 1.25%,
基本同规模。但形状完全不同:GPT-2 XL 是 48×1600 深而窄,这个是 28×2048 +
GQA/SwiGLU/RoPE。相当于用 2025 年的架构重新分配 GPT-2 XL 那笔参数预算。

未验证的猜想(别写成结论):`在`/`北京` 这种拆分从语言学上更干净(介词 +
地名不该并成一个 token),对翻译泛化可能有利。没人测过。

---

## 四、训练配方

### 为什么 Phase 2 用 LoRA 而不是全参数解冻

全参数解冻在 Qwen3-0.6B 时代(v7–v16)是**破坏性的**。LoRA tie-safe 才是
跑通的配方:transformer 走低秩旁路(q_proj/v_proj, r=16, α=32),
embed_tokens 全参训,lm_head 因 tie 自动同步。

注意博客《低成本实践:训练GPT》里把第三步写成「放开全部参数」—— 那是 ReTok
论文的通用三步,不是这个项目实际做的。

### `min_lr_ratio` 不能是 0

衰减到 0 是已知的病态。v18 用 0.01。

### Aurora:在 SOTA 路径里,但当前配置下**未经消融**

`src/optim.py` 里 Aurora 必须留 —— 去掉就复现不出 20.46。训练日志:
`Aurora params: 3,211,264 | Adam params: 167,737,344`,优化的正是 LoRA 的
`lora_A`/`lora_B`。

但它的理由站得不牢:

1. 唯一那次消融(`v8_s2_aurora` vs `v8_s2_muon5e5`,500 步)结论是**打平**,
   不是 Aurora 更好。留它靠的是「不对称风险收益」的推论。
2. 那次是在 **0.6B + 全参数 Phase 2** 上做的;现在是 **1.7B + LoRA**。
3. Aurora 的立论(tall matrix 的 neuron death、MLP expansion 在甜区)针对
   **稠密权重矩阵**。LoRA 下它优化的是 `lora_A`(16×2048)和 `lora_B`(2048×16)
   —— 极端低秩因子,几何性质与原论文讨论的对象不同,那套理由不见得能迁移。

现成的实验:同配方关掉 `--use_aurora` 跑一次(1500 步约 5 小时),就知道那 6%
开销买到了什么。

### 两阶段的实际用量和耗时

来自 v18 的训练日志,不是估算:

| | 步数 | 数据 | 耗时 | 速度 |
|---|---|---|---|---|
| Phase 1 | 3815 | 1B token | **25.1 小时** | 23.7 秒/步 |
| Phase 2 tie | 1500 | 200M token | **5.2 小时** | 12.4 秒/步 |
| 合计 | | **1.2B token** | 约 30.3 小时 | |

loss 轨迹(每 50 步采样):

    p1   step   50: 3.8089  →  step 3800: 2.8079
    p2   step   50: 2.5718  →  step 1500: 2.3942

**p2 的起点(2.57)低于 p1 的终点(2.81)** —— 因为 anneal 用的是更高质量的
200M token 混合。2026-07-27 在新目录上各跑 12 步复验过:p1 落在 2.77–4.02、
p2 起点 3.13 与它自己的 p1 终点 3.25 连续(LoRA 的 B 零初始化保证接上去那一刻
不破坏基座),形状和衔接都对。

### 显存:配方是 48GB 的,不是 24GB 的

`--batch_size 16` 是 v18 在 A6000(48GB)上训的。24GB 的 4090 会 OOM ——
失败的分配约 5GB,正是 loss 里 `logits.float()` 的
`16 × 1023 × 81903 × 4 字节 = 5.4GB`。

**Makefile 的默认值保持 v18 原样**(它就是产出已发布权重的那份配方),
小卡按等效 batch 覆盖:`make p1 BATCH=4 ACCUM=64`。

### 数据混比

    main   1B token   EN 0.663 / CN 0.336   8 源
    anneal 200M token EN 0.60  / CN 0.40    6 源

权重见 `prepare/encode_corpus.py`。**池子 ≠ 消耗量**:各源实际吃掉的都在
1 亿 token 上下,而上游单个文件动辄几亿 —— SkyPile 一个 jsonl 就有约 319M
token,够喂满它 126M 的份额。`n_parts` 取的是池子规模(对齐当初的采样多样性),
想省磁盘调小即可。

---

## 四点五、Summer-0.5B:从零预训练那条线

ReTok 是「拿现成的模型换词表再修回来」。Summer-0.5B 反过来:**不要任何现成
权重**,用 Qwen3-0.6B-Base 的架构、Summer 的 81903 词表,从随机初始化训一个
只对中英翻译负责的底座。两条线共用词表、语料池和 `src/train.py`,别的都不共用。

### 为什么叫 0.5B 而基座叫 0.6B

换词表之后 tie 的嵌入从 155.6M 掉到 83.9M:

| | 嵌入 | transformer | 合计 |
|---|---:|---:|---:|
| Qwen3-0.6B-Base(151936 词表) | 155.6M | 440.5M | 596.0M |
| Summer-0.5B(81903 词表) | 83.9M | 440.5M | **524.3M** |

transformer 一个参数没动,少掉的 71.8M 全在嵌入。**叫 0.6B 会误导** ——
差 12%,而且差在最容易被当成「同规模」的地方。

### 12B token 是怎么定的,不是拍的

2026-07-28 在 4090 上实测(seq 1024、bf16 autocast、torch.compile、Muon+Adam、
梯度累积 8):

| batch | 梯度检查点 | 分块 CE | tok/s | B token/天 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 8 | 关 | 开 | 25852 | 2.23 | 21.17 GiB |
| 16 | 开 | 开 | 20516 | 1.77 | 17.78 GiB |
| 16 | 开 | **关** | — | — | **OOM** |

取 batch 16 + 检查点:21.17 GiB 在 24GB 卡上只剩 2.8G 余量,跑六天碰上碎片化
就前功尽弃,换 20% 吞吐买这个稳当。1.75B token/天 × 6.9 天 = 12B,
即 **23 token/参数**,正好在 Chinchilla 最优附近。

参照:Qwen3-0.6B-Base 吃了 **36T token** —— 差 3000 倍。

### 中英改成 50:50,质量优先于多样性

ReTok 的 main mix 是 EN 0.663 / CN 0.336,那是继承 Qwen 的语料分布。从零训练
没有可继承的东西,配比直接服务于目标:翻译两个方向都要考,所以对半。

12B token 对 0.52B 参数只有 23 token/参数,每个 token 都得算数,所以原始网页
(CC_EN / CC_CN / SkyPile)保留但降权,教育向和合成教科书顶上去。Gutenberg
压到 0.03 —— 19 世纪英语对现代新闻翻译帮助小,而且池子实测只有 1.0B token。

**池子喂不满不会报错。** worker 读完手上的文件就停,产出比目标少,最后拼出来
的照样是一份合法的训练数据,只是配比不是你写的那个。CCI3-HQ 和
CN_FineWeb_Edu 的 `n_parts` 就是为这份配比扩过的(实测前者 2.6B、后者 0.9B
token,而配比要吃 1.68B 和 1.44B —— 后者**池子比消耗量还小**)。
`encode_corpus.py` 末尾因此有一道实际产出对账,低于 90% 直接报错。

### 从零训练必须 fp32 参数

ReTok 两阶段的参数是 bf16,跑得好好的。从零训练不行:

bf16 的相对精度是 2⁻⁸ = 0.39%。退火末段 Adam 的学习率降到 3e-5,而嵌入权重的
量级是 0.02 —— 单步更新只占权重的 0.15%,**低于 bf16 的分辨率,直接被舍掉**。
表现是训练末段「学不动了」,而 loss 曲线看着只是正常收敛。

继续预训练不吃这个亏,是因为它的起点已经是个好模型,末段那点微调本来就可有可无。
实测代价:fp32 参数只多花 2.7GB 显存,**吞吐几乎不变**(20,200 vs 20,516 tok/s)。

### WSD 而不是 cosine

cosine 把整条学习率曲线绑死在 `max_steps` 上。而从零预训练**开跑的时候总步数
其实定不下来** —— 要看中途的 few-shot 翻译曲线才知道跑到哪儿够。想提前收尾,
前面那些步就是按错的曲线走的;想延长,学习率已经衰到底,续不动。

WSD 的恒定段没有这个耦合:改主意时用新的 `max_steps` 续跑即可,前面走过的路
一模一样,退火窗口自动落在新的末尾。

### 看 BLEU,不看 loss

从零预训练的 loss 从头到尾都在降,**降到哪儿算能翻译,曲线上看不出来**。
1.9 和 1.8 之间可能什么都没变,也可能正好跨过 few-shot 涌现的门槛。所以
`prepare/eval_hook.py` 每 2000 步直接测一次 WMT22 5-shot BLEU,把它当主曲线。
全程 vLLM 不换后端 —— 混了后端趋势就是假的(见第二节)。

### 初始化:这个项目最大的一个「不报错的洞」

2026-07-28 之前 `src/model.py` **完全没有初始化代码**。ReTok 那条路永远从
checkpoint 加载权重,所以这个洞一直不显形;从零训练一走上去,拿到的是 torch
的默认值:

    nn.Embedding  → N(0, 1)         标准差比该有的 0.02 大 50 倍
    nn.Linear     → U(±1/√fan_in)   均匀分布,且 fan_in 一变标准差就跟着变

嵌入 std=1 而且 tie 到 lm_head,初始 logits 的标准差会是 32 而不是 0.64,
初始 loss 上百。**照样能训**,只是慢、抖、偶尔崩,回头查不出原因。

现有的五项测试一条都拦不住它 —— equiv / lora / retok / ppl 全是「加载已有权重
之后」比对,tok 只管分词器。所以新增了 `test/test_init_scratch.py`,判据落在
初始化之后、训练之前的统计量上:嵌入 std、残差出口 std = std/√(2L)、
RMSNorm 全 1、激活按 √深度增长、初始 loss ≈ ln(V)。

### 续训必须恢复优化器状态

`--resume_from` 原来是个空占位符(帮助文本自己写着 unused),而 `save_steps`
只存权重。跑六天不能续,等于赌六天不断电。

只加载权重的「假续训」尤其坏:Muon 的动量是 5 步 Newton-Schulz 正交化之后的
累积量,丢掉它等于让模型在训练中途重新经历一次冷启动 —— loss 跳一下再慢慢
压回去,而日志上只显示「续训成功」。数据位置同理:不记录吃到哪儿,断一次就
重复喂一批,重复的部分等于白训,**而且看不出来**。

验证方式是对拍:不中断跑到 step 30 vs 从 checkpoint-20 续到 30,
step 25 的 loss 是 **7.7733 vs 7.7733**,逐位相同。

---

## 五、结构与依赖

### vLLM 为什么能加载纯 torch 训出来的权重

**vLLM 根本不用我们的代码。** 它有自己的 Qwen3 实现,认模型靠读 `config.json`
里的 `architectures: ["Qwen3ForCausalLM"]` / `model_type: qwen3`,然后按
**state_dict 的 key 名**把权重灌进去。

所以「`src/` 自己实现模型」和「用 vLLM 评测」不冲突 —— 两边共享的是**权重文件
和 key 命名**,不是代码。这就是那条红线的由来:

> `src/model.py` 的 state_dict key 必须与 HF 的 `Qwen3ForCausalLM` 一致

key 是我们与 vLLM / transformers 之间**唯一的契约**。改了它,vLLM 会静默地灌
不进去或灌错 —— 而且照样跑出数字。

推论:**`trans` 和 `mono` 走 vLLM,所以测不到 `src/model.py`。** 能锚住自写
模型的只有 `test_model_equiv.py` 和 `--only ppl`。

分词器是 vLLM 唯一处理不了的部分(它认不了 piece 词表),所以评测时传
`skip_tokenizer_init=True`,由 `prepare/translate.py` 自己编码好 id 再用
`TokensPrompt(prompt_token_ids=...)` 喂进去。

### 发布包自足:只要 torch + PieceTokenizer

发布包里带模型代码(`model.py` / `checkpoint.py` / `tokenizer.py` /
`example_load.py`),所以**下载的人不需要 transformers,也不需要 safetensors
库**。BERTc 的发布包一直是这个做法。

2026-07-27 用 import hook 屏蔽 `transformers` / `safetensors` / `peft` 实测过:
加载、分词、前向全部跑通,1,577,147,392 参数。

词表文件用**上游名** `Summer-Tokenizer.pt` / `Summer-Tokenizer.dict.txt` ——
与 PieceTokenizer 仓库 `save/` 下同名,拿到发布包的人一眼就知道词表出自哪里,
不用靠 sha256 反推。旧名 `piece.model` / `dict.txt` 在 `prepare/tokenizer.py`
里保留为回退,因为已发布的模型和 v18 的 checkpoint 都在用。

### `src/` 只依赖 torch

模型、LoRA、优化器、safetensors 读写都是自己实现的。这条约束的意义是让读者
**看得见每一步在做什么** —— safetensors 的格式只有二十来行代码,藏进依赖里
就看不见了。

| 自写 | 行数 | 替掉 |
|---|---|---|
| `src/model.py` | ~290 | `transformers.Qwen3ForCausalLM` |
| `src/lora.py` | ~180 | `peft` |
| `src/checkpoint.py` | ~120 | `safetensors` 库 |
| `src/optim.py` | ~400 | Muon / Aurora(本来就是自写) |

验证:与 transformers 逐层对齐 float32 最大 **1.66e-6**、argmax **100% 一致**;
与 peft 对拍 logits **逐位相同**。

### `src/` 不碰文本

分词、字→id 全在 `prepare/`,`src/` 只读预编码好的 id。所以 PieceTokenizer
不是 `src/` 的依赖。`--mode sft` 现在会**显式报错** —— 自写的 Attention 只做
`is_causal`,不接受 `attention_mask`,而 SFT 会 padding,忽略 mask 会让 padding
位参与注意力,算错且不报错。

### 一个 venv 就够 —— 但方向搞反了就以为不行

现在只有一个:`~/.venv-e`(Python 3.11),训练和评测都在里面。

**先试的方向是错的。** 当时想把 vllm + comet 装进 3.14 那个训练 venv,失败:

- vllm 没有可用的 cp314 构建
- comet 的依赖链(torchmetrics 0.10.3)要 `functools._HashedSeq`,3.14 里没有了
- `uv pip install vllm unbabel-comet` 把解拽回 **vllm 0.2.5**(2023 年底,不认识
  Qwen3),连带降级 numpy / pandas / pydantic。已靠事前的 `uv pip freeze` 完整回滚

于是写下了「两个 venv 是必要的」。**这个结论下早了** —— 它只证明了「vllm 上不了
3.14」,没有证明「训练下不到 3.11」。反方向根本没试。

反方向是通的,因为 `src/` 只依赖 torch,而 torch 有 cp311。2026-07-27 验证:

    分词       两个 Python 上 vocab_sha / dict_sha / 5 段文本逐 id 完全相同
    forward    固定切片 next-token loss 2.3334(3.11)vs 2.3331(3.14)
    对拍       与 transformers 逐层对齐、与 peft 误差 0.000e+00,都在 3.11 上跑通
    训练       同 seed 12 步,step 1-2 loss 逐位相同,之后差在第 4 位小数
    速度       稳态都是 0.30s/step;3.11 启动慢 10 秒,那是 compile 开销不是吞吐

合并要补两块,都不明显:

- **`piece_tokenizer` 必须是 editable 装法**,不能是拷进 site-packages 的裸
  `.so`。`resolve_assets()` 靠 `piece_tokenizer.__file__` 反查词表,拷进去就
  查不到了。
- **`peft` 得装**,不然 `test_lora.py` 静默降级成「自查通过(未与 peft 对拍)」——
  数学判据就没了,而它照样打印「通过」。

另外 PieceTokenizer 的 `setup.py` 传的是 `-DPYTHON_EXECUTABLE`,**新版 CMake 的
`FindPython` 不认这个名字**(要 `Python_EXECUTABLE`),于是它去挑系统 python。
之前在 3.14 上没暴露纯属巧合 —— 系统 python 恰好也是 3.14,ABI tag 撞对了。
换 3.11 就露馅:产出 `cpython-314` 的名字,setuptools 在等 `cpython-311`,
报的错却是「.so 不存在」。见 `prepare/install_deps.sh` 里的处理。

### venv 改名会留下 venv 外面的死引用

`~/.venv-eval` 改名成 `~/.venv-e` 时,修了 venv 内部 68 个文件的路径
(`bin/` 的 shebang + `pyvenv.cfg`),`lib/` 下确认干净,以为完了。**不是** ——
JIT 编译缓存在 venv 外面:

    ~/.cache/flashinfer/.../build.ninja    写死旧路径 → vLLM 引擎起不来
    ~/.cache/vllm/torch_compile_cache/     6644 个文件提到旧路径(实测无害)
    ~/.triton/                             30 个文件同上

报错长这样,跟改名一点关系看不出来:

    ninja: error: '~/.venv-eval/lib/python3.11/.../sampling.cu',
    needed by 'csrc_sampling.cuda.o', missing and no known rule to make it
    RuntimeError: Engine core initialization failed

**整个 vLLM 评测栈都是坏的,而训练和 `make test` 五项全绿** —— 因为它们不碰
vLLM。所以「测试全过」不等于「改名没出事」。修法:`rm -rf ~/.cache/flashinfer`,
它会自己重建。3.8GB 的 `~/.cache/vllm` 不用动,那里面的旧路径是惰性记录。

一般规律:**改动解释器路径之后,要把用它的每条链路都真跑一遍**,而不只是跑
测试 —— 编译缓存、`.pth`、egg-link、外部工具的配置都可能留着旧路径。

教训两条:

- `uv pip install --dry-run <A>` 的结果**不代表** `install <A> <B>` 的结果。
  装之前先 `uv pip freeze` 存快照。
- **「A 装不进 B」不等于「B 装不进 A」。** 一个方向失败就写下结论,会把一个
  临时状态固化成架构。

### 并发默认不写死

`prepare/encode_corpus.py` 原来写死 `--num_workers 28` —— 那是 2× A6000 那台
机器的核数。本机 16 核,28 个 spawn 进程各自加载分词器 + 预分配 numpy 数组,
1B token 预算下能把 61GB 内存压满,**2026-07-27 实测把整台机器卡死过一次**。

现在默认 `min(12, cpu_count() - 1)`。再多也只是抢内存,I/O 早就饱和了。

同类的还有 `requirements.txt` —— 改造前根本没有这个文件,新用户不知道该装什么。
现在有了,而且**就一个** —— 曾经拆成训练 / 评测两份,那是两个 venv 时代的遗留:
同一个环境、没有「只装其中一份」的场景,拆开只是多一条要记的命令。

### 没有多卡代码路径

本机一张 RTX 4090。旧脚本里的 `--nproc_per_node=2` 和 `--gpus 0 1` 是
2× A6000 时代的遗留,已删。

---

## 六、数据源溯源

十个语料源逐个核过。详表在 `data/source.py` 开头,这里只记方法和结论。

**确证**(sha256 或首条记录逐字段比对):Gutenberg、FineWebEdu、Wikipedia_EN、
CCI3-HQ、CN_FineWeb_Edu、Cosmopedia。

**已替换**(原始出处不明或不可得):

- `Wikipedia_EN/CN` → `HuggingFaceFW/finewiki`,与 BERTc 同源
- `CC_EN/CC_CN` → `HuggingFaceFW/fineweb` + `fineweb-2`

后者原本读的 `c2-en`/`c2-cn` 是自己跑 pipeline 处理 CC-MAIN-2023-50 的产物,
数据内嵌 `file_path` 指向已不存在的 `/mnt/data/now/DataPipeline`。对照
FineWebEdu 的 `file_path: s3://commoncrawl/…` 才是上游原版,反证了 c2 是本地
产物,生产代码已经没了。

**`C4_EN`/`C4_CN` 是误标。** v17 确实用 `allenai/c4`,v18 换成了 CC-MAIN-2023-50
的切片却没改变量名 —— 读代码的人会一直以为是 C4。已正名为 `CC_EN`/`CC_CN`。

Gutenberg 是这么找到的:目录里残留着 `.cache/huggingface/download/*.metadata`,
里面有 blob sha256;拿 README 的特征值(`num_examples: 9930` + `sha256` 字段)
扫 HF 上 63 个 gutenberg 相关数据集,命中一个,再用 sha256 确认。

注册表里的 `repo_id` 是**公开等价物**,不保证与 v18 当初吃进去的字节相同。
这不影响复现 v18(checkpoint 已发布,回归测试钉的是 checkpoint 行为),
本表管的是将来从零重建语料。

**验证方式:`make -C data probe`** —— 每个源只下第一个文件并真读一遍。
会写错的是 glob / fmt / text_field 这三样,一个文件就能暴露,不需要下满
100GB。改了 `source.py` 之后跑它,别跑 `make download`。

---

## 七、被推翻的结论 / 没做的事

- **「vLLM 评测是确定性的」** —— 错。见第二节。
- **`docs/reports/v18_tie_lineage.md` 里 P2 vs P2-tie 的 BLEU 对比落在噪声内。**
  WMT22 zh-en 差 −0.1102、WMT23 差 +0.0046,而噪声 range 是 0.1033。
  这不推翻 v18_p2_tie 当 SOTA —— 报告同时给了结构性理由(保住 tied embedding
  不变量,对下游 SFT 和打包重要),那个理由跟指标无关。被削弱的只是「指标上
  也不差」这半句。
- **v19(from-scratch)从未训练。** 预编码在 2026-05-24 跑过一半就停了
  (`output/v19_pretok.log`),训练没开始。相关代码在 `_attic/`。
- **`compare_tokenizers.py` 比的是新旧两版 piece,不是 Qwen vs piece。**
  项目立论所需的那个对比(第三节那张表)之前从来没测过。
