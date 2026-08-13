# Summer-0.5B:从零训一个 0.5B 双语底座

2026-08-01 ~ 08-12。与 ReTok 那条线(给 Qwen3-1.7B-Base 换词表)**无关** ——
这是另一条线:不借任何现成权重,随机初始化开始训。

这份只记**底座怎么训出来的**。它走完下游翻译流水线的结果在
[Interpreter 的 `docs/reports/summer-0.5b-downstream.md`](https://github.com/Ismantic/Interpreter/blob/main/docs/reports/summer-0.5b-downstream.md)
——下游那一段的坑也记在那边,因为出问题的代码在那个仓库。

S0 接下来还要走另一条路:midtrain → SFT,做成通用 chat 模型
(nanochat 那套)。所以它不只是 S1 的对照组。

## 模型

```
架构    Qwen/Qwen3-0.6B-Base 的 config(28 层 / hidden 1024 / GQA 16:8 /
        head_dim 128 / tie / RoPE theta 1e6)
词表    自训 81903 piece(与 ReTok 线同一份)
参数    524,336,128(embed 83,868,672 与 lm_head 绑权重 + transformer 440,467,456)
```

**为什么叫 0.5B 而不是 0.6B**:Qwen3-0.6B-Base 本身是 596.0M,其中嵌入
155.6M(词表 151643)。换成 81903 的词表,嵌入降到 83.9M,总数变成 524.3M。
名字跟着实际参数量。

## 两段预训练

| | 数据 | 步数 | token |
|---|---|---|---|
| **S0** | `scratch` 中英 50:50,**纯单语** | 1 → 45,149 | 11.8B |
| **S1** | `anneal_mt` 含 30% 中英平行语料 | 40,000 → 45,149 | 1.2B |

S1 从 S0 的 step 40,000 分叉,同起点、同超参、同学习率时间表(WSD,decay 从
40,635 起)——**只差数据**。所以两者的差可以整体归因到那 30% 平行语料。

S0 跑完整个衰减段这件事不是计划:原本要在 40,635 换数据集,但那需要人在场
按一下,而当时没人。结果白得一个干净的单语对照组。

配方:Muon(440,401,920 个 2D 参数,lr 0.02)+ 辅助 AdamW(83,934,208,lr 3e-4),
fp32 主权重 + bf16 autocast,分块 CE(chunk 4096),batch 16 × accum 16 × 1024。
单张 RTX 4090,S0 跑了 2 天 19 小时。

## 底座的成绩(WMT22 5-shot / 1000 条 / vLLM)

| | zh→en BLEU / COMET | en→zh BLEU / COMET |
|---|---|---|
| Qwen3-1.7B-Base | 22.34 / 0.8122 | 38.34 / 0.8597 |
| ReTok v18_tie 1.7B | 20.46 / 0.7933 | 36.03 / 0.8444 |
| Qwen3-0.6B-Base | 17.66 / 0.7894 | 32.31 / 0.8369 |
| **Summer-0.5B-S1** | **8.99 / 0.6855** | **27.29 / 0.7743** |
| **Summer-0.5B-S0** | **0.54 / 0.4638** | **3.97 / 0.5872** |
| 换完词表未训(参考) | 6.22 / 0.2603 | 3.37 / 0.3542 |

### S0 的「零」是这条线最重要的观察

step 8000 时不看 BLEU 而是直接看生成:

```
zh-en   src: 是否有途径处罚他
        hyp: 在疫情防控期间,他被确诊为新冠肺炎,目前在武汉工作。
        ref: Is there a way to punish him?
```

流畅、合语法、话题连贯的中文 —— 但**完全无视 5-shot 示例,连输出语言都不对**。
语言模型学到了,in-context learning 没有。这是定性缺口,不是「再多跑几千步会
慢慢涨上来」的量。整个 S0(11.8B token)跑完仍然如此。

一上平行语料退火(1.2B token),同一句变成:

```
hyp: - No, no, no.                     ← step 42000,语言对了,内容不对
hyp: Or will the restaurant be ready in a few days?   ← 基本译对
```

**S0 → S1 是整条线里最大的单步变化**(COMET 0.4638 → 0.6855)。让 ICL 出现的
不是参数量也不是总 token 数,是那 30% 平行语料。

## 通用能力(nanochat 口径的五项)

对照有两列:**Qwen3-0.6B-Base 是关键那一列** —— 同尺寸、36T token、已知能力,
它量的是「同样大小的模型,预训练充分是什么样」。nanochat d20(560M / 约 11B
token)那一列只当量级参考,原因见下。

| | 随机 | nanochat d20 | Qwen3-0.6B-Base | Summer-0.5B-S0 |
|---|---|---|---|---|
| ARC-Easy (acc) | 0.25 | 0.3876 | 0.6549 | 0.5564 |
| ARC-Easy (acc_norm) | 0.25 | — | 0.5816 | 0.4949 |
| ARC-Challenge (acc) | 0.25 | 0.2807 | 0.3345 | 0.2466 |
| ARC-Challenge (acc_norm) | 0.25 | — | 0.3823 | 0.2671 |
| MMLU | 0.25 | 0.3151 | 0.5041 | 0.2520 |
| GSM8K (8-shot) | 0 | 0.0455 | — | 0.0121 |
| C-Eval(中文) | 0.25 | — | — | 0.2363 |

对 Qwen3-0.6B-Base 的差:ARC-Easy −0.099、ARC-Challenge −0.088、**MMLU −0.252**。

**MMLU 那个 −0.252 是最诚实的数。** 同样 0.5B 的模型,预训练充分的有 0.5041
的知识面,我们就是随机(0.2520)。ARC 上差不到 0.1 —— 基础语言理解学到了一些;
MMLU 上完全没有 —— 13B token 装不下知识。这就是 36T 对 13B 的代价。

C-Eval 0.2363 略低于随机属噪声。GSM8K 0.0121 在 8-shot 下,基本是零。

### 两套协议量的不是一回事

同一个 S0、同一批 ARC-Easy 题目,换个协议差 0.30:

```
似然打分(lm_eval acc)   0.5564    不给字母,比各选项**正文**的对数似然
字母 MC(nanochat)       0.2563    题面渲染成 MC,模型输出**一个字母**
```

两个都叫「ARC-Easy 准确率」,而且单看任何一个都不会觉得有问题。所以之前纠结
nanochat 报的是 `acc` 还是 `acc_norm` 是问错了问题 —— **两个都不是**。

它们量的能力也不同,所以 `prepare/` 里两套都留:

| | 量什么 |
|---|---|
| 似然打分 | 底座知识。SFT 前后都能测,但看不见 SFT 加了什么 |
| 字母 MC | **能不能听懂指令** —— 正是 midtrain + SFT 要加的能力 |

整条 chat 线做完之前,没有任何一个指标在量它要加的那个能力。

**实现之前先校验了协议。** 拿 nanochat 官方 d20 checkpoint 跑我们这套打分逻辑
(`prepare/mc_eval.py` 的 `render_mc` + 字母上 argmax,只换模型和分词器):

```
复现        0.3994  (949/2376)
官方 meta   0.4033           差 -0.0039
```

差的这点来自过滤掉 answerKey 非字母的少数行,以及 bf16 数值。**这一步是必须的**
—— 取 logits 的位置差一格、字母 token 前多个空格,都会给出一个看着合理的错数字。

读这套数还要知道:**nanochat 的 SFT 数据里就有 ARC 的 train split**
(`chat_sft.py:88-89`,ARC-Easy 2.3K + ARC-C 1.1K),它学过「MC 题面 → 输出字母」
这个格式本身。我们的混比里 ARC 是零。所以差距要分开看:格式会不会 / 知识有没有。

### 字母 MC 的结果:格式学会了,答案还是瞎猜

ARC-Easy,两种提示格式各考一遍(`prepare/mc_eval.py --render`):

```
                   nanochat 格式          我们训练用的格式
                 格式跟随    acc        格式跟随    acc
summer s0         0.0000   0.2563       0.0008   0.2357
summer sft v1        —        —         0.9566   0.2572
summer sft v2     0.0930   0.2584       0.9575   0.2647
nanochat d20      0.9996   0.3994          —        —
随机                         0.25                  0.25
```

「格式跟随」= 不加约束、在全词表上取 argmax,首选是不是候选字母之一。
**只看被约束后的准确率分不出两种失败** —— 模型压根不想输出字母(约束等于替它
瞎猜)和它想输出字母但选错了,数字可以一模一样。

读出来三件事:

1. **格式不是瓶颈。** 用我们自己训练时的渲染考,格式跟随 95.75%,和 nanochat
   的 99.96% 同一档。midtrain 里 MMLU_AuxTrain 占 0.58(约 60M token)确实
   教会了这件事。
2. **拿 nanochat 的格式考我们的模型只有 9.3%,那是训练/测试格式不匹配** ——
   两边差四处:有无 "Multiple Choice question:" 前缀、字母在选项前还是后、
   分隔符 ". " 还是 "="、有无 "Respond only with the letter" 那句。所以
   `--render` 两套都留,报数时必须写明用的哪套。
3. **格式修正之后准确率仍是 0.2647,随机是 0.25。** 这才是真正的差距:nanochat
   高出随机 0.15,我们高出 0.015。

把四项、两套协议并排看(都是 sft v2,字母 MC 用我们自己的渲染):

| | 似然 acc | 字母 MC acc | 格式跟随 | 随机 |
|---|---|---|---|---|
| ARC-Easy | **0.4941** | 0.2647 | 0.9575 | 0.25 |
| ARC-Chall | 0.2432 | 0.2509 | 0.9804 | 0.25 |
| MMLU | 0.2598 | 0.2601 | 0.9945 | 0.25 |
| C-Eval(中) | 0.2355 | 0.2419 | **0.3899** | 0.25 |

**四项里只有 ARC-Easy 有真信号,而且只在似然协议下有。** 其余三项两种协议下
都贴着随机 —— 那不是「知识接不到输出上」,是本来就没有。只有 ARC-Easy 一项
符合「知识存在但转不成字母」这个描述。

两条独立的发现:

- **MMLU 格式跟随 99.45%,准确率仍是 0.2601。** midtrain 里那 60M token 就是
  MMLU auxiliary_train —— **同一个任务族、同一套格式训过**,格式学到了,答案
  没有。同分布的格式训练不会顺带带来知识。
- **中文侧格式跟随只有 0.3899,英文是 0.96–0.99。** 我们的 MC 数据全是英文
  (MMLU auxiliary_train),中文那半(COIG-CQIA)没有一条选择题。「看到选项
  就答字母」只在英文语境里学会了。**这是配方漏洞,不是能力问题** —— 中文侧
  要有 MC 数据才谈得上测中文选择题。

nanochat 那 +0.15 有一部分来自它 SFT 直接训了 ARC 的 train split
(`chat_sft.py:88-89`);我们的 SFT 混比里选择题是零(MC 数据只在 midtrain,
而 midtrain 整段算 loss,「答字母」这个信号被问题文本稀释)。

### nanochat 那一列不能精确比

一开始我把 ARC-Easy 的 0.5564 当成「超过 nanochat 的 0.3876」,**那是错的**。

`acc` 和 `acc_norm` 是两个指标,而 nanochat 报的是哪个没有公开说明。实测在
Qwen3-0.6B-Base 上两者能差 0.07,且方向不一致:

```
                acc      acc_norm
arc_easy       0.6549    0.5816    ← acc 更高(选项短,未归一化偏向短选项)
arc_challenge  0.3345    0.3823    ← 反过来
```

光凭一个数字对不上号。所以那一列只当量级参考。另外我们走
`prepare/benchmark.py`(绑 piece 分词器),Qwen 走 lm_eval 自带的 vllm 后端 ——
**严格说这也是混后端**(`WHY.md` 第二节记过实测差 2.2 个点),所以上面那些差值
也该当量级看,不当精确值。

### 不实现 CORE

nanochat 的 base 报 CORE(DCLM 那 22 个任务的中心化准确率),其中若干个 lm_eval
没有等价配置(bigbench 的几个子任务、jeopardy、coqa 的切分)。凑一个「近似
CORE」会得到看着能比、其实不可比的数,比没有更糟。

## 通用 chat 那条线(midtrain → SFT)

S0 除了给 Interpreter 当翻译底座,还走了另一条:补上 nanochat 那套
midtrain + SFT,做成通用 chat 模型。配方与格式在
[`docs/DATA_FORMAT.md`](../DATA_FORMAT.md),这里只记结果。

```
S0(单语 11.8B)→ midtrain(对话 103M,整段算 loss)→ SFT(对话 12M,只算助手回复)
```

### 特殊 token 从「同一个向量」分化开了

S0 训完 11.8B token 后,`<user>` / `<assistant>` / `<end>` / `<pad>` 的余弦
**全是 +1.0000** —— 预训练语料里它们出现 0 次,四个是同一个向量。

| | user-assistant | user-end |
|---|---|---|
| S0 | +1.0000 | +1.0000 |
| + midtrain | +0.9625 | +0.9614 |
| + SFT | +0.9627 | +0.9606 |

**分化只发生在 midtrain。** SFT 推不动它们,原因是结构性的:`<assistant>` 落在
prefix 里(mask=0),拿不到输出侧梯度。这反过来说明 midtrain 这一段不能省。

幅度仍然很小(0.96,而普通 token 之间平均 0.285)—— 103M token 只够推这么多。

### 停止符换成专用 `<end>` 之后

`<eos>` 在预训练里是文档分隔符(占 0.094%,S0 全程约 1100 万次),含义是
「一篇结束,**后面还有下一篇**」;早先 SFT 拿它表示「回答结束,到此为止」,
两个含义直接冲突。改用专用的 `<end>`(81902,预训练里 0 次)之后:

| | P(停止符) 中位 | 排名中位 | 教师强制下能停 |
|---|---|---|---|
| 用 `<eos>` 训,1 epoch | 0.325 | 1 | 50.8% |
| 用 `<eos>` 训,3 epoch | 0.357 | 1 | 53.3% |
| **用 `<end>` 训** | **0.595** | **1** | **65.0%** |

多训 3 倍只带来 +2.5 点,换专用 token 带来 **+14 点** —— 瓶颈确实在含义冲突,
不在训练量。

### 但自由生成还是停不下来,原因是复读

### 自由生成的停止率:和 nanochat d20 的直接对照

同一批 20 条英文 prompt、贪心解码、各用自己的停止符:

```
nanochat d20     95%(19/20)  平均生成  48 token
summer v2        65%(13/20)  平均生成 385 token   预算 600
summer v2        10%( 2/20)  平均生成 197 token   预算 200 ← 判据不当
```

预算 200 那个 10% 是**我的判据造成的假象**:我们 SFT 数据答案中位 338 token,
只有 20.6% ≤200,天花板就是 20.6%。放到 600 之后 65%,和教师强制的 65.0%
精确吻合,平均生成 385 ≈ 训练答案平均 374 —— 模型忠实复现了训练分布。

> **量停止率之前先看训练答案的长度分布。** 预算低于分布时测到的是「被截断」,
> 不是「不会停」。我先按 200 测出 10%,据此写下「瓶颈是复读、根因是预训练
> token 太少」—— 两句都是错的,而且错得看起来很合理。

nanochat 那 30 个点的领先里有一部分是配方:它的 SFT 混比一大半是短答案任务
(ARC / GSM8K / 拼写),我们一半是 COIG-CQIA 的长答(中位 1126 token)。

**剩下的差距不是预训练规模。** nanochat d20 是 561M / 约 11B token,和 S0
同一档。

后来的进展改了这一段的结论,详见 [`../POSTTRAIN.md`](../POSTTRAIN.md):

- 「头号嫌疑是输出侧学习率」的证据被推翻了。原先引的「`<end>` 位移只排
  9921/81903」是读反的 —— 降序排名 9921 是**前 12%**,动得比 88% 的 token 都多;
  `<pad>` 出现 0 次也排前 16%(tie 让 softmax 把所有行当负类压)。
  **位移大不等于学到了。**
- 真正量到的是:**打包方式**才是大头。midtrain 从 best-fit 改成连续流之后,
  中文停止率 29% → 69%、中文格式跟随 0.4657 → 0.7834,英文侧几乎不动。
  机制仍未解释。

关于 nanochat 这个 checkpoint 怎么才跑起来:官方 d20 chatsft 权重是干净的
(122 张量 / 560.99M),卡住的是**代码版本** —— 仓库 HEAD 已经是新架构,多出
`resid_lambdas` 等 25 个键。`git fetch --unshallow` 之后退到引入那次提交的
父提交,`load_state_dict(strict=True)` 全键匹配。**社区移植版没去修** ——
它的 rotary 有布局 bug,补错了就输出垃圾,而垃圾永远不会停,会得到一个假的低
停止率。

## 走完下游之后

同一条 SFT → CPO → GRPO(与两个 1.7B 底座 31 项超参完全一致)之后:

| | 参数 | 预训练 | zh→en BLEU / COMET | en→zh BLEU / COMET |
|---|---|---|---|---|
| Interpreter-Qwen3-1.7B | 1.7B | 36T | 20.31 / 0.8053 | 33.69 / 0.8540 |
| Interpreter-Qwen3-1.7B-ReTok | 1.7B | 36T | 19.29 / 0.7986 | 32.46 / 0.8512 |
| **Interpreter-Summer-0.5B** | **0.52B** | **13B** | **14.29 / 0.7460** | **30.93 / 0.7978** |

参数量差 3.3 倍、预训练 token 差 2,700 倍,COMET 差 0.053。**译文流畅但不忠实,
不到可用水平** —— 完整分析和排除掉的方向都在 Interpreter 那份记录里,
这里不重复。

## 发布

| | |
|---|---|
| [`Ismantic/Summer-0.5B-S0`](https://huggingface.co/Ismantic/Summer-0.5B-S0) | 纯单语底座。5-shot 基本为零,是 S1 的对照与后续 midtrain 的起点 |
| [`Ismantic/Summer-0.5B-S1`](https://huggingface.co/Ismantic/Summer-0.5B-S1) | 平行语料退火过的底座 |

发布包只依赖 torch + PieceTokenizer,自带 `model.py` / `tokenizer.py` /
`example_load.py`。权重 bf16(约 1GB);**fp32 主权重留在 `output/`** ——
那是训练的需要(见 `WHY.md` 第四点五),不是发布的需要。

## 复现

```bash
make -C prepare init-scratch      # 随机初始化的起点
make -C prepare encode-scratch    # 12B token / 中英各半 / 分 shard
make -C prepare s0-service        # 装成 systemd 服务,开机自动续训
make -C prepare encode-anneal-mt  # 带平行语料的退火集
make -C prepare s1 STEP=40000     # 从 step 40,000 分叉
```

## 机器

单张 RTX 4090(功耗上限 350W)+ Ryzen 7 7700 + Manjaro / 内核 6.1。
S0 期间三次非训练原因的中断,累计白跑约 8.4 小时:两次外部信号打断,一次
硬重启(2026-08-02 05:36,掉电签名 —— wtmp 记 crash、journal 无 shutdown
序列、pstore 空)。之后才装的 systemd 服务 + 开机自动续训 + `health.csv`
状态采样。上传阶段又遇到 WiFi 间断掉线,三次上传两次死在
`[Errno 101] Network is unreachable`,最后靠幂等重试传完。
