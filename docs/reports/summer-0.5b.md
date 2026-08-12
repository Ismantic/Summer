# Summer-0.5B:从零训一个 0.5B 双语模型,再走完下游翻译流水线

2026-08-01 ~ 08-12。与 ReTok 那条线(给 Qwen3-1.7B-Base 换词表)**无关** ——
这是另一条线:不借任何现成权重,随机初始化开始训。

## 问的是什么

ReTok 那条线证明了「换掉词表还能把能力恢复回来」。但它的能力仍然来自
Qwen 的 36T token。所以剩下一个更硬的问题:

> **完全自己训的底座,走完同一条下游流水线,能到哪里?**

判据定在下游(`../Interpreter` 的 SFT → CPO → GRPO 产出的翻译质量),
不是 5-shot BLEU —— 后者只是途中的体温计。

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

## 下游(WMT23 / prepare/evaluate.py / 同协议)

| 节点 | zh→en BLEU / COMET | en→zh BLEU / COMET |
|---|---|---|
| S1 底座 5-shot | 9.25 / 0.6659 | 27.27 / 0.7525 |
| + SFT(bf16,**有 bug**) | 4.28 / 0.5518 | 10.59 / 0.5936 |
| + SFT(fp32,修复后) | 12.19 / 0.7103 | 30.49 / 0.7785 |
| + CPO | 14.58 / 0.7351 | 32.56 / 0.7915 |
| + GRPO 512 prompt | 14.68 / 0.7396 | 33.96 / 0.7975 |
| + GRPO 2560 | 14.43 / 0.7436 | 32.59 / 0.7975 |
| **+ GRPO 5120(发布的那个)** | **14.29 / 0.7460** | **30.93 / 0.7978** |
| + GRPO 6000 | 14.03 / 0.7448 | 30.55 / 0.7982 |

对照(同测试集、同协议、同 COMET 模型):

| | 参数 | 预训练 | zh→en | en→zh |
|---|---|---|---|---|
| Interpreter-Qwen3-1.7B | 1.7B | 36T | 20.31 / 0.8053 | 33.69 / 0.8540 |
| Interpreter-Qwen3-1.7B-ReTok | 1.7B | 36T | 19.29 / 0.7986 | 32.46 / 0.8512 |
| **Interpreter-Summer-0.5B** | **0.52B** | **13B** | **14.29 / 0.7460** | **30.93 / 0.7978** |

**en→zh 的 BLEU 一度超过 1.7B 的 ReTok 模型**(GRPO 512 时 33.96 vs 32.46),
但 COMET 始终差约 0.053。BLEU 反超而 COMET 落后,说明产出的是「词面像、
意思不够准」的译文 —— 下一节有实例。

## 三个 bug,都属于「不报错,只是悄悄给出错误结果」

### 1. bf16 持参把小 lr 的更新舍掉(影响最大)

`Interpreter/src/train.py` 原来直接以 bf16 加载并训练,没有 fp32 主权重。
bf16 在 1.0 附近的最小间隔是 2⁻⁸ = 0.0039,而 RMSNorm 的权重初值正是 1.0,
lr 2e-5 下一步的更新约 1e-5 —— **每次更新都被舍入回精确的 1.0**。

```
                RMSNorm 动了    q_proj 相对位移    SFT 结果
bf16              0/113          0.00021         4.28 / 10.59
float32          24/113          0.00115        12.19 / 30.49
v18_tie 对照     73/113          0.00177        (它底座强,本来只需微调)
```

同数据、同超参、同步数,**只差参数精度,BLEU 差 2.85 倍**。

v18_tie 时代没暴露:1.7B / 36T token 的底座本来只需要微调,它也有 40 个 norm
被冻着,只是不影响结果。换成一个真正需要被训动的底座就致命。

讽刺的是这个坑 Summer 自己的 `WHY.md` 里就记着(预训练用 `--param_dtype
float32` 正是为它),Interpreter 那一层没有。

### 2. GRPO 的超参不能照搬

原配置 lr 1e-6 / β 0.04 在这个模型上**跑 3000 个 prompt 毫无变化**。探边界:

```
lr 1e-6  / β0.04     rollout COMET 0.6553 → 0.6685    平
lr 1e-5  / β0.01                  0.6576 → 0.6840
lr 2e-5  / β0.005                 0.6572 → 0.7089    最快,边界内最优
lr 5e-5  / β0.002                 0.6553 → 0.6587    第 4 轮发散,KL 冲到 11.4
```

**这四次用的是同一批 prompt 分组**,所以逐轮对比有效。`WHY.md` 里早记着
「从 SFT 直起那条路径照搬 CPO 路径的 lr/β 会停滞」,同一条经验在换底座时同样
适用 —— 我们的模型离最优点更远,需要更大的步长和更松的 KL 约束。

诊断过程里我错了一次:先看到 rollout COMET 十轮不涨就判定「GRPO 空转」,
但**每轮用的是不同的 prompt**,轮间差异主要反映那批句子有多难,不是策略进步。
那个证据不成立。有效的判据只有固定测试集上的评测。

### 3. CPO 的偏好数据从来不是 on-policy

`data/gen_candidates.py` 用 `AutoTokenizer`,**加载不了 piece 分词器**。查遍
git 历史,piece 版的候选生成脚本从来没存在过 —— 当年 ReTok 线把 SFT/CPO/GRPO
三个**训练**脚本移植到了 `PieceTokenizerWrapper`,但候选生成没移植,它读的是
Qwen 线那份偏好数据。

这个缺口不会报错:偏好数据是纯文本 jsonl,共享它根本不经过分词器。于是
`data/README.md` 和 `WHY.md` 里「主线自生成(on-policy)」这个标签,实际上
只对 Qwen 线成立 —— 而那两处正是 WHY.md 自己测出「离策略偏好数据差约
0.006 COMET」的地方。已发布的 ReTok 模型在 CPO 这一步是被这么打了折的。

新写了 `Interpreter/prepare/gen_candidates.py`(vLLM + PieceTokenizerWrapper),
Summer-0.5B 这条线的 CPO 是这个项目第一次真 on-policy。效果与 v18_tie 那次
明显不同:**BLEU 与 COMET 同涨**(+2.39/+2.07 BLEU,+0.025/+0.013 COMET),
而 v18_tie 那次 CPO 是拿 BLEU 换 COMET 的(en→zh BLEU 从 40 砸到 31)。

## 排除掉的方向

### SFT 跑够没有用

SFT 从 1 epoch 加到 3 epoch:

```
             末段 loss    SFT 后 COMET        CPO 后 COMET
1 epoch        1.922     0.7103 / 0.7785    0.7351 / 0.7915
3 epoch        1.920     0.7142 / 0.7798    0.7318 / 0.7892
```

**多跑两个 epoch,loss 只从 1.922 降到 1.920。** SFT 阶段涨 0.004,走完 CPO
反而低 0.003。这条路封死。

原因也清楚:SFT 数据只有 36,800 条 / 302 万 token,是预训练的 **0.07%**。
它能做的是把已有能力对齐到任务格式 —— 已经做完了,加倍它没有增量。

### GRPO 加大数据量收益递减

```
prompt    zh-en COMET   增量      en-zh COMET   en-zh BLEU
  512       0.7396        —         0.7975        33.96
 1280       0.7425     +0.0029      0.7969        33.32
 2560       0.7436     +0.0011      0.7975        32.59
 3840       0.7442     +0.0006      0.7977        31.12
 5120       0.7460     +0.0018      0.7978        30.93
 6000       0.7448     −0.0012      0.7982        30.55
```

**en→zh 的 COMET 在 512 个 prompt 时就到顶了,之后十一倍数据一动不动**,
而 BLEU 掉了 3.4 分。GRPO 的奖励就是 COMET,它在系统性地拿词面重合换语义
忠实 —— 这不是 bug,是奖励函数的定义。zh→en 的 COMET 还在慢慢爬,但每 1280
个 prompt 只买到约 0.002,到 6000 时开始回落。

## 质量:不到可用水平

数字之外看译文。同一个 checkpoint(GRPO 5120)的实际输出:

```
好的:
src: 增加了本文件适用对象(见第 1 章),
hyp: The scope of the application of this document has been expanded (see chapter 1),

致命的:
src: The hacked up version of Jedi Knight was crashing because it was calling
     a function off the end of a vtable.
hyp: 被攻击的《钢铁侠》版本正在崩溃,因为它正在向 vtable 的末尾发起一个功能请求。
     ↑ Jedi Knight → 《钢铁侠》(凭空换了作品)、hacked up → 被攻击、
       calling a function → 发起功能请求

丢信息的:
src: ...only creating the viewport using a Direct3D object, not a Direct3D3 object
hyp: 这仅仅是创建了一个视图,而不是一个 Direct3D3 对象
     ↑ "using a Direct3D object" 整段丢失,逻辑变了;viewport 译成「视图」
```

```
流畅度   好。中英两侧都读得通,语法自然
忠实度   不可靠。实体替换、成分丢失、术语错译
```

**问题不在于错得多,而在于错得看不出来。** 它产出的是自信、通顺、看起来
专业的译文,读者无法从译文本身判断哪句可信 —— 这比明显崩坏的输出更危险。

对应的量化证据在底座上做过(1000 条,n-gram precision 衰减):

```
                    1-gram  2-gram  3-gram  4-gram   1→4 衰减
Summer-0.5B-S1 zh-en  35.1    12.6    5.6     2.6      13.5×
Qwen3-0.6B-Base zh-en 51.0    23.3   12.2     6.7       7.6×
ReTok-1.7B zh-en      55.9    27.5   15.2     8.7       6.4×
```

1-gram 只差 1.5 倍,4-gram 差 2.6 倍 —— **词选对了,连不成正确的长片段**。
这与「流畅但不忠实」是同一件事的两种测法。

顺带纠正一个我自己下早了的结论:最初用 5 条样本算出 brevity penalty ≈ 0.52,
判断是「输出太短」。1000 条上重算是 hyp 平均 27.0 词 vs ref 26.3 词,
**BP = 1.000,完全没有欠生成**。5 条样本的统计不能当判据 —— 这也是后来给
`eval_hook.py` 加 `--save_all_samples` 的原因。

## 结论

**13B token 买到的是「会翻译但不可信」。**

```
达到       en→zh BLEU 一度超过 1.7B / 36T token 的模型
没达到     COMET 始终差 0.053;译文不忠实,不能给别人看
花掉       约 4 天 GPU(S0 2天19小时 + S1 18小时 + 下游若干)
```

参数量差 3.3 倍、预训练 token 差 2,700 倍,而 COMET 只差 0.053 —— 这个比例
说明下游流水线(尤其是 CPO 和 GRPO)能补回相当多东西,但补不上底座的忠实度。

**真正可复用的收获是那三个 bug。** 它们与训练量无关,在强底座上被余量掩盖着,
换个弱底座才暴露。修完之后 v18_tie 那条线也受益(它有 40 个 RMSNorm 也被
bf16 冻着)。

## 发布

| | |
|---|---|
| [`Ismantic/Summer-0.5B-S0`](https://huggingface.co/Ismantic/Summer-0.5B-S0) | 纯单语底座。5-shot 基本为零,是 S1 的对照与后续 midtrain 的起点 |
| [`Ismantic/Summer-0.5B-S1`](https://huggingface.co/Ismantic/Summer-0.5B-S1) | 平行语料退火过的底座 |
| [`Ismantic/Interpreter-Summer-0.5B`](https://huggingface.co/Ismantic/Interpreter-Summer-0.5B) | 走完 SFT→CPO→GRPO 的翻译模型 |

发布包只依赖 torch + PieceTokenizer,自带 `model.py` / `tokenizer.py` /
`example_load.py`。权重 bf16(约 1GB);fp32 主权重留在 `output/`,那是训练的
需要,不是发布的需要。

## 复现

```bash
make -C prepare init-scratch      # 随机初始化的起点
make -C prepare encode-scratch    # 12B token / 中英各半 / 分 shard
make -C prepare s0-service        # 装成 systemd 服务,开机自动续训
make -C prepare encode-anneal-mt  # 带平行语料的退火集
make -C prepare s1 STEP=40000     # 从 step 40,000 分叉
```

下游在 `../Interpreter`:

```bash
RECIPE=summer05b_s1 bash prepare/pipeline.sh
```

`recipe/summer05b_s1.sh` 里三处 `# ab-deviation` 标注就是上面那三个 bug 的
修复(fp32 持参 × 2、GRPO 超参、on-policy 候选生成),`test/test_recipe_ab.py`
保证其余 31 项超参与两个 1.7B 底座完全一致。

## 机器

单张 RTX 4090(功耗上限 350W)+ Ryzen 7 7700 + Manjaro / 内核 6.1。
S0 期间三次非训练原因的中断,累计白跑约 8.4 小时:两次外部信号打断,一次
硬重启(2026-08-02 05:36,掉电签名 —— wtmp 记 crash、journal 无 shutdown
序列、pstore 空)。之后才装的 systemd 服务 + 开机自动续训 + `health.csv`
状态采样。上传阶段又遇到 WiFi 间断掉线,三次上传两次死在
`[Errno 101] Network is unreachable`,最后靠幂等重试传完。
