# Summer-0.5B:从零训一个 0.5B 双语底座

2026-08-01 ~ 08-12。与 ReTok 那条线(给 Qwen3-1.7B-Base 换词表)**无关** ——
这是另一条线:不借任何现成权重,随机初始化开始训。

这份只记**底座怎么训出来的**。它走完下游翻译流水线的结果在
[Interpreter 的 `docs/reports/summer-0.5b-downstream.md`](https://github.com/Ismantic/Interpreter/blob/main/docs/reports/summer-0.5b-downstream.md)
——那三个「不报错只给错结果」的 bug 也在那边,因为出问题的代码在那个仓库。

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

## 走完下游之后

同一条 SFT → CPO → GRPO(与两个 1.7B 底座 31 项超参完全一致)之后:

| | 参数 | 预训练 | zh→en BLEU / COMET | en→zh BLEU / COMET |
|---|---|---|---|---|
| Interpreter-Qwen3-1.7B | 1.7B | 36T | 20.31 / 0.8053 | 33.69 / 0.8540 |
| Interpreter-Qwen3-1.7B-ReTok | 1.7B | 36T | 19.29 / 0.7986 | 32.46 / 0.8512 |
| **Interpreter-Summer-0.5B** | **0.52B** | **13B** | **14.29 / 0.7460** | **30.93 / 0.7978** |

参数量差 3.3 倍、预训练 token 差 2,700 倍,COMET 差 0.053。**译文流畅但不忠实,
不到可用水平** —— 完整分析、三个 bug、以及排除掉的方向都在 Interpreter 那份
记录里,这里不重复。

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
