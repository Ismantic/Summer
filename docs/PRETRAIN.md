# 从零重做一遍

把 `Qwen/Qwen3-1.7B-Base` 换成 81903 piece 词表,两阶段恢复。全程只需要
Hugging Face 和 GitHub,不依赖任何本机既有文件。

**先说清楚:复现 v18 的 checkpoint 不需要这一整套。** 权重已经发布在
`Ismantic/Qwen3-1.7B-Base-ReTok`,`make test` 钉的是 checkpoint 的行为。
这份文档是给「想自己重跑一遍」或「想换个配方」的人看的。

## 时间和磁盘

单张 RTX 4090。

| 步骤 | 时间 | 磁盘 |
|---|---|---|
| `make deps` | 几分钟 | ~200MB |
| `make -C data download` | 数小时 | **100GB+** |
| `make -C prepare retok` | ~10 分钟 | 3.2GB |
| `make -C prepare encode` | 数小时 | 4.6GB(两段 `.pt`) |
| `make -C prepare p1` | **25.1 小时**(3815 步,23.7 秒/步) | 3.2GB |
| `make -C prepare p2` | **5.2 小时**(1500 步,12.4 秒/步) | 3.2GB |

两阶段合计 **约 30.3 小时 / 1.2B token**。这两个数字来自 v18 的实际训练日志
(`output/v18_p1_train.log` 的 90461 秒、`v18_p2_tie_train.log` 的 18645 秒),
不是估的。

## 显存:原配方要 48GB,4090 装不下

`prepare/Makefile` 里的 p1 / p2 用 `--batch_size 16`,那是 v18 在 **A6000
(48GB)** 上训的配方 —— **原样保留是因为它就是产出已发布权重的那份配方**。

在 24GB 的 4090 上会 OOM。失败的分配约 5GB,正是 loss 里那个
`logits.float()`:`16 × 1023 × 81903 × 4 字节 = 5.4GB`。

按**等效 batch 不变**换算即可(v18 是 16×16=256):

```bash
make -C prepare p1 BATCH=4 ACCUM=64     # 4×64 = 256,与 v18 等效
make -C prepare p2 BATCH=4 ACCUM=32     # 4×32 = 128,与 v18 的 16×8 等效
```

实测 4090 上 p1 约 27 秒/步、p2 约 14 秒/步 —— 比 A6000 慢一些,但跑得动。

语料那 100GB 是**池子**,不是消耗量 —— 两段训练总共只吃 1.2B token。想省磁盘
就调小 `data/source.py` 里的 `n_parts`,只要池子 ≥ 消耗量就行,区别只是采样
多样性。各源的实际消耗量列在那个文件的注释里。

## 0. 依赖

```bash
make deps
```

clone 并编译 [PieceTokenizer](https://github.com/Ismantic/PieceTokenizer)。
**81903 词表和中文分词词典都在它仓库的 `save/` 下**,本仓库不留副本 ——
代码用 `prepare.tokenizer.resolve_assets()` 反查。

本机的解释器路径写进 gitignore 的 `local.mk`:

```makefile
PY      = /path/to/venv-3.14/bin/python     # 训练,只要 torch
```

**一个 3.11 的 venv 就够**(训练 + 评测)。曾经拆成两个是因为 vllm 和 comet
上不了 3.14 —— 但把训练搬到 3.11 就都解决了。原因和验证见
`WHY.md` 第五节。

## 1. 数据

```bash
make -C data status      # 每个源下了没
make -C data probe       # 每源只下一个文件并真读一遍,验注册表写对没
make -C data download    # 下全部
```

**改了 `source.py` 之后跑 `probe`,不要跑 `download`。** 会写错的只有
glob / fmt / text_field 这三样,一个文件就能暴露。

`BAAI/CCI3-HQ` 是 gated 的,要先去它的 HF 页面接受条款再
`huggingface-cli login`。这是全流程唯一的手动步骤。

数据源的出处和溯源依据在 `data/source.py` 开头 —— 十个源里六个是 sha256 或
首条记录确证的,三个是有意替换的(原始出处不可得),都写明了。

## 2. 词表手术

```bash
make -C prepare retok
```

`prepare/retok.py` 做的事:对新词表的每个 piece,用旧 BBPE 编码它的文本,
拿对应的旧嵌入初始化 ——

实测(81903 词表 vs Qwen 151643 BBPE,2026-07-27 实跑):

| | 占比 | 处理 |
|---|---|---|
| 一对一 | **66.4%**(54343) | 直接用那一行 |
| 多对一 | 32.5%(26658) | 取平均 |
| 编不出来 | 1.1%(895) | 用全词表均值兜底 |

老文档里的「~73.5% 一对一」是 65007 词表时代的数,词表变大之后需要重学的行
更多了 —— 这也是 Phase 1 存在的理由。

**这一步不可逆。** 嵌入映射一改,所有已有 checkpoint 全部失配 —— 而且模型
照样能加载、照样跑。已发布的权重就是按当前规则产出的。

顺带写出 `frozen_ids.json`(一对一映射的行)。那些行的初始化是精确的,
Phase 1 可以冻结它们、只训真正需要重学的约 26%。

## 3. 预编码

```bash
make -C prepare encode
```

两段:

| | token | 混合 | 用途 |
|---|---|---|---|
| `--mix main` | 1B | EN 0.663 / CN 0.336,8 源 | Phase 1 |
| `--mix anneal` | 200M | EN 0.60 / CN 0.40,6 源 | Phase 2 退火 |

产出 `[N, 1024]` 的 int32 张量。权重表在 `prepare/encode_corpus.py`,
路径/格式/字段全部来自 `data/source.py` —— 这个文件里不出现任何本机路径。

## 4. Phase 1:冻结 transformer

```bash
make -C prepare p1
```

只训 `embed_tokens` + `lm_head`(tie,同一份权重)。换词表后约 26% 的嵌入行是
平均或兜底初始化的,先把它们学出来,**不动原始 Qwen 权重**。

1B token / 3815 步 / `adam_lr 1e-4` / cosine 到 `min_lr_ratio 0.01`。

**`min_lr_ratio` 不能是 0**,衰减到 0 是已知的病态。

## 5. Phase 2:LoRA tie-safe

```bash
make -C prepare p2
```

transformer 走低秩旁路(`q_proj`/`v_proj`,r=16,α=32),`embed_tokens` 全参训,
`lm_head` 因 tie 自动同步。200M 高质量 token 退火 / 1500 步。

**全参数解冻在 0.6B 时代是破坏性的**(v7–v16 都失败了),LoRA 才是跑通的配方。

优化器:LoRA 参数走 Aurora,embed 走辅助 AdamW。注意 **Aurora 在这个配置下
没有做过消融** —— 唯一那次对照是在 0.6B 全参数上做的,结论是打平。见
`WHY.md` 第四节。

训完自动把 LoRA 合并回基座,存成标准 HF 布局(310 个张量,没有 `lm_head`)。

## 6. 评测

```bash
make -C prepare trans CKPT=output/phase2 TAG=myrun   # WMT22 BLEU + COMET
make -C prepare bench CKPT=output/phase2 TAG=myrun   # mono 六任务
python prepare/sweep.py table --tags base_1.7b myrun # 出对照表
```

看数字之前先读 `WHY.md` 第二节:**vLLM 贪心解码不可复现,BLEU 跑间 range
约 0.1;两个后端的数字不能混比。**

## 7. 发布

```bash
make -C save export      # 导出干净的 HF 上传目录
make -C save verify      # 上传后核对 sha256
```

发布包里 **`Summer-Tokenizer.dict.txt` 必须带** —— 少了它中文的 token id
会变,而下载的人不会发现(round-trip 照样正确)。

发布包**自足**:带 `model.py` / `checkpoint.py` / `tokenizer.py`,所以下载的人
只要 `torch` + `PieceTokenizer`,不需要 transformers 也不需要 safetensors 库。

## 出了问题先查

`WHY.md` 第一节列了所有「改错了不报错」的地方。最常见的三个:

1. loss 明显偏高但不报错 → 查 RoPE 是不是被 autocast 降精度了
2. 中文效果异常 → 查中文分词词典在不在(`Summer-Tokenizer.dict.txt`,
   旧 ckpt 里叫 `dict.txt`)
3. 权重像是没加载 → 查 state_dict 的 key,`from_pretrained` 会报错但
   `load_state_dict(strict=False)` 不会
