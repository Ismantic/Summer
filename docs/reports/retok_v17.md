# 用新词表重做 ReTok（v17）

把 `Qwen3-0.6B-Base` 的分词器换成新的 piece 词表（81903），再用 v15/v16 的配方做
两阶段继续预训练。本文档记录完整步骤、命名约定、以及当前环境缺失的输入。

新 run 统一记为 **v17**（= 新词表 + v15 Phase 1 配方 + v16 Phase 2 配方）。

---

## 0. 当前环境状态（重要）

这个工作目录目前基本只剩**代码**，v15/v16 脚本假设存在的东西大多已不在：

| 需要的输入 | 状态 |
|---|---|
| `~/new/Qwen3-0.6B-Base`（base 权重） | ❌ 不存在（在另一台机器，需重新下载） |
| 训练语料（`/mnt/data/...`、`~/Data/...`） | ❌ `/mnt/data` 整个不存在，需重新下载 |
| `output/`（预处理 .pt、所有 checkpoint） | ❌ 目录不存在 |
| 新词表 `piece.model` | ✅ `~/Shiyu/PieceTokenizer/scripts/output/piece.model`（81899） |
| `dict.txt`（中文分词词典） | ✅ Summer 目录内已有，可复用 |

所以这不是"重跑几个脚本"，而是**从零重建**：重新下模型、下语料、重新预处理、重新训练。
重新预处理是必须的——词表变了，token id 全变，旧的 `.pt` 即使有也作废。

---

## 1. 路径与命名约定

| 用途 | 路径 |
|---|---|
| 新分词器模型 | `~/Shiyu/Summer/piece_v2.model`（已生成，见 Step 1） |
| 换分词器后的模型目录 | `~/new/Qwen3-0.6B-Base-new-tok-v2` |
| Phase 1 训练数据 | `output/phase1_train_512_v17.pt` |
| Phase 2 anneal 数据 | `output/v17_anneal_512.pt` |
| Phase 1 checkpoint | `output/phase1_ckpt_v17` |
| Phase 2 checkpoint | `output/phase2_ckpt_v17` |

旧的 65007 词表产物（`piece_mt.model`、旧 `-new-tok`）一律不动，新 run 全部用 `v17` / `v2` 后缀，
避免覆盖。

---

## Step 1 — 修复新词表的特殊 token ✅（已完成）

**问题**：`tools/replace_tokenizer.py:134` 硬性要求新词表里有 `<pad>/<user>/<assistant>/<system>`
四个特殊 token。你这版 `piece.model`（81899）建词表时没带 `EXTRA_TOKENS`，只有
`<unk>/<s>/</s>`，直接跑会崩。

**原理**：`.model` 是纯文本（`[CounterSpec]`/`[NormalizerSpec]`/`[Pieces]` 三段），piece 里的
`\t \n \r \` 在序列化时已转义成 `\xHH`（`piece_spec.h:12`），所以按行处理绝对安全。
`count --extra-tokens` 的做法（`piece_counter.cc:144-150`）就是在词表末尾追加 type=3、
score=0 的 CONTROL piece。**直接在文件层面追加这 4 行即可，无需对 21GB 语料重新训练。**

为此写了 `PieceTokenizer/scripts/add_extra_tokens.py`。已执行：

```bash
cd ~/Shiyu/PieceTokenizer/scripts
python add_extra_tokens.py \
    --input  output/piece.model \
    --extra-tokens "<pad>,<user>,<assistant>,<system>" \
    --output ~/Shiyu/Summer/piece_v2.model
```

产出 `piece_v2.model`：vocab 81903，`<pad>=81899 <user>=81900 <assistant>=81901
<system>=81902`，`pad_id=81899`。已验证可正常加载、编解码无误，与"带 `--extra-tokens`
重新 count"的结果逐字节一致。

---

## Step 2 — 下载 base 模型

`Makefile` 的 `make download` 会把 `Qwen/Qwen3-0.6B-Base` 拉到
`~/new/Qwen3-0.6B-Base`：

```bash
cd ~/Shiyu/Summer
make download         # 或 HF_ENDPOINT=https://hf-mirror.com make download
```

约 1.2GB。下完确认目录里有 `model.safetensors` / `config.json` / `tokenizer.json`。

---

## Step 3 — 换分词器（不可逆手术）

```bash
python tools/replace_tokenizer.py \
    --old_model_path     ~/new/Qwen3-0.6B-Base \
    --new_tokenizer_path ~/Shiyu/Summer/piece_v2.model \
    --output_path        ~/new/Qwen3-0.6B-Base-new-tok-v2

# 关键：复制【词表训练时用的那个】中文分词词典 —— 必须是 PieceTokenizer/dict.txt
# （359987 行，md5 2225d23），不是 Summer/dict.txt（320000 行，内容不同）。
cp ~/Shiyu/PieceTokenizer/dict.txt ~/new/Qwen3-0.6B-Base-new-tok-v2/dict.txt
```

`tools/replace_tokenizer.py` 会：为 81903 个 piece 逐个用 Qwen BBPE 编码、取均值映射到旧 embedding
（一对一 / 多对一 / 均值兜底）；resize 模型 embedding 到 81903；把 `piece_v2.model` 复制成目录里
的 `piece.model`，写出 `token_mapping.json`。

> 注意：新词表 81903 比旧的 65007 大 ~26%，需要在 Phase 1 学习的多对一/兜底行更多，
> Phase 1 的负担略重。embedding 矩阵约 81903×1024，模型体积略增，无妨。

---

## Step 4 — 下载并预处理训练语料

### 4a. 需要哪些数据集

Phase 1 用 v15 的混合（即"v8 mix" = `data_prep/pretokenize_v7.py` 的 10 个源），Phase 2 anneal 用
`data_prep/pretokenize_v12.py --mix anneal` 的 6 个源。两者并集：

| 数据集 | 用于 | 现成下载脚本 |
|---|---|---|
| FineWeb-Edu (sample-10BT) | P1+P2 | `data_prep/download_data.py` |
| Wikipedia EN / CN | P1+P2 | 无，需自备 |
| Gutenberg | P1 | 无 |
| C4 EN / CN | P1 | 无 |
| CnnDailyMail / PeopleDaily | P1 | 无（旧为本地 txt） |
| SkyPile-150B | P1 | `data_prep/download_data.py` |
| CCI3-HQ | P1 | 无 |
| Cosmopedia-v2 | P2 | `data_prep/download_cosmopedia.py` |
| Chinese-FineWeb-Edu V2.2 | P2 | `data_prep/download_chinese_fineweb_edu.py` |

**两个选项**：

- **严格复现 v15**：凑齐上面 v7 的全部 10 个源。最忠实，但部分源（Wikipedia/Gutenberg/C4/
  CnnDailyMail/PeopleDaily/CCI3-HQ）没有现成下载脚本，要自己找。
- **务实替代**：Phase 1 直接用 `data_prep/pretokenize_v12.py --mix main`（9 个源，下载脚本齐全）。
  注意 v15 之所以比 v14 好就是因为用了 v7 mix 而非 v12 mix——换 v12 main mix 是近似，不是
  严格复现，但工程上省事很多。

> 下载后，**必须把 `data_prep/pretokenize_v7.py` / `data_prep/pretokenize_v12.py` 里 `SOURCES` 的 glob 路径
> 改成数据实际落地的新路径**（旧路径 `/mnt/data/...` 全部失效）。

### 4b. 预处理（用新分词器！）

```bash
# Phase 1：2B token 池（v7 mix），约 3 小时
python data_prep/pretokenize_v7.py \
    --tokenizer_model ./piece_v2.model \
    --cn_dict ./dict.txt \
    --output output/phase1_train_512_v17.pt \
    --total_tokens 2000000000

# Phase 2 anneal：~200M token
python data_prep/pretokenize_v12.py --mix anneal \
    --tokenizer_model ./piece_v2.model \
    --cn_dict ./dict.txt \
    --output output/v17_anneal_512.pt \
    --total_tokens 200000000
```

磁盘：2B token 的 int32 `.pt` 约 8GB；`/home` 当前剩 ~242G，够用。

---

## Step 5 — Phase 1 训练（照搬 v15 配方）

冻结 transformer，只训 `embed_tokens`+`lm_head`，7500 步、cosine、`min_lr_ratio 0.01`。
新建 `runs/run_v17_p1.sh`（从 `runs/run_v15.sh` 复制，改三个路径）：

```bash
NEW_TOK=~/new/Qwen3-0.6B-Base-new-tok-v2
TRAIN_PT=output/phase1_train_512_v17.pt
V17_P1=output/phase1_ckpt_v17

torchrun --nproc_per_node=2 --master_port=29517 train/finetune_muon.py \
    --model_path $NEW_TOK --train_data $TRAIN_PT --mode clm \
    --freeze_transformer --output_dir $V17_P1 \
    --max_seq_length 512 --batch_size 32 --gradient_accumulation_steps 4 \
    --max_steps 7500 --warmup_steps 1000 \
    --adam_lr 1e-4 --min_lr_ratio 0.01 --lr_schedule cosine \
    --max_grad_norm 1.0 --save_steps 1000 --logging_steps 100 \
    --inline_eval_cmd "bash runs/inline_eval_vllm_v2.sh {step} $V17_P1 v17_p1 7500"
```

> 与 `runs/run_v15.sh` 的唯一实质差异：`inline_eval` 改用 `runs/inline_eval_vllm_v2.sh`
> （v15 原脚本用的 `runs/inline_eval_progressive.sh` 已废弃，会受 transformers/vLLM 后端漂移影响，
> 见 `docs/eval/pipeline.md`）。

---

## Step 6 — Phase 2 训练（照搬 v16 配方）

在 Phase 1 ckpt 上解冻、Aurora、anneal 数据、2000 步。新建 `runs/run_v17_p2.sh`（从 `runs/run_v16.sh`
复制）：

```bash
V17_P1=output/phase1_ckpt_v17
ANNEAL_PT=output/v17_anneal_512.pt
V17_P2=output/phase2_ckpt_v17

torchrun --nproc_per_node=2 --master_port=29518 train/finetune_muon.py \
    --model_path $V17_P1 --train_data $ANNEAL_PT --mode clm \
    --output_dir $V17_P2 \
    --max_seq_length 512 --batch_size 16 --gradient_accumulation_steps 4 \
    --max_steps 2000 --warmup_steps 200 \
    --muon_lr 5e-5 --adam_lr 5e-5 --min_lr_ratio 0.01 --lr_schedule cosine \
    --max_grad_norm 1.0 --use_aurora --save_steps 500 --logging_steps 100 \
    --inline_eval_cmd "bash runs/inline_eval_vllm_v2.sh {step} $V17_P2 v17_p2 2000"
```

---

## Step 7 — 评估

```bash
python evals/prefetch_eval_datasets.py            # 一次性，拉 mmlu/ceval/gsm8k 数据集
bash runs/smoke_new_tasks.sh                # ~2min 冒烟（建议先指向 v17 ckpt）
python evals/eval_with_piece_vllm.py --model_path output/phase2_ckpt_v17 \
    --tasks "lambada:0,piqa:5,hellaswag:10,arc_challenge:25,mmlu:5,ceval-valid:5,gsm8k:5"
python evals/eval_analysis.py                     # 生成 % loss vs base 对比表
```

评估口径见 `docs/eval/pipeline.md`。注意：所有对比必须同一后端（统一 vLLM），不要混 transformers。

---

## 步骤总览 / 进度

- [x] **Step 1** 修复特殊 token → `piece_v2.model`（81903）
- [x] **Step 2** 下载 base 模型（从 ModelScope 直连，不走代理）
- [x] **Step 3** 换分词器 → `Qwen3-0.6B-Base-new-tok-v2`，已 sanity check 通过
- [ ] **Step 4** 下载语料 + 重新预处理
- [ ] **Step 5** Phase 1 训练（v15 配方）
- [ ] **Step 6** Phase 2 训练（v16 配方）
- [ ] **Step 7** 评估

阻塞项：Step 4 的训练语料在另一台机器，需要重新下载。

## 环境注意事项（踩过的坑）

- **base 模型**：HF 走代理慢且 hf-mirror 走代理会失败；改用 ModelScope 直链
  `https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-Base/resolve/master/<file>`，
  且 **清空代理环境变量**（`https_proxy= http_proxy= all_proxy=`）后 curl 直连，很快。
- **装包**：venv `~/.venv` 没有 `pip`，必须用
  `uv pip install --python ~/.venv/bin/python ...`。
- **piece_tokenizer**：venv 里原先装的是 `~/ShiyuLab/Tokenizer` 的旧版绑定
  （`load()` 只接受单参数），与 `core/tokenizer_wrapper.py` 期望的 `load(model, cn_dict)`
  双参数不符。已用 `uv pip install --reinstall --no-cache ~/Shiyu/PieceTokenizer`
  重新编译安装，现指向正确的 repo。
- **cn-dict 不是纯加速**：验证发现它会**改变分词结果** —— 在词边界 pre-cut，避免 BPE
  跨词乱 merge（例：`编码器和解码器`，no-dict 会错切成 `和解|码`，cn-dict 正确切 `和|解码`）。
  词表是带 dict 训练的，所以 pretokenize / 训练 / eval 必须**全程带 dict.txt**。已确认
  `train/finetune_muon.py:_copy_tokenizer_artifacts` 会把 dict.txt 复制进每个 checkpoint，
  eval 加载 checkpoint 时不会退化 —— 链路无坑。

## Step 3 验证

`verify_retok.py`（19 项检查，全部 PASS）：cn-dict 模式生效（编码 5.5ms vs no-dict
2069ms，377x）、编解码 round-trip、特殊 token 就位、embedding 映射抽查 294 个与
`replace_tokenizer` 的 mean-of-BBPE 规则 100% 一致、forward 数值无 NaN/Inf。换分词器
统计：one-to-one 66.1% / multi-to-one 33.1% / fallback 0.8%（旧 65007 词表约
73.5% one-to-one，新词表更大、需 Phase 1 学习的行更多）。
