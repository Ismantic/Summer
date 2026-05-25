# Char-level RoBERTa-style MLM(零外部依赖)

中文字级 BERT-base MLM 预训。**纯字级输入,无 piece BPE / 无 LTP / 无同义词词典 / 无 NSP / 无 SOP / 无 WWM**,完全可移植可独立成 repo。

最终目标:训出来的 backbone 作为 **CWS teacher** 的底座(后接 multi-criteria CWS fine-tune + silver labeling + Wapiti CRF 部署模型)。

## 架构

- **BERT-base**(可调):12 layers / 768 hidden / 12 heads / 3072 FFN
- **Vocab**:字级(扫训练语料构建,常见 5 个 specials + 频次 >= N 的字 ≈ 7500-8000)
- **max_position**:512(适合 CWS 句级输入)
- **预训目标**:纯 MLM
  - 15% 单字 mask(80% [MASK] / 10% random / 10% unchanged)
  - 动态 mask(每 forward 重新采样,RoBERTa-style)
  - 无 NSP,`type_vocab_size=1`
- **参数估算**:vocab 7500 时约 **107M**(embed 5.8M + transformer 86.5M + 内部 modules)

## 外部依赖

```
pip install torch transformers
```

**仅此**。无 LTP / 无 Synonyms / 无 piece_tokenizer / 无 lm-eval / 无 vllm。

## 四步流程

### 1. 构建字级 vocab

```bash
python build_char_vocab.py \
    --corpus /path/to/chinese_corpus.txt \
    --output vocab.txt \
    --min_freq 10
```

输出 `vocab.txt`,每行一个字符,id = 行号。前 5 行固定是 specials:
```
[PAD]    id=0
[UNK]    id=1
[CLS]    id=2
[SEP]    id=3
[MASK]   id=4
的        id=5
一        id=6
...
```

`--min_freq 10` 控制 vocab 大小:21GB 中文 corpus 用 10 大约得到 7500 字,覆盖率 99.95%+。

### 2. 数据编码(char → id)

```bash
python encode_char_data.py \
    --corpus /path/to/chinese_corpus.txt \
    --vocab vocab.txt \
    --output train.pt \
    --seq_len 512 \
    --total_tokens 1000000000
```

输出 `[N, 512]` int32 PyTorch tensor,保存为 `.pt`。1B 字约 2GB 文件。

### 3. 构建 init 模型(随机权重)

```bash
python build_bert_init.py \
    --vocab vocab.txt \
    --output_dir ./bert_init
```

输出目录含 `config.json` + `model.safetensors` + `vocab.txt` + `mask_token_id.txt`。

### 4. 启动训练

```bash
bash run_bert_train.sh ./bert_init ./train.pt ./bert_train_ckpt
```

或直接调用:

```bash
python train_bert_mlm.py \
    --model_path ./bert_init \
    --train_data ./train.pt \
    --output_dir ./bert_train_ckpt \
    --max_seq_length 512 \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --gradient_checkpointing --compile \
    --max_steps 8000 --warmup_steps 1000 \
    --lr 1e-4 --min_lr_ratio 0.1 \
    --mlm_prob 0.15 \
    --save_steps 2000
```

## 时间预算(单卡 4090)

| 数据量 | 步数 | 时长 |
|---|---|---|
| 1B 字 | 8,000 | ~15-20h |
| 2B 字 | 16,000 | ~30-40h |
| 5B 字 | 40,000 | ~3-5 天 |

eff_bs = 32 × 8 × 512 = 131,072 字 / step。

## 跟下游 CWS 的关系

预训完成后,**`bert_train_ckpt` 就是通用 char backbone**,可以:

1. 接 4-类 CRF head 做 **CWS**(BIES 标注 + multi-criteria 联合训 → F1 ~96.5-97%)
2. 接 NER head 做命名实体
3. 接其它字级序列标注任务

CWS / silver labeling / Wapiti 部署的代码以后会加进本目录(独立可移植)。

## 文件清单

```
BERT/
├── build_char_vocab.py    # 扫语料 → vocab.txt
├── encode_char_data.py    # corpus + vocab → [N, seq_len].pt
├── build_bert_init.py     # vocab.txt → 随机权重 BertForMaskedLM
├── train_bert_mlm.py      # MLM 训练主循环(纯 MLM,动态 mask)
├── run_bert_train.sh      # 入口:8000 步 ≈ 1B 字 POC
└── README.md              # 本文件
```

整目录无任何 hard-coded 外部路径,可直接整体搬走。
