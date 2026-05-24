# BERT-base MLM Pretraining (POC)

Encoder-only BERT-base from-scratch 预训练的最小实现,独立目录可移植到任何
新仓库,不依赖外部代码。

## 架构

- BERT-base:12 layers / 768 hidden / 12 heads / 3072 FFN
- Vocab:`piece_vocab_size + 1`(末位预留 `<mask>` id)
- max_position:1024(可调)
- MLM only,无 NSP

参数估算:vocab 81903 + 标准 BERT-base = **~149M**(embed 62.9M + transformer 86.5M)。

## 外部依赖

只依赖 PyTorch + transformers + 一个 piece tokenizer 目录(含 `piece.model`,
可选 `dict.txt` for CN 分词)。无需 lm-eval / vllm。

```
pip install torch transformers
```

## 三步流程

### 1. 构建 init 模型(随机权重)

```bash
python build_bert_init.py \
    --piece_dir /path/to/piece_tokenizer_dir \
    --output_dir ./bert_init \
    --piece_vocab_size 81903
```

`piece_dir` 期望提供:
- `piece.model`(必)— C++ PieceTokenizer 训练产物
- `dict.txt`(可选)— CN 分词字典
- `token_mapping.json`(可选)— `{"pad_id": ..., "bos_id": ..., ...}`

build 脚本会:
1. 创建 BertForMaskedLM(`piece_vocab_size + 1` 个 embedding)
2. Qwen3-style 随机 init(`std=0.02`)
3. 拷 piece 文件到 `bert_init/`
4. 写 `mask_token_id.txt`(`= piece_vocab_size`)

### 2. 训练数据

需要外部预切好的 `[N, seq_len]` int32 / int64 PyTorch tensor,通过 `torch.save`
保存为 `.pt` 文件。每个 chunk 内 token id 必须在 `[0, piece_vocab_size)` 范围
(不会出现 mask_token_id,因为那是训练时 collator 临时插入的)。

### 3. 启动训练

```bash
bash run_bert_train.sh ./bert_init /path/to/train.pt ./bert_train_ckpt
```

或直接调用 train 脚本:

```bash
python train_bert_mlm.py \
    --model_path ./bert_init \
    --train_data /path/to/train.pt \
    --output_dir ./bert_train_ckpt \
    --max_seq_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing --compile \
    --max_steps 8000 --warmup_steps 500 \
    --lr 1e-4 --mlm_prob 0.15 \
    --save_steps 2000
```

## MLM 实现细节

`train_bert_mlm.py:mlm_mask_batch` 实现原始 BERT mask policy:
- 15% 位置选中(避开 `<pad>`)
- 其中 80% → `<mask>` token id(末位新增)
- 10% → 随机 token
- 10% → 不变
- 标签:未选中位置设 `-100`(CE loss 忽略)

## 文件清单

```
BERT/
├── build_bert_init.py   # 随机 init Bert + 拷 piece tokenizer
├── train_bert_mlm.py    # MLM 训练主循环
├── run_bert_train.sh    # 8000 步 1B token POC 入口
└── README.md            # 本文件
```

无其他依赖,可以整目录搬到任意位置。
