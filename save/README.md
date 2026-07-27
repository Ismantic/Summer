# save/ —— 导出发布包、上传、核对

这一层解决的问题:**把训练产物变成别人下下来就能用的东西,并且能对得上账。**

| | |
|---|---|
| `export.py` | 从 checkpoint 导出干净的 HF 上传目录(权重 + 分词器产物 + 模型卡 + 推理示例) |
| `releases.py` | 已发布内容的唯一记录。`--verify` 做三方 sha256 核对:登记值 / HF 线上 / 本地文件 |
| `generate.py` | 交互式续写。只依赖 torch + PieceTokenizer,不用 transformers |
| `sota/` | SOTA checkpoint(gitignore) |

`make help` 看命令。

## 发布包里有什么

```
config.json  model.safetensors          权重(310 个张量,tie,没有 lm_head)
piece.model  dict.txt                   分词器 —— **两个都必须带**
token_mapping.json                      pad/bos/eos 的 id
tokenizer_wrapper.py                    HF 风格的分词器外壳
README.md  requirements.txt             模型卡 + 依赖
```

**`dict.txt` 不是可选的。** 少了它中文的 token id 会变(不只是慢),而
round-trip 照样正确、不报错 —— 下载的人不会发现。见 `docs/WHY.md` 第一节。

## 发布之前

1. `make -C save export` 导出
2. `make test`(仓库根)确认回归全绿
3. 人工看一遍模型卡:数字有没有过期、口径写清楚没有
4. 上传后 `make -C save verify` 核对 sha256,并把新的 sha 记进 `releases.py`

第 4 步不能省。**发布是不可逆的** —— 一旦有人下走了,再改就对不上了。

## 上传

```bash
huggingface-cli login
huggingface-cli repo create Ismantic/Qwen3-1.7B-Base-ReTok --type model
huggingface-cli upload Ismantic/Qwen3-1.7B-Base-ReTok \
    hf_upload/Qwen3-1.7B-Base-ReTok . --repo-type model
```

`repo create` 推断不出组织的话,去 HF 网页在 `Ismantic` 下建好,再跑同样的
upload。

**只传最终合并后的权重。** `checkpoint-500/1000/1500/` 是训练中途的 LoRA
adapter,不是发布产物。`export.py` 默认用硬链接,同一文件系统上不会真的复制
那 3GB;要真副本加 `--copy`。

许可:基座 `Qwen/Qwen3-1.7B-Base` 是 Apache-2.0,衍生模型沿用。

## 兼容性

权重能被 transformers 当 Qwen3 加载(`AutoModelForCausalLM` 实测可用),
**但分词器不是标准的** —— 这个发布用的是 `piece.model` + `token_mapping.json`,
`AutoTokenizer` 走不通。生成要用发布包里的 `tokenizer_wrapper.py`,
或本仓库的 `prepare/tokenizer.py`。

因此 HF 的在线推理(hosted inference)用不了,除非哪天把 piece 分词器封装成
标准 `AutoTokenizer`。
