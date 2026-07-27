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
Summer-Tokenizer.pt                     81903 词表 —— **与上游 PieceTokenizer 的
Summer-Tokenizer.dict.txt               save/ 下同名,溯源自明**
token_mapping.json                      pad/bos/eos 的 id
model.py  checkpoint.py  tokenizer.py   推理代码 —— 所以**不需要 transformers,
                                        也不需要 safetensors 库**
example_load.py  example_vllm.py        两个可跑的示例
README.md  requirements.txt             模型卡 + 依赖(就 torch 一个)
```

**中文分词词典不是可选的。** 少了它中文的 token id 会变(不只是慢),而
round-trip 照样正确、不报错 —— 下载的人不会发现。见 `docs/WHY.md` 第一节。

## 发布之前

1. `make -C save export` 导出
2. `make test`(仓库根)确认回归全绿
3. 人工看一遍模型卡:数字有没有过期、口径写清楚没有
4. 上传后 `make -C save verify` 核对 sha256,并把新的 sha 记进 `releases.py`

第 4 步不能省。**发布是不可逆的** —— 一旦有人下走了,再改就对不上了。

## 上传

```bash
huggingface-cli login          # 需要 write 权限的 token
make upload-dry                # 先看要传什么
make upload-code               # 只传代码 + 词表,不碰 3GB 权重
make upload                    # 全部(含权重)
```

**权重没变时用 `upload-code`** —— 重传 3GB 纯属浪费,而且覆盖已发布权重本身
就是不必要的风险(别人可能已经下走了)。判断权重变没变:`make verify`。

`--code-only` 只跳过 `model.safetensors`;词表跟代码一起传,因为它必须与
加载器期望的文件名保持同步。

**上传要走官方源**,hf-mirror 是只读镜像。`save/upload.py` 会检查
`HF_ENDPOINT` 并在指向镜像时拦下来。

**只传最终合并后的权重。** `checkpoint-500/1000/1500/` 是训练中途的 LoRA
adapter,不是发布产物。`export.py` 默认用硬链接,同一文件系统上不会真的复制
那 3GB;要真副本加 `--copy`。

许可:基座 `Qwen/Qwen3-1.7B-Base` 是 Apache-2.0,衍生模型沿用。

## 兼容性

权重能被 transformers 当 Qwen3 加载(`AutoModelForCausalLM` 实测可用),
**但分词器不是标准的** —— 用的是 `Summer-Tokenizer.pt` + `token_mapping.json`,
`AutoTokenizer` 走不通。生成要用发布包自带的 `tokenizer.py`。

vLLM 能加载权重(它用自己那份 Qwen3 实现,按 state_dict key 灌),但同样认不了
这个词表 —— 要传 `skip_tokenizer_init=True` 并自己编码 id,见发布包里的
`example_vllm.py`。

因此 HF 的在线推理和 `vllm serve` 都用不了,除非哪天把 piece 分词器封装成
标准 `AutoTokenizer`。
