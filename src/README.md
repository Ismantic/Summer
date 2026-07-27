# src/ —— 模型与训练

**只依赖 torch。** 这一层解决的问题:把 ReTok 的两阶段训练讲清楚,而且每一步
都看得见。

| | 行数 | 替掉了 |
|---|---|---|
| `model.py` | ~290 | `transformers.Qwen3ForCausalLM` |
| `lora.py` | ~180 | `peft` |
| `checkpoint.py` | ~120 | `safetensors` 库 |
| `optim.py` | ~400 | Muon / Aurora(本来就是自写) |
| `train.py` | ~440 | 两阶段训练入口 |
| `evaluate.py` | ~105 | 固定切片上的 next-token loss |

## 为什么不用现成的库

装个 transformers 能让 290 行缩到 3 行,也就没什么可看的了。safetensors 的
格式本身只有二十来行代码,藏进依赖里反而看不见。

代价是要自己保证正确性,所以有三道防线(`make test`):

| | 判据 | 实测 |
|---|---|---|
| `test_model_equiv.py` | 与 transformers 逐层对齐 | float32 最大 1.66e-6,argmax 100% 一致 |
| `test_lora.py` | 与 peft 对拍 | logits 逐位相同 |
| `test_reproduce_sota.py --only ppl` | 端到端锚点 | 2.3334 vs 2.3331 |

## 两条约束

**`src/` 只依赖 torch。** 加依赖之前先想能不能放 `prepare/`。

**`src/` 不碰文本。** 分词、字→id 全在 `prepare/`,这里只读预编码好的 id。
所以 PieceTokenizer 不是 `src/` 的依赖,`--mode sft`(要现场分词 + padding)
会显式报错。

`src/` 是 package,训练脚本从仓库根跑:

```bash
python src/train.py --model_path ... --train_data ....pt --mode clm
```

## 不能改错的地方

- **state_dict 的 key 必须与 HF 的 `Qwen3ForCausalLM` 一致。** 改模块名会让
  已发布的权重全部失配,而模型照样随机初始化跑起来、loss 照样降。
- **RoPE 的 `inv_freq @ position_ids` 必须强制 float32。** 被 autocast 降到
  bf16 的话,整数过 256 就不精确(1023→1024),位置编码错乱,不报错。
- **tie 的 checkpoint 里没有 `lm_head.weight`。** 加载时绑回 embed。
- **LoRA 的 scaling 是 alpha/r 不是 alpha;lora_B 必须零初始化。**

详见 `docs/WHY.md`。
