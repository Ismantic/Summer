# 评测栈

## 三个入口

| | 跑什么 | 需要 | 耗时 |
|---|---|---|---|
| `prepare/benchmark.py` | mono 六任务(lm-eval-harness) | vllm + lm_eval | ~36 分钟 |
| `prepare/translate.py` | WMT BLEU + COMET | vllm + comet 模型 | ~5 分钟 |
| `src/evaluate.py` | 固定切片 next-token loss | `PY`(只要 torch) | 几秒 |

批量:`prepare/sweep.py run --ckpt tag=path ...`,跑完出对照表。
只汇总已有结果:`prepare/sweep.py table --tags ...`。

Makefile 里是 `make -C prepare bench / trans / sweep`。

## 这三个各防一段,不能互相替代

**`benchmark.py` 和 `translate.py` 走 vLLM,测不到 `src/model.py`** ——
vLLM 从 checkpoint 直接加载权重,用它自己的 Qwen3 实现。那两项全绿也说明不了
自写模型对不对。

能锚住 `src/model.py` 的只有:

- `test/test_model_equiv.py` —— 与 transformers 逐层对齐
- `src/evaluate.py`(即 `make test-ppl`)—— 走自己的 forward

RoPE 那个 autocast bug 就是这么抓到的:等价性测试漏了(它用 256 长度、
没开 autocast),PPL 锚点抓到了。

## 数字怎么读

### vLLM 的贪心解码不可复现

同一 ckpt、同一条命令跑 6 次:

| | range | sd |
|---|---|---|
| zh-en BLEU | 0.1033 | 0.022 |
| en-zh BLEU | 0.1337 | 0.056 |
| zh-en COMET | 0.0010 | 0.0005 |
| en-zh COMET | 0.0005 | 0.0002 |

**0.1 量级的 BLEU 差是噪声。** COMET 稳两个数量级,比 BLEU 可靠。

mono 六任务里样本量大的(lambada 5153 / hellaswag 10042)能精确重现;
ceval 的子任务只有 18–31 题,翻一两道就看得见,聚合值实测漂 +0.0022。

### 不能混后端

实测同一个 Qwen3-1.7B-Base:

| 任务 | transformers | vLLM | 差 |
|---|---|---|---|
| lambada_openai | 0.6290 | 0.6513 | **+0.0223** |
| piqa | 0.7720 | 0.7731 | +0.0011 |
| arc_challenge | 0.5546 | 0.5512 | −0.0034 |

lambada 差 2.2 个点,方向还不一致。所以 `eval_results/full/` 的目录名带后端
后缀,`sweep.py` 的 `read_mono()` **只认 `_vllm`,找不到就留空**。

### gsm8k 恒在 0.035 附近

换词表打断了 Qwen3 的数值 token 化,两个 phase 都救不回来。**词表代价,
不是回归。**

## 产物布局

```
eval_results/full/<tag>_vllm/<task>/result.json      mono
eval_results/translate_wmt22/<tag>_*.json            BLEU + COMET
eval_results/ppl/<tag>.json                          固定切片 loss
```

后端写在目录名里是硬约定 —— 见上面「不能混后端」。

## 数据集

`python data/prefetch_eval_datasets.py` 一次性拉齐 mono 那六个 benchmark,
之后可以 `HF_DATASETS_OFFLINE=1` 离线跑。

WMT 测试集不用管 —— `prepare/translate.py` 走 sacrebleu 自带的下载器。

COMET 模型:`make -C data download-comet`,路径从 `data/source.py` 注册表取。
**`--compute_comet` 必须配 `--save_all_samples`** —— 不加的话脚本只留 5 条
样本,会 WARN 一句然后跳过 COMET,不报错。

## 为什么是 vLLM 而不是 transformers.generate

transformers 的 generate 在版本之间不稳,曾经因此在 under-trained ckpt 上
得出过错误结论。vLLM 的生成路径独立于 transformers 版本。

代价是贪心解码不可复现(见上)。关掉 `enable_prefix_caching` 和异步调度大概
能确定化,但那会让新数字与 `eval_results/` 里已有的 tag 不可比 —— 没有采纳,
只把可复现性的代价记下来。

## 单卡

本机一张 RTX 4090。旧版本 `retro_eval_*.py` 的 `--gpus 0 1` 是 2× A6000 时代
的分片,已删。`sweep.py` 顺序跑。
