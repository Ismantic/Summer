# data/ —— 下载语料与模型资产

这一层解决的问题:**只靠 Hugging Face 就能把所有输入准备齐,不依赖本机既有文件。**

| | |
|---|---|
| `source.py` | 数据源注册表,**唯一的真相来源**。开头有十个语料源逐个溯源的依据 |
| `download.py` | 按注册表下载。glob→正则筛 + 客户端截断 n_parts + 清代理 |
| `prefetch_eval_datasets.py` | lm-eval 那六个 benchmark 的数据集 |

`make help` 看命令。

## 改了 source.py 之后跑 `make probe`,不要跑 `make download`

全量 100GB+,而**会写错的只有 glob / fmt / text_field 这三样,一个文件就能
暴露**。`probe` 给每个源下第一个匹配文件,再用 `iter_text` 真读 20 篇 ——
`compileall` 和 dry-run 都看不出这类错。

全量下载只有真要重建语料时才需要。v18 的 checkpoint 已经发布,复现它不需要语料。

## 池子 ≠ 消耗量

各源实际吃掉的都在 1 亿 token 上下,而上游单个文件动辄几亿 —— SkyPile 一个
jsonl 就有约 319M token,够喂满它 126M 的份额。`n_parts` 取的是**池子规模**
(对齐 v18 当初的采样多样性),不是消耗量。想省磁盘调小即可,只要池子 ≥ 消耗量,
训练照跑。

## 需要 HF 登录

`BAAI/CCI3-HQ` 是 gated 数据集,要先在它的 HF 页面接受条款,再
`huggingface-cli login`(或设 `HF_TOKEN`)。其余源都是公开直下。

这是「只依赖 HF 和 GitHub」之外唯一的手动步骤。
