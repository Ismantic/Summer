# prepare/ —— 编排

这一层解决的问题:**把 data/ 的原始输入变成 src/ 能吃的东西,以及调度训练和评测。**

`src/` 只依赖 torch、且不碰文本 —— 所有需要分词器、vLLM、lm_eval、COMET 的
事情都在这一层。

| | |
|---|---|
| `install_deps.sh` | clone + 编译 PieceTokenizer。**词表也在它仓库里,本仓库不留副本** |
| `tokenizer.py` | PieceTokenizer 的 HF 风格外壳 + `resolve_assets()` 反查词表 |
| `retok.py` | 词表手术(不可逆)。顺带导出 `frozen_ids.json` |
| `encode_corpus.py` | 预编码成 `[N, seq_len]` 的 `.pt`。`--mix main` / `--mix anneal` |
| `benchmark.py` | mono 六任务(vLLM + lm_eval) |
| `translate.py` | WMT BLEU + COMET(vLLM) |
| `sweep.py` | 多 ckpt 批量评测 + 出对照表 |

`make help` 看命令,`make status` 看每一步的产物在不在。

## 不可逆的那一步

`retok.py` 的嵌入映射一改,**所有已有 checkpoint 全部失配** —— 而且模型照样
能加载、照样跑。已发布的 `Ismantic/Qwen3-1.7B-Base-ReTok` 就是按当前映射规则
产出的。

## 重建 PieceTokenizer 之前先抓基线

    1. python test/capture_baseline.py     ← 先
    2. bash prepare/install_deps.sh        ← 再重建
    3. python test/test_tokenizer.py       ← 最后比对

顺序反了就失去意义。分词器是 C++ 编的,换 commit、换编译器都可能改变编码,
**不会有任何提示**,而编码一变 81903 词表和已发布权重就对不上了。

## 缺 dict.txt 会报错,不会降级

少了中文分词词典,中文的 token id 就变了 —— 不只是慢。而 round-trip 照样正确,
看着没事。所以 `tokenizer.py` 直接报错。详见 `docs/WHY.md` 第一节。
