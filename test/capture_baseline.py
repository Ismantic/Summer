"""重建 PieceTokenizer **之前**抓分词器行为基线。

    python test/capture_baseline.py           # 写 test/fixtures/tokenizer_baseline.json
    python test/capture_baseline.py --force   # 覆盖已有基线

## 顺序不能反

基线是用来发现「重建把行为改了」的 —— 重建之后再抓,抓到的就是新行为,
比对永远通过,等于没有防线。所以:

    1. python test/capture_baseline.py          ← 先
    2. bash prepare/install_deps.sh             ← 再重建
    3. python test/test_tokenizer.py            ← 最后比对

## 为什么这件事要紧

**PieceTokenizer 的编码一旦变了,81903 词表和已发布模型的 embedding 就对不上,
但代码不会报错** —— 只会悄悄训出/推出垃圾结果。

而且本机已经出现过版本漂移:`.venv` 里装的 piece_tokenizer 来自
`BERTc/deps/PieceTokenizer`(commit 7331d09),不是文档说的
`Shiyu/PieceTokenizer`(commit 5c3f081)。两边 `src/` 恰好逐字节相同,所以
没出事 —— 但那是运气,不是设计。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "test" / "fixtures" / "tokenizer_baseline.json"

# 覆盖几类容易出问题的输入:中文词边界、中英混排、空白、特殊 token、
# 长文本(dict 预切分路径)、罕见字。
PROBES = [
    "中国科学院计算技术研究所在北京市海淀区",
    "他平时喜欢锻练身体,昨天却因为发烧请了假。",
    "The quick brown fox jumps over the lazy dog.",
    "Qwen3-1.7B-Base 的 tokenizer 换成了 piece,vocab 81903。",
    "  多个   空格\t和制表符\n还有换行  ",
    "㥄龢齾齉爩",
    "1234567890 3.14159 -42 1e-9",
    "def f(x): return x ** 2  # 代码片段",
    "中国科学院计算技术研究所在北京市海淀区。" * 30,
    "",
    " ",
    "a",
    "中",
]


def collect() -> dict:
    import piece_tokenizer as pt

    from prepare.tokenizer import resolve_assets

    model, cn_dict = resolve_assets()
    tok = pt.Tokenizer()
    tok.load(model, cn_dict)

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    cases = []
    for s in PROBES:
        ids = tok.encode_as_ids(s)
        cases.append({
            "text": s,
            "ids": ids,
            "n": len(ids),
            "roundtrip_ok": tok.decode(ids) == s,
        })
    return {
        "vocab_size": tok.vocab_size(),
        "piece_model": {"path": str(model), "sha256": sha(model)},
        "cn_dict": {"path": str(cn_dict), "sha256": sha(cn_dict)},
        "module": str(Path(pt.__file__).resolve()),
        "cases": cases,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=str(BASELINE))
    a = p.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out} 已存在。要重抓请加 --force —— 但先想清楚:\n"
              f"  如果 PieceTokenizer 刚重建过,现在抓到的是**新**行为,\n"
              f"  覆盖掉旧基线就永远发现不了那次重建改了什么。")
        return 1

    data = collect()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"基线写入 {out}")
    print(f"  vocab      {data['vocab_size']}")
    print(f"  piece      {data['piece_model']['sha256'][:16]}…")
    print(f"  cn_dict    {data['cn_dict']['sha256'][:16]}…")
    print(f"  探针       {len(data['cases'])} 条,"
          f"{sum(c['roundtrip_ok'] for c in data['cases'])} 条 round-trip 正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
