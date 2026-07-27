"""重建 PieceTokenizer **之后**,比对行为有没有变。

    python test/test_tokenizer.py

配合 `test/capture_baseline.py` 用,顺序不能反:

    1. python test/capture_baseline.py     ← 重建前抓基线
    2. bash prepare/install_deps.sh        ← 重建
    3. python test/test_tokenizer.py       ← 比对(本文件)

`install_deps.sh` 装完会自动跑这一步。

## 为什么必须有这道关

**编码一旦变了,81903 词表和已发布模型的 embedding 就对不上,但代码不会报错**
—— 模型照样加载、照样输出、loss 照样是个正常数字,只是全错。

没有基线就没法发现:分词器是 C++ 编的,换个 commit、换个编译器版本、
换个 CMake 选项都可能改变行为,而这些都不会有任何提示。

## 没有基线时会怎样

打印提示并**返回失败**(不是「通过」)。BERTc 那条约定:缺输入而没跑,
绝不能和「跑了且通过」混为一谈。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "test" / "fixtures" / "tokenizer_baseline.json"


def main() -> int:
    if not BASELINE.exists():
        print(f"没有基线文件 {BASELINE}")
        print("  先跑 `python test/capture_baseline.py` —— 而且必须在重建")
        print("  PieceTokenizer **之前**跑,不然抓到的是重建后的行为,比对没意义。")
        return 1

    import piece_tokenizer as pt

    from prepare.tokenizer import resolve_assets

    old = json.loads(BASELINE.read_text())
    model, cn_dict = resolve_assets()
    tok = pt.Tokenizer()
    tok.load(model, cn_dict)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    ok = True
    print(f"基线抓于  {old['module']}")
    print(f"当前模块  {Path(pt.__file__).resolve()}")

    # 词表文件本身变没变 —— 变了的话下面的 id 对不上是必然的,先说清楚
    print("\n=== 词表文件 ===")
    for key, path in (("piece_model", model), ("cn_dict", cn_dict)):
        now, was = sha(path), old[key]["sha256"]
        same = now == was
        ok &= same
        print(f"  {key:12} {now[:16]}…  "
              f"{'与基线相同' if same else '**变了**(基线 ' + was[:16] + '…)'}")

    print(f"\n=== vocab_size ===")
    same = tok.vocab_size() == old["vocab_size"]
    ok &= same
    print(f"  {tok.vocab_size()}  基线 {old['vocab_size']}  "
          f"{'ok' if same else '**变了**'}")

    print(f"\n=== 编码行为({len(old['cases'])} 条探针)===")
    bad = []
    for c in old["cases"]:
        ids = tok.encode_as_ids(c["text"])
        rt = tok.decode(ids) == c["text"]
        if ids != c["ids"] or rt != c["roundtrip_ok"]:
            bad.append((c, ids, rt))
    for c, ids, rt in bad[:5]:
        preview = c["text"][:40].replace("\n", "\\n")
        print(f"  **不一致** {preview!r}")
        print(f"      基线 {len(c['ids'])} 个 id: {c['ids'][:12]}")
        print(f"      现在 {len(ids)} 个 id: {ids[:12]}")
        if rt != c["roundtrip_ok"]:
            print(f"      round-trip {c['roundtrip_ok']} → {rt}")
    ok &= not bad
    print(f"  {len(old['cases']) - len(bad)}/{len(old['cases'])} 条一致"
          f"{'' if not bad else f'  **{len(bad)} 条变了**'}")

    print("\n" + ("行为未变" if ok
                  else "**行为变了** —— 已发布模型的 embedding 会对不上,而且不报错"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
