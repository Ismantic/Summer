"""列出换分词器时「多对一映射」的中文 piece。

多对一 = 新词表的 piece 用 Qwen BBPE 编码后得到 >1 个旧 token，embedding 取均值
（见 replace_tokenizer.py:build_embedding_mapping）。这里只看含汉字的 piece。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import piece_tokenizer as pt
from transformers import AutoTokenizer

NEW = "/home/tfbao/Shiyu/Summer/piece_v2.model"
OLD = "/home/tfbao/new/Qwen3-0.6B-Base"
SPECIAL = {"<unk>", "<s>", "</s>", "<pad>", "<user>", "<assistant>", "<system>"}

new_tok = pt.Tokenizer()
new_tok.load(NEW)
old_tok = AutoTokenizer.from_pretrained(OLD, trust_remote_code=True)


def has_cjk(s):
    return any('一' <= c <= '鿿' for c in s)


cn_total = one2one = 0
multi = []   # (piece, n_old_tokens)
for i in range(new_tok.vocab_size()):
    try:
        piece = new_tok.id_to_piece(i)
    except Exception:
        continue
    if piece in SPECIAL or not has_cjk(piece):
        continue
    cn_total += 1
    text = piece.replace("▁", " ") or " "
    old_ids = old_tok.encode(text, add_special_tokens=False)
    if len(old_ids) > 1:
        multi.append((piece.replace("▁", ""), len(old_ids)))
    elif len(old_ids) == 1:
        one2one += 1

print(f"含汉字的 piece 总数: {cn_total}")
print(f"  一对一: {one2one} ({100*one2one/cn_total:.1f}%)")
print(f"  多对一: {len(multi)} ({100*len(multi)/cn_total:.1f}%)")

# 按拆成几个旧 token 分布
from collections import Counter
dist = Counter(n for _, n in multi)
print("\n多对一拆分数分布 (拆成 N 个旧 BBPE token):")
for n in sorted(dist):
    print(f"  {n} 个: {dist[n]}")

# 按字数给样例
print("\n=== 样例（按 piece 字数分组，每组前 25 个）===")
by_len = {}
for piece, n in multi:
    by_len.setdefault(len(piece), []).append((piece, n))
for L in sorted(by_len):
    if L == 0:
        continue
    sample = by_len[L][:25]
    print(f"\n[{L} 字]  共 {len(by_len[L])} 个:")
    print("  " + "  ".join(f"{p}({n})" for p, n in sample))

# 完整列表存文件
out = "/home/tfbao/Shiyu/Summer/output/multi_mapped_cn.txt"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for piece, n in multi:
        f.write(f"{piece}\t{n}\n")
print(f"\n完整 {len(multi)} 个已写入 {out}")
