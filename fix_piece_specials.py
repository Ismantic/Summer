"""修补 NEW piece.model:
  - id 81899 行的 piece 从 "<pad>  改回 <pad>
  - id 81902 行的 piece 从 <system>" 改回 <system>
  - 把 pad_id 从 0 改到 81899

NEW piece.model 训练时 EXTRA_TOKENS="<pad>,<user>,<assistant>,<system>" 整串
被当 1 个 string 传进去,split 后头尾两端没剥掉外层引号,造成 <pad>/<system>
piece_to_id 撞回 0(unk)。
"""
import sys

SRC = "/home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece.model"
DST = "/home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece_fixed.model"

with open(SRC, "rb") as f:
    raw = f.read()

# 整行 anchor 替换(前后 \n 锚定,确保唯一)
def patch(raw, old, new):
    pat = b"\n" + old + b"\n"
    rep = b"\n" + new + b"\n"
    n = raw.count(pat)
    if n != 1:
        sys.exit(f"FATAL: {old!r} 命中 {n} 次,期望 1")
    return raw.replace(pat, rep, 1)

# 1. pad piece
raw = patch(raw,
    b'81899\t"<pad>\t0\t3\t\t',
    b'81899\t<pad>\t0\t3\t\t')

# 2. system piece
raw = patch(raw,
    b'81902\t<system>"\t0\t3\t\t',
    b'81902\t<system>\t0\t3\t\t')

# 3. pad_id 从 -1 → 81899
raw = patch(raw, b"pad_id=-1", b"pad_id=81899")
# 4. head 的 vocab_size 81899 → 81903(与 [Pieces].size 对齐)
raw = patch(raw, b"vocab_size=81899", b"vocab_size=81903")

with open(DST, "wb") as f:
    f.write(raw)
print(f"DONE: {DST}")
