"""对比新旧 piece tokenizer:
  旧 = /home/tfbao/Shiyu/Summer/piece_v2.model (81899 + 补丁 4 extra)
  新 = /home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece.model (训练时直接含 4 extra)

指标:
  1. vocab 实际大小、可见 specials
  2. EN/CN/CODE 样本 tokens_per_char、tokens_per_doc(越低越好,压缩率高)
  3. 几条固定句子的切分粒度可视对比
"""
import time, random, os, sys
from piece_tokenizer import Tokenizer as PieceTokenizer

OLD = "/home/tfbao/Shiyu/Summer/piece_v2.model"
NEW = "/home/tfbao/Shiyu/PieceTokenizer/scripts/output/piece.model"
DICT = "/home/tfbao/Shiyu/PieceTokenizer/dict.txt"
EN_FILE = "/home/tfbao/Shiyu/Summer/tokenizer_corpus/en.txt"
CN_FILE = "/home/tfbao/Shiyu/Summer/tokenizer_corpus/cn.txt"
N_SAMPLES = 2000

def load(model):
    tk = PieceTokenizer()
    tk.load(model, DICT)
    return tk

def stats(tk, lines, label):
    total_tok = 0; total_chars = 0; total_bytes = 0
    t0 = time.time()
    for s in lines:
        ids = tk.encode_as_ids(s)
        total_tok += len(ids)
        total_chars += len(s)
        total_bytes += len(s.encode("utf-8"))
    el = time.time() - t0
    return {
        "docs": len(lines),
        "tokens": total_tok,
        "chars": total_chars,
        "bytes": total_bytes,
        "tok_per_char": total_tok / max(1, total_chars),
        "tok_per_byte": total_tok / max(1, total_bytes),
        "tok_per_doc": total_tok / max(1, len(lines)),
        "encode_sec": el,
    }

def sample(path, n):
    """从大文件随机抽 n 行(避免读整文件)。"""
    sz = os.path.getsize(path)
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # 随机 n*3 offset,各往后读一行(丢首半行)
        offsets = sorted(random.sample(range(sz - 1000), min(n*4, 100_000)))
        for off in offsets:
            f.seek(off)
            f.readline()  # 半行丢弃
            line = f.readline().strip()
            if line and len(line) > 50:
                out.append(line)
                if len(out) >= n:
                    break
    return out

random.seed(42)
print("加载 EN/CN 样本...")
en_lines = sample(EN_FILE, N_SAMPLES)
cn_lines = sample(CN_FILE, N_SAMPLES)
print(f"  EN: {len(en_lines)} 行,平均 {sum(len(s) for s in en_lines)/len(en_lines):.0f} chars")
print(f"  CN: {len(cn_lines)} 行,平均 {sum(len(s) for s in cn_lines)/len(cn_lines):.0f} chars")

# 固定典型句子
SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Quantum entanglement is a physical phenomenon that occurs when pairs of particles interact.",
    "人工智能正在深刻改变软件工程的实践方式。",
    "中华人民共和国成立于一九四九年十月一日。",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    return quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "Translate: The cat sat on the mat. -> 猫坐在垫子上。",
    "中文混 English 句子,看 tokenizer 怎么处理 mixed-language input。",
]

print("\n加载 OLD tokenizer...")
old = load(OLD)
print(f"  vocab_size = {old.vocab_size()}")
print("加载 NEW tokenizer...")
new = load(NEW)
print(f"  vocab_size = {new.vocab_size()}")

print("\n" + "="*78)
print("【1】压缩率(tokens per char,越低越好 — 单 token 覆盖更多字符)")
print("="*78)
print(f"{'corpus':<8} {'metric':<14} {'OLD':>12} {'NEW':>12} {'Δ':>10} {'NEW/OLD':>9}")
print("-"*78)
for label, lines in [("EN", en_lines), ("CN", cn_lines)]:
    s_old = stats(old, lines, label)
    s_new = stats(new, lines, label)
    for key, name in [("tok_per_char","tokens/char"),
                      ("tok_per_byte","tokens/byte"),
                      ("tok_per_doc","tokens/doc")]:
        a, b = s_old[key], s_new[key]
        d = b - a
        ratio = b/a*100
        if key == "tok_per_doc":
            print(f"{label:<8} {name:<14} {a:>12.1f} {b:>12.1f} {d:>+10.1f} {ratio:>8.1f}%")
        else:
            print(f"{label:<8} {name:<14} {a:>12.4f} {b:>12.4f} {d:>+10.4f} {ratio:>8.1f}%")

print("\n" + "="*78)
print("【2】典型样本切分对比")
print("="*78)
for s in SAMPLES:
    o_ids = old.encode_as_ids(s)
    n_ids = new.encode_as_ids(s)
    o_pieces = [old.id_to_piece(i) for i in o_ids]
    n_pieces = [new.id_to_piece(i) for i in n_ids]
    print(f"\n输入 ({len(s)} chars): {s[:80]}{'...' if len(s)>80 else ''}")
    print(f"  OLD ({len(o_ids):3d} tok): {' '.join(o_pieces[:30])}{' ...' if len(o_pieces)>30 else ''}")
    print(f"  NEW ({len(n_ids):3d} tok): {' '.join(n_pieces[:30])}{' ...' if len(n_pieces)>30 else ''}")
    diff = len(n_ids) - len(o_ids)
    marker = "↓更优" if diff < 0 else ("↑更差" if diff > 0 else "=")
    print(f"  Δ = {diff:+d} tokens  {marker}")
