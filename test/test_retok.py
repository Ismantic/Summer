"""ReTok 词表替换后的验证脚本。

重点验证:
  1. PieceTokenizer cn-dict 模式确实启用（词表是带 dict.txt 训练的，推理也必须带）
  2. 编解码 round-trip 正确
  3. 特殊 token 就位
  4. embedding 映射与 replace_tokenizer 的规则一致（抽查）
  5. 模型 forward 数值正常（无 NaN/Inf）

    python test/test_retok.py                       # 默认查当前 SOTA ckpt
    python test/test_retok.py --model_path <dir>    # 查 retok 刚产出的目录

不带 --model_path 时按顺序找:save/sota/v18_p2_tie → output/phase2_ckpt_v18_tie
→ output/init_retok。都没有就跳过(**不算通过**)。

原版写死了 0.6B 时代的一个本机路径,
那两个目录早就不在了 —— 阶段 4 改成走参数 + 注册表。
"""
import os
import json
import time
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
import piece_tokenizer as pt
from prepare.tokenizer import PieceTokenizerWrapper
from src.model import Qwen3ForCausalLM


def _resolve_new_tok():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None)
    a, _ = ap.parse_known_args()
    if a.model_path:
        return a.model_path
    for c in ("save/sota/v18_p2_tie", "output/phase2_ckpt_v18_tie",
              "output/init_retok"):
        if os.path.exists(os.path.join(ROOT, c, "config.json")):
            return os.path.join(ROOT, c)
    return None


def _resolve_old_base():
    """原始 Qwen 基座 —— 从注册表取,取不到就退回 HF repo id。"""
    from data import source
    d = source.get("qwen_base").dir()
    return str(d) if (d / "config.json").exists() else source.get("qwen_base").repo_id


NEW_TOK = _resolve_new_tok()
if NEW_TOK is None:
    print("跳过:找不到任何换过词表的模型目录"
          "(save/sota/ 或 output/ 下)。先跑 `make -C prepare retok`。")
    raise SystemExit(0)
OLD_BASE = _resolve_old_base()
print(f"新词表模型  {NEW_TOK}")
print(f"原始基座    {OLD_BASE}")

_ok = True


def section(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


def check(name, cond, detail=""):
    global _ok
    if not cond:
        _ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---- 1. cn-dict 模式 ----
section("1. PieceTokenizer cn-dict 模式（词表是带 dict 训练的）")
from prepare.tokenizer import _DICT_NAMES, _MODEL_NAMES, _first_in
_PIECE = _first_in(NEW_TOK, _MODEL_NAMES)
_DICT = _first_in(NEW_TOK, _DICT_NAMES)
check("词表存在于模型目录", _PIECE is not None, str(_PIECE))
check("中文分词词典存在于模型目录", _DICT is not None, str(_DICT))

raw = pt.Tokenizer()
raw.load(_PIECE)                                                     # 不带 dict
cn = pt.Tokenizer()
cn.load(_PIECE, _DICT)                                               # 带 dict

cn_text = "机器翻译是自然语言处理领域的重要任务，深度学习模型在大规模语料上预训练。" * 40
t0 = time.time(); ids_cn = cn.encode_as_ids(cn_text); t_cn = time.time() - t0
t0 = time.time(); ids_raw = raw.encode_as_ids(cn_text); t_raw = time.time() - t0
print(f"  长中文 {len(cn_text)} 字: cn-dict {t_cn * 1000:.1f}ms / "
      f"no-dict {t_raw * 1000:.1f}ms（加速 {t_raw / max(t_cn, 1e-9):.0f}x）")
check("cn-dict 编码够快 (<100ms，否则退化成 O(n^2))", t_cn < 0.1, f"{t_cn * 1000:.1f}ms")

# cn-dict 不仅是加速：它在词边界 pre-cut，会改变分词结果（避免 BPE 跨词乱 merge，
# 如「编码器和解码器」no-dict 会错切成 和解|码）。词表是带 dict 训练的，故
# pretokenize / 训练 / eval 必须全程带 dict.txt。
texts = [
    "机器翻译是自然语言处理领域的重要任务。",
    "深度学习模型在大规模语料上进行预训练，然后做下游微调。",
    "中英混合 the model 在 benchmark 上达到了 state-of-the-art 的水平。",
    "人工智能、机器学习与神经网络的关系。",
    "Transformer 架构由编码器和解码器堆叠而成，依赖自注意力机制。",
    "他说：“今天天气不错。”然后就出门了。",
]
diff = sum(cn.encode_as_ids(t) != raw.encode_as_ids(t) for t in texts)
print(f"  {len(texts)} 段文本中 {diff} 段 cn-dict 与 no-dict 分词不同 —— cn-dict 会改变")
print("  分词结果（非纯加速）。故 pretokenize/训练/eval 必须全程统一带 dict.txt。")
check("cn-dict 编码具确定性（同文本两次一致）",
      cn.encode_as_ids(cn_text) == ids_cn)

tok = PieceTokenizerWrapper(NEW_TOK)   # wrapper 应自动加载 dict.txt
check("wrapper 自动走 cn-dict 模式（编码 == cn-dict，训练/eval 入口一致）",
      tok.encode(cn_text) == ids_cn)

# ---- 2. round-trip 编解码 ----
section("2. 编解码 round-trip")
samples = [
    "机器翻译", "Hello, world!", "中英混合 mixed text 测试 123",
    "标点符号：、。！？《》「」", "The quick brown fox jumps over the lazy dog.",
    "深度学习模型在大规模语料上预训练。",
]
for s in samples:
    dec = tok.decode(tok.encode(s))
    check(f"round-trip {s[:18]!r}", dec == s, "" if dec == s else f"-> {dec!r}")

# ---- 3. 特殊 token ----
section("3. 特殊 token")
mapping = json.load(open(os.path.join(NEW_TOK, "token_mapping.json")))
for name, tid in [("<pad>", 81899), ("<user>", 81900),
                  ("<assistant>", 81901), ("<system>", 81902)]:
    got = cn.piece_to_id(name)
    check(f"{name} -> id {tid}", got == tid, f"got {got}")
check("vocab_size == 81903", tok.vocab_size == 81903, str(tok.vocab_size))
check("token_mapping.json pad_id == 81899", mapping["pad_id"] == 81899)

# ---- 4. embedding 映射正确性 ----
#
# **只对刚做完手术、一步都没训过的模型成立。** 训练会把嵌入挪走,拿训练过的
# checkpoint(比如 v18)来比会 100% 不符 —— 那不是 bug,是这项检查不适用。
# 所以默认跳过,要查得显式加 --check_embeddings(通常紧跟 `make -C prepare retok`)。
new_model = Qwen3ForCausalLM.from_pretrained(NEW_TOK, dtype=torch.bfloat16)
new_emb = new_model.model.embed_tokens.weight.data

import argparse as _ap
_a, _ = _ap.ArgumentParser(add_help=False).parse_known_args()
_check_emb = "--check_embeddings" in sys.argv
section("4. embedding 映射抽查（mean-of-BBPE 规则）")
if not _check_emb:
    print("  跳过 —— 这项只对未训练的 retok 产物有意义,加 --check_embeddings 开启")
else:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  old_tok = AutoTokenizer.from_pretrained(OLD_BASE, trust_remote_code=True)
  old_emb = AutoModelForCausalLM.from_pretrained(
      OLD_BASE, dtype=torch.bfloat16).model.embed_tokens.weight.data

  random.seed(0)
  mism = checked = one2one = multi = 0
  for i in random.sample(range(3, 81899), 300):
      try:
          piece = cn.id_to_piece(i)
      except Exception:
          continue
      if piece in ("<unk>", "<s>", "</s>"):
          continue
      text = piece.replace("▁", " ") or " "
      if not text.strip():
          text = " "
      old_ids = old_tok.encode(text, add_special_tokens=False)
      if not old_ids:
          continue
      expected = old_emb[old_ids].float().mean(0).to(old_emb.dtype)
      if not torch.equal(expected, new_emb[i]):
          mism += 1
      checked += 1
      one2one += (len(old_ids) == 1)
      multi += (len(old_ids) > 1)
  check(f"embedding 映射抽查 {checked} 个 (one2one={one2one}, multi={multi})",
        mism == 0, f"{mism} 个与规则不符")

# ---- 5. 模型 forward 数值 ----
section("5. 模型 forward 数值")
ids = tok.encode("机器翻译 hello world 深度学习模型", add_special_tokens=True)
with torch.no_grad():
    logits = new_model(torch.tensor([ids]))
check("logits 形状 (1, L, 81903)", tuple(logits.shape) == (1, len(ids), 81903),
      str(tuple(logits.shape)))
check("logits 全部有限（无 NaN/Inf）", bool(torch.isfinite(logits).all()))

print(f"\n{'=' * 60}\n验证结果: {'全部通过 ✅' if _ok else '有失败项 ❌'}\n{'=' * 60}")
raise SystemExit(0 if _ok else 1)
