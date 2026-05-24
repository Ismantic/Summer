"""BERT-base init:
  - 12 layers / 768 hidden / 12 heads / 3072 ff
  - vocab 81904 (v18 piece 的 81903 + 1 个 <mask> 在末位)
  - max_position 1024(沿用 v18 数据 seqlen)
  - 随机初始化(BERT 原生 init,std=0.02)
"""
import os, shutil, torch
from transformers import BertConfig, BertForMaskedLM

V18_PIECE_DIR = "/home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18_tie"  # 取 piece.model + dict.txt
OUT = "/home/tfbao/Shiyu/Summer/BERT/bert_init"
PIECE_VOCAB = 81903
MASK_TOKEN_ID = 81903   # 新增,在 81903 这一行(piece 数据不会出现这个 id)
TOTAL_VOCAB = PIECE_VOCAB + 1  # 81904

os.makedirs(OUT, exist_ok=True)

config = BertConfig(
    vocab_size=TOTAL_VOCAB,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=1024,  # 沿用 v18 数据 seqlen
    type_vocab_size=1,             # MLM only,no NSP → 单段
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=81899,            # v18 piece 的 <pad>
)

torch.manual_seed(42)
model = BertForMaskedLM(config)
model.init_weights()

total = sum(p.numel() for p in model.parameters())
print(f"BERT-base init created:")
print(f"  vocab_size: {TOTAL_VOCAB} (piece {PIECE_VOCAB} + 1 mask at id {MASK_TOKEN_ID})")
print(f"  hidden: {config.hidden_size}, layers: {config.num_hidden_layers}, heads: {config.num_attention_heads}")
print(f"  max_position: {config.max_position_embeddings}")
print(f"  total params: {total/1e6:.1f}M")
print(f"  embed: {model.bert.embeddings.word_embeddings.weight.numel()/1e6:.1f}M")

model.save_pretrained(OUT, safe_serialization=True)
print(f"\nSaved to {OUT}/")

# 拷贝 piece tokenizer artifacts(虽然 BERT 不直接用 wrapper,但便于以后 inference)
for f in ["piece.model", "dict.txt", "token_mapping.json",
          "special_tokens_map.json", "tokenizer_config.json"]:
    src = os.path.join(V18_PIECE_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, f))
        print(f"  copied {f}")

# 写一个 mask_token_id 记录
with open(os.path.join(OUT, "mask_token_id.txt"), "w") as f:
    f.write(str(MASK_TOKEN_ID))
print(f"\nmask_token_id = {MASK_TOKEN_ID} (recorded in mask_token_id.txt)")
