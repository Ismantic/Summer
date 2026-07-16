"""v19 from-scratch init:
  - 用 Qwen3-0.6B 架构 (hidden 1024 / 28 layers / 16 heads / GQA 8)
  - 词表换成 v18 piece (81903, tie_word_embeddings=true)
  - 所有 weight 随机初始化 (Qwen3 原生 initializer_range=0.02)
  - 拷贝 v18 piece tokenizer (piece_fixed.model + dict.txt + token_mapping.json)
"""
import os
import shutil
import json
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Config

QWEN3_06B = "/home/tfbao/new/Qwen3-0.6B-Base"
V18_TIE = "/home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18_tie"   # 取 piece + dict + mapping
OUT = "/home/tfbao/new/Qwen3-0.6B-fromscratch-v19"
NEW_VOCAB = 81903

os.makedirs(OUT, exist_ok=True)

# 1. 拷 Qwen3-0.6B config + 改 vocab
config = AutoConfig.from_pretrained(QWEN3_06B, trust_remote_code=True)
print(f"原 Qwen3-0.6B-Base: vocab={config.vocab_size}, hidden={config.hidden_size}, "
      f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
      f"tie={config.tie_word_embeddings}")

config.vocab_size = NEW_VOCAB
config.tie_word_embeddings = True
config.bos_token_id = 1
config.eos_token_id = 2
config.pad_token_id = 81899  # v18 piece 的 <pad>

# 2. 用 config 实例化(weight 全随机初始化,走 Qwen3 自带的 init_weights)
import torch
torch.manual_seed(42)
model = Qwen3ForCausalLM(config)
# 显式触发 init(默认 from_config 应该会调,保险再来一遍)
model.init_weights()

# tie 验证
print(f"\ntie 验证:")
print(f"  config.tie_word_embeddings = {model.config.tie_word_embeddings}")
print(f"  embed.weight.shape = {tuple(model.model.embed_tokens.weight.shape)}")
print(f"  lm_head.weight.shape = {tuple(model.lm_head.weight.shape)}")
print(f"  embed.weight is lm_head.weight: {model.model.embed_tokens.weight is model.lm_head.weight}")

# 3. 参数统计
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
embed_params = model.model.embed_tokens.weight.numel()
transformer_params = total - embed_params  # tie 时 lm_head 复用 embed
print(f"\n参数统计:")
print(f"  embed = {embed_params/1e6:.1f}M")
print(f"  transformer = {transformer_params/1e6:.1f}M")
print(f"  total = {total/1e6:.1f}M ({total/1e9:.2f}B)")

# 4. 保存
print(f"\n保存到 {OUT}/")
model.save_pretrained(OUT, safe_serialization=True)

# 5. 拷贝 tokenizer artifacts
print(f"\n拷贝 v18 piece tokenizer:")
for f in ["piece.model", "dict.txt", "token_mapping.json",
          "special_tokens_map.json", "tokenizer_config.json"]:
    src = os.path.join(V18_TIE, f)
    dst = os.path.join(OUT, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} (missing in v18_tie)")

print(f"\nDONE: v19 init model at {OUT}")
print(f"  total: {total/1e9:.3f}B params, random-initialized")
print(f"  vocab: {NEW_VOCAB} (v18 piece, tied embed/lm_head)")
