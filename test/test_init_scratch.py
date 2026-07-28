"""从零初始化的判据 —— 防的是「根本没初始化」这一类错。

## 为什么需要单独一条

`src/model.py` 在 2026-07-28 之前**完全没有初始化代码**:`__init__` 里
建完 `nn.Embedding` / `nn.Linear` 就直接用 torch 的默认值。ReTok 那条路
永远从 checkpoint 加载权重,所以这个洞一直不显形;从零预训练一走上去,
拿到的就是 `nn.Embedding` 的 N(0,1) —— 标准差比该有的大 50 倍。

**这类错不报错。** 模型照样前向、loss 照样降,只是慢、抖、偶尔崩,
回头查不出原因。现有五项测试一条都拦不住它:

    equiv / lora / retok / ppl   都是「加载已有权重之后」比对
    tok                          只管分词器

所以判据必须直接落在**初始化之后、训练之前**的那些统计量上。

## 四条判据

1. 嵌入标准差 ≈ initializer_range(错成 torch 默认就是 1.0,差 50 倍)
2. 残差出口(o_proj / down_proj)≈ std/√(2L)(少了这条深层激活会越叠越大)
3. RMSNorm 权重全 1(错成随机数的话第一步就崩)
4. 初始 loss 落在 ln(V) 附近 —— 均匀分布是理论下界,好太多说明 logits
   量级不对,差太多说明初始化太大
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import Qwen3Config, Qwen3ForCausalLM        # noqa: E402

VOCAB = 4096
LAYERS = 6
STD = 0.02


def build():
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=256, intermediate_size=768,
                      num_hidden_layers=LAYERS, num_attention_heads=8,
                      num_key_value_heads=4, head_dim=32,
                      tie_word_embeddings=True, torch_dtype="float32",
                      initializer_range=STD)
    m = Qwen3ForCausalLM(cfg).to(torch.float32)
    m.init_weights(seed=0)
    m.lm_head.weight = m.model.embed_tokens.weight
    return cfg, m


def main():
    cfg, m = build()
    fails = []

    def check(name, ok, detail):
        print(f"  {'ok ' if ok else '**FAIL**'} {name}: {detail}")
        if not ok:
            fails.append(name)

    print("从零初始化判据")

    emb = m.model.embed_tokens.weight
    check("嵌入 std", abs(emb.std().item() - STD) < 0.1 * STD,
          f"{emb.std().item():.5f}  (期望 {STD}, torch 默认会是 1.0)")

    resid_std = STD / math.sqrt(2 * LAYERS)
    for tag, w in (("o_proj", m.model.layers[0].self_attn.o_proj.weight),
                   ("down_proj", m.model.layers[0].mlp.down_proj.weight)):
        check(f"残差出口 {tag} std", abs(w.std().item() - resid_std) < 0.15 * resid_std,
              f"{w.std().item():.5f}  (期望 {resid_std:.5f} = {STD}/√{2 * LAYERS})")

    q = m.model.layers[0].self_attn.q_proj.weight
    check("非残差出口 q_proj std", abs(q.std().item() - STD) < 0.15 * STD,
          f"{q.std().item():.5f}  (期望 {STD} —— 不该被残差缩放)")

    norms = [(n, p) for n, p in m.named_parameters()
             if "norm" in n or "layernorm" in n]
    bad = [n for n, p in norms if not torch.allclose(p, torch.ones_like(p))]
    check("RMSNorm 全 1", not bad, f"{len(norms)} 个,异常 {len(bad)} 个")

    # 逐层激活 RMS:残差缩放对了应该是 √深度 的增长,不是指数
    x = m.model.embed_tokens(torch.randint(0, VOCAB, (2, 128)))
    cos, sin = m.model.rotary_emb(x, torch.arange(128)[None])
    rms0 = x.pow(2).mean().sqrt().item()
    for layer in m.model.layers:
        x = layer(x, cos, sin)
    growth = x.pow(2).mean().sqrt().item() / rms0
    check("激活增长倍数", growth < 6 * math.sqrt(LAYERS),
          f"{growth:.1f}×  (√深度量级;爆炸的话是残差缩放没生效)")

    ids = torch.randint(0, VOCAB, (2, 128))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(
            m(ids)[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)).item()
    uniform = math.log(VOCAB)
    check("初始 loss ≈ ln(V)", uniform - 0.1 < loss < uniform + 1.0,
          f"{loss:.4f}  (ln({VOCAB}) = {uniform:.4f})")

    if fails:
        print(f"\n{len(fails)} 条不过:{fails}")
        return 1
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
