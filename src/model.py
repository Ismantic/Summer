"""Qwen3 稠密模型,纯 torch 实现。**只依赖 torch。**

替掉 `transformers.Qwen3ForCausalLM`。规模参考(v18 用的 1.7B):

    28 层 / hidden 2048 / intermediate 6144 / 16 个 q 头 / 8 个 kv 头(GQA)
    head_dim 128 / RMSNorm eps 1e-6 / RoPE θ=1e6 / SwiGLU / 全程无 bias
    词表 81903 / 输入输出嵌入绑权重

## 不能改错的地方

**state_dict 的 key 必须与 HF 的 `Qwen3ForCausalLM` 完全一致。** 已发布的
`Ismantic/Qwen3-1.7B-Base-ReTok` 用的是标准 HF 命名;改模块名或嵌套层级会让
那份权重全部失配,**而模型照样能随机初始化跑起来、loss 照样降、不报错**。
`test/test_model_equiv.py` 是这条的防线:同权重、与 transformers 逐层对齐。

对应关系(每层 11 个张量,共 28×11+2=310):

    model.embed_tokens.weight                      → embed_tokens
    model.layers.{i}.input_layernorm.weight        → layers[i].input_layernorm
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.self_attn.{q,k}_norm.weight   ← Qwen3 特有,逐头 RMSNorm
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.norm.weight                              → norm

**没有 `lm_head.weight`。** tie_word_embeddings=true,权重与 embed 共享。

**RoPE 用 half-split(NeoX)约定,不是 interleave。** 两种都"能跑",但结果
完全不同 —— 用错了 loss 会明显偏高而不会报错。见 `_rotate_half`。

**RMSNorm 的中间计算在 float32。** 直接用 bf16 算方差会有肉眼可见的偏差,
同样不报错。
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3Config:
    vocab_size: int = 81903
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path) -> "Qwen3Config":
        raw = json.loads(Path(path).read_text())
        if raw.get("model_type") not in (None, "qwen3"):
            raise ValueError(f"不是 qwen3 架构:model_type={raw.get('model_type')}")
        # head_dim 在有些 config 里缺省,按 hidden/heads 推
        raw.setdefault("head_dim",
                       raw["hidden_size"] // raw["num_attention_heads"])
        raw.setdefault("torch_dtype", raw.get("dtype", "bfloat16"))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    @property
    def dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.torch_dtype]


class RMSNorm(nn.Module):
    """与 `Qwen3RMSNorm` 逐位对齐:方差在 float32 里算,再转回原 dtype 乘权重。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


class RotaryEmbedding(nn.Module):
    """RoPE。inv_freq 存成 buffer(非持久化 —— 它是算出来的,不进 state_dict)。"""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32)
                               / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids [B,T] → freqs [B,T,head_dim/2] → emb [B,T,head_dim]
        inv = self.inv_freq[None, :, None].to(x.device).float()      # [1,D/2,1]
        pos = position_ids[:, None, :].float()                       # [B,1,T]

        # **必须强制 float32,不能让 autocast 把这个 matmul 降到 bf16。**
        # bf16 尾数只有 8 位,整数超过 256 就不精确:257→256、1023→1024。
        # 位置编码一错,长序列上的注意力全歪 —— 而且不报错,短序列还测不出来。
        # 实测代价:seq_len=1024 上 next-token loss 从 2.3331 涨到 2.7605。
        # HF 的 Qwen3RotaryEmbedding 里也有这条防护(注释写着 Force float32)。
        dev = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            freqs = (inv.float() @ pos.float()).transpose(1, 2)      # [B,T,D/2]
            emb = torch.cat((freqs, freqs), dim=-1)                  # [B,T,D]
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """half-split(NeoX)约定:后半取负拼到前面。

    **不是 interleave(GPT-J)约定。** 两种都能跑通、都不报错,但结果不同。
    HF 的 Qwen3 用的是这个。
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                    # [B,1,T,D]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    """GQA:把 kv 头复制 n 份对齐 q 头数。n=1 时直接返回。"""
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.n_head = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv
        self.scaling = self.head_dim ** -0.5

        h, d = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * d, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.o_proj = nn.Linear(self.n_head * d, h, bias=False)
        # Qwen3 特有:对每个头的 d 维做 RMSNorm,在 RoPE 之前
        self.q_norm = RMSNorm(d, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        # 顺序要紧:proj → view 成 [B,T,H,D] → 逐头 norm → transpose → RoPE
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_head, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           scale=self.scaling)
        return self.o_proj(o.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(nn.Module):
    """对应 HF 的 `model.*` 那一层前缀。"""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.gradient_checkpointing = False

    def forward(self, input_ids, position_ids=None):
        x = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1],
                                        device=input_ids.device)[None]
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False:新式实现,能和 requires_grad=False 的
                # 冻结参数共存(Phase 1 冻结 transformer 时正是这个场景)
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.config = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, position_ids=None):
        return self.lm_head(self.model(input_ids, position_ids))

    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        self.model.gradient_checkpointing = enable

    def save_pretrained(self, out_dir) -> None:
        """存成 HF 布局:`config.json` + `model.safetensors`。

        tie 的模型不写 `lm_head.weight` —— 与上游 checkpoint 一致(310 个张量)。
        分词器产物(piece.model / dict.txt / token_mapping.json)由调用方负责
        拷过来,这里不碰文本相关的东西。
        """
        import dataclasses

        from .checkpoint import save_safetensors

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dataclasses.asdict(self.config)
        cfg["model_type"] = "qwen3"
        cfg["architectures"] = ["Qwen3ForCausalLM"]
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        save_safetensors(self.state_dict_for_save(),
                         out_dir / "model.safetensors", metadata={"format": "pt"})

    @classmethod
    def from_pretrained(cls, model_dir, device="cpu", dtype=None):
        """从 HF 格式的目录加载。key 对不上就报错,不静默跳过。"""
        from .checkpoint import load_sharded

        model_dir = Path(model_dir)
        cfg = Qwen3Config.from_json(model_dir / "config.json")
        model = cls(cfg).to(dtype or cfg.dtype)

        sd = load_sharded(model_dir, device="cpu")
        # tie 的 checkpoint 里没有 lm_head.weight —— 补上指向 embed 的引用,
        # 这样 strict=True 才不会误报缺失
        if cfg.tie_word_embeddings and "lm_head.weight" not in sd:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 手动检查而不是 strict=True:好给出能看懂的报错
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict 对不上 {model_dir}:\n"
                f"  缺少 {len(missing)} 个:{sorted(missing)[:6]}\n"
                f"  多余 {len(unexpected)} 个:{sorted(unexpected)[:6]}\n"
                f"  这类错**不会自己暴露** —— 权重没加载上模型照样跑。")
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model.to(device).eval()

    def state_dict_for_save(self) -> dict[str, torch.Tensor]:
        """导出成 HF 格式:tie 的话去掉 `lm_head.weight`,与上游 checkpoint 一致。"""
        sd = self.state_dict()
        if self.config.tie_word_embeddings:
            sd.pop("lm_head.weight", None)
        return sd
