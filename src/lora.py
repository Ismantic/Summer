"""LoRA。**只依赖 torch。**

替掉 `peft`。v18 只用到 peft 的一个很小的子集 —— r=16、alpha=32、
target q_proj/v_proj、dropout 0、无 bias、不用 dora/rslora/qalora。
peft 那些 `alpha_pattern` / `corda_config` / `qalora_group_size` 一个都没用上。

    y = W x + (alpha / r) · B (A x)

    lora_A  [r, in]    正态初始化(peft 默认 kaiming_uniform,见下)
    lora_B  [out, r]   零初始化 —— 保证接上去的瞬间等价于原模型

## 与 peft 的 checkpoint 互通

已发布的 `output/phase2_ckpt_v18_tie/checkpoint-1500/adapter_model.safetensors`
是 peft 0.19.1 写的,112 个张量,key 长这样:

    base_model.model.model.layers.{i}.self_attn.{q,v}_proj.lora_{A,B}.weight

`load_adapter()` 直接吃这个格式,`adapter_state_dict()` 也按这个格式导出 ——
所以自写实现和 peft 训出来的权重可以互换。

## 不能改错的地方

**scaling 是 alpha/r,不是 alpha。** v18 是 32/16 = 2.0。写成 alpha 会让
adapter 的作用放大 16 倍 —— 不报错,只是结果全乱。

**lora_B 必须零初始化。** 否则模型在训练第一步之前就已经偏离基座了,
「LoRA 接上去等价于原模型」这个前提不成立。
"""
import math

import torch
import torch.nn as nn

PEFT_PREFIX = "base_model.model."


class LoRALinear(nn.Module):
    """包住一个 `nn.Linear`,旁路加低秩分支。基座权重保持冻结。"""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r 必须为正,收到 {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # 跟着基座的 device 建 —— apply_lora() 通常在模型已经搬到 GPU 之后调用,
        # 用 torch.empty 的默认值会建在 CPU 上,然后在 forward 里炸设备不匹配。
        # dtype 用 float32:低秩因子本身很小,全精度更稳,也与已发布的 adapter
        # (peft 写的,F32)一致;forward 里再转回基座 dtype。
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features,
                                               device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               device=dev, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_lora()

        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_lora(self) -> None:
        # 与 peft 一致:A 用 kaiming_uniform(a=√5),B 置零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.dropout(x).to(self.lora_A.dtype)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(h, self.lora_A), self.lora_B)
        return out + self.scaling * delta.to(out.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """基座权重 + scaling·B@A,dtype 跟基座走。"""
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def _iter_targets(model: nn.Module, targets):
    """产出 (父模块, 属性名, 全限定名) —— 名字在 targets 里的 nn.Linear。"""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                yield mod, child_name, f"{name}.{child_name}" if name else child_name


def apply_lora(model: nn.Module, r: int = 16, alpha: int = 32,
               targets=("q_proj", "v_proj"), dropout: float = 0.0) -> list[str]:
    """就地把命中的 Linear 换成 LoRALinear,返回被替换的全限定名。"""
    targets = set(targets)
    replaced = []
    for parent, attr, fqn in _iter_targets(model, targets):
        setattr(parent, attr, LoRALinear(getattr(parent, attr), r, alpha, dropout))
        replaced.append(fqn)
    if not replaced:
        raise ValueError(f"没有任何模块命中 {sorted(targets)} —— 检查 target 名字")
    return replaced


def lora_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """按 peft 的命名导出,能被 peft 直接加载。"""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            base = f"{PEFT_PREFIX}{name}"
            out[f"{base}.lora_A.weight"] = m.lora_A.detach()
            out[f"{base}.lora_B.weight"] = m.lora_B.detach()
    return out


def load_adapter(model: nn.Module, path, strict: bool = True) -> int:
    """加载 peft 格式的 adapter。返回加载上的模块数。

    key 对不上就报错。默默跳过会让「adapter 没生效」这件事完全看不出来 ——
    模型照样跑,只是等于没微调。
    """
    from .checkpoint import load_safetensors

    sd = load_safetensors(path)
    mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not mods:
        raise RuntimeError("模型里没有 LoRALinear —— 先 apply_lora() 再加载")

    seen, missing = set(), []
    for name, m in mods.items():
        ka = f"{PEFT_PREFIX}{name}.lora_A.weight"
        kb = f"{PEFT_PREFIX}{name}.lora_B.weight"
        if ka in sd and kb in sd:
            with torch.no_grad():
                m.lora_A.copy_(sd[ka].to(m.lora_A.dtype))
                m.lora_B.copy_(sd[kb].to(m.lora_B.dtype))
            seen |= {ka, kb}
        else:
            missing.append(name)

    extra = sorted(set(sd) - seen)
    if strict and (missing or extra):
        raise RuntimeError(
            f"adapter 对不上 {path}:\n"
            f"  模型里有、文件里没有:{missing[:5]}\n"
            f"  文件里有、模型里没有:{extra[:5]}\n"
            f"  这类错**不会自己暴露** —— adapter 没加载上模型照样跑。")
    return len(mods) - len(missing)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """把低秩分支折回基座权重,并还原成普通 `nn.Linear`。返回合并的模块数。

    合并之后 state_dict 就回到标准 HF 命名了 —— 这正是
    `output/phase2_ckpt_v18_tie/model.safetensors` 的由来(310 个张量,
    没有任何 lora_ 前缀)。
    """
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.data = child.merged_weight()
                base.weight.requires_grad_(True)
                setattr(mod, child_name, base)
                n += 1
    return n


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))
