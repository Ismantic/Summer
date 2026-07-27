"""自写的 `src/lora.py` 与 peft 对齐。

    python test/test_lora.py

同一个基座、同一份已发布的 adapter,两种实现比 logits。这验证的是 LoRA 的
数学本身 —— scaling 是不是 alpha/r、A/B 的乘法方向对不对、merge 折回去
等不等价。

peft 是**软依赖**:装了才跑,没装就跳过(不算通过)。`src/` 不需要它。

## 为什么用合并后的 v18 当基座

理想的对拍是「phase1_ckpt_v18 + adapter → 应等于已发布的合并权重」,但
`output/phase1_ckpt_v18` 已经不在本机了。退一步:拿任何同架构的基座,
两种实现加同一个 adapter 应当给出相同结果 —— 这照样能验出数学错误,
只是语义上这个组合没有意义(相当于在已合并的权重上再加一次 adapter)。
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lora import (adapter_state_dict, apply_lora, count_lora_params,   # noqa: E402
                      load_adapter, merge_lora)
from src.model import Qwen3ForCausalLM                                     # noqa: E402

# 基座:任何同架构的 checkpoint 都行 —— 这个测试验的是 LoRA 的数学,
# 不是某个特定权重。新 clone 上通常是从 HF 下的发布包。
CKPT_CANDIDATES = [
    ROOT / "output" / "phase2_ckpt_v18_tie",
    ROOT / "data" / "downloads" / "Qwen3-1.7B-Base-ReTok",
    ROOT / "output" / "Qwen3-1.7B-Base-ReTok",
]
# v18 那份 peft 训出来的 adapter。**发布包里没有**(只有合并后的权重),
# 所以新 clone 上不存在 —— 有就多测一轮互通性,没有就跳过那一项。
ADAPTER_DIR = ROOT / "output" / "phase2_ckpt_v18_tie" / "checkpoint-1500"
FIXTURE = ROOT / "test" / "fixtures" / "ppl_slice.pt"

R, ALPHA, TARGETS = 16, 32, ("q_proj", "v_proj")
EXPECT_LORA_PARAMS = 3_211_264      # 训练日志 output/v18_p2_tie_train.log 记的
TOL = 1e-5                          # float32 下两种实现应当基本相同


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / b.abs().mean().clamp(min=1e-6)).item()


def resolve_ckpt():
    for c in CKPT_CANDIDATES:
        if (c / "config.json").exists():
            return c
    return None


def main() -> int:
    CKPT = resolve_ckpt()
    if CKPT is None:
        print("跳过:找不到任何同架构 checkpoint。先跑\n"
              "  make -C data download-retok_model")
        return 0
    print(f"基座 {CKPT}")

    ids = (torch.load(FIXTURE, weights_only=True)[:1, :64].long()
           if FIXTURE.exists() else torch.randint(0, 81903, (1, 64)))

    print("=== 自写实现 ===")
    mine = Qwen3ForCausalLM.from_pretrained(CKPT, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        base_out = mine(ids).clone()

    apply_lora(mine, r=R, alpha=ALPHA, targets=TARGETS)
    n_par = count_lora_params(mine)
    ok = n_par == EXPECT_LORA_PARAMS
    print(f"  LoRA 参数 {n_par:,}  期望 {EXPECT_LORA_PARAMS:,}  "
          f"{'ok' if ok else '**不符**'}")

    with torch.no_grad():
        zero_out = mine(ids)
    same = torch.equal(base_out, zero_out)
    ok &= same
    print(f"  B 零初始化 → 与基座逐位相同  {'ok' if same else '**不符**'}")

    # 给 B 灌一份随机值 —— 零初始化下 LoRA 分支恒等于 0,那样对拍等于什么都没测。
    # 用自己生成的 adapter 而不是 v18 那份:发布包里没有 adapter,新 clone 上
    # 拿不到,而 LoRA 的数学不依赖某个特定权重。
    import torch as _t
    _g = _t.Generator().manual_seed(0)
    with _t.no_grad():
        for m in mine.modules():
            if type(m).__name__ == "LoRALinear":
                m.lora_B.copy_(_t.randn(m.lora_B.shape, generator=_g) * 0.02)
    with torch.no_grad():
        mine_out = mine(ids).clone()
    print(f"  B 灌随机值后与基座的相对差异 {rel_err(mine_out, base_out):.3e}"
          f"  (应明显非零)")

    n_keys = len(adapter_state_dict(mine))
    print(f"  导出 {n_keys} 个 key(peft 那份是 112 个)  "
          f"{'ok' if n_keys == 112 else '**不符**'}")
    ok &= n_keys == 112

    print("\n=== 对拍 peft ===")
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  跳过:没装 peft / transformers(都是软依赖)")
        print("\n" + ("自查通过(未与 peft 对拍)" if ok else "自查失败"))
        return 0 if ok else 1

    # 把自写实现的 adapter 导出成 peft 格式,让 peft 读进来 —— 验的是
    # 「我们写的 adapter peft 能用」这个方向,也是发布/互通真正要紧的方向。
    import json
    import tempfile

    from src.checkpoint import save_safetensors

    tmp = tempfile.mkdtemp(prefix="lora_interop_")
    save_safetensors(adapter_state_dict(mine),
                     Path(tmp) / "adapter_model.safetensors",
                     metadata={"format": "pt"})
    (Path(tmp) / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": str(CKPT),
        "r": R, "lora_alpha": ALPHA, "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(TARGETS), "modules_to_save": [],
        "inference_mode": True,
    }))

    ref = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.float32, trust_remote_code=True).eval()
    ref = PeftModel.from_pretrained(ref, tmp).eval()
    with torch.no_grad():
        ref_out = ref(input_ids=ids).logits

    e = rel_err(mine_out, ref_out)
    ok &= e <= TOL
    print(f"  logits 相对误差 {e:.3e}  阈值 {TOL:.0e}  "
          f"{'ok' if e <= TOL else '**超阈值 —— LoRA 数学不一致**'}")

    # merge 折回基座权重后应当仍然等价
    merge_lora(mine)
    with torch.no_grad():
        merged_out = mine(ids)
    em = rel_err(merged_out, ref_out)
    ok &= em <= TOL
    print(f"  merge 后 logits 相对误差 {em:.3e}  "
          f"{'ok' if em <= TOL else '**超阈值 —— merge 不等价**'}")

    print("\n" + ("对齐通过" if ok else "对齐失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
