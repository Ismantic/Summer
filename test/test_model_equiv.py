"""自写的 `src/model.py` 与 transformers 的 `Qwen3ForCausalLM` 对齐。

    python test/test_model_equiv.py
    python test/test_model_equiv.py --model_path <ckpt>

**这是阶段 3 的安全网。** `test_reproduce_sota.py` 的 trans/mono 两项走 vLLM,
vLLM 用它自己的 Qwen3 实现,根本不经过我们的代码 —— 那两项全绿也说明不了
模型写对了。能锚住 `src/model.py` 的只有:

  1. 这个文件:同权重、逐层 hidden state + 最终 logits 对齐 transformers
  2. `test_reproduce_sota.py --only ppl`:固定切片上的 next-token loss

两个都需要。逐层对齐能**定位**错在哪一层,PPL 只告诉你错了。

## 为什么跑两个 dtype

**float32 才是结构判据。** 2026-07-27 实测同一份权重:

    float32    层0 8.4e-07   最大 1.4e-06   logits 4.5e-07
    bfloat16   层0 3.9e-03   最大 1.4e-02   logits 3.9e-03

bf16 那 1.4e-2 不是 bug —— 换成 float32 就掉到 1e-6,说明两边算的是同一个
函数,差异全来自算子融合和累加顺序(SDPA 的内核选择、RMSNorm 融不融合)。
所以:

  - float32 卡到 1e-5:**结构写错了一定过不去**。RoPE 用错 interleave 约定、
    漏掉 q/k norm、GQA 复制方向反了,都会让误差大好几个数量级
  - bf16 放到 3e-2:只用来发现「数值上离谱」,不用来判对错

一开始我把 bf16 的阈值定成 5e-3,报了"模型写错了" —— 是阈值定错了。
误差从层 0 起**平缓累积**而不是某层突然跳变,这个形状本身就说明是数值问题;
结构性错误会在第一层就炸。

transformers 是**软依赖**:装了才跑这个测试,没装就跳过(不算通过)。
`src/` 本身不依赖它。
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import Qwen3ForCausalLM                          # noqa: E402

DEFAULT_CKPT = ROOT / "output" / "phase2_ckpt_v18_tie"
FIXTURE = ROOT / "test" / "fixtures" / "ppl_slice.pt"

# (逐层 hidden 上限, logits 上限)。依据见模块开头那张实测表。
TOL = {
    torch.float32:  (1e-5, 1e-5),    # 结构判据,写错了一定过不去
    torch.bfloat16: (3e-2, 1e-2),    # 只查「数值上离谱」
}
SKIPPED = -1        # 缺输入而没跑。绝不能和「跑了且通过」混为一谈


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = b.abs().mean().clamp(min=1e-6)
    return ((a - b).abs().mean() / denom).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(DEFAULT_CKPT))
    p.add_argument("--rows", type=int, default=2, help="用 fixture 的前几行")
    p.add_argument("--seq", type=int, default=256, help="每行截断到多长")
    a = p.parse_args()

    ckpt = Path(a.model_path)
    if not (ckpt / "config.json").exists():
        print(f"跳过:{ckpt} 下没有 config.json")
        return 0
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("跳过:没装 transformers(它是软依赖,src/ 不需要它)")
        return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ckpt   {ckpt}")
    print(f"设备   {dev}")

    if FIXTURE.exists():
        ids = torch.load(FIXTURE, weights_only=True)[:a.rows, :a.seq].long()
    else:
        print(f"  {FIXTURE} 不在,退回随机 token")
        ids = torch.randint(0, 81903, (a.rows, a.seq))
    ids = ids.to(dev)
    print(f"输入   {tuple(ids.shape)}")

    ok = True
    for dt in (torch.float32, torch.bfloat16):
        tol_h, tol_l = TOL[dt]
        name = str(dt).replace("torch.", "")
        print(f"\n{'=' * 58}\n=== {name}(hidden 阈值 {tol_h:.0e} / "
              f"logits {tol_l:.0e})\n{'=' * 58}")

        mine = Qwen3ForCausalLM.from_pretrained(ckpt, device=dev, dtype=dt).eval()
        ref = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=dt, trust_remote_code=True).to(dev).eval()

        mine_h, ref_h, hooks = [], [], []
        for layer in mine.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, buf=mine_h: buf.append(o.detach())))
        for layer in ref.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, buf=ref_h: buf.append(
                    (o[0] if isinstance(o, tuple) else o).detach())))
        with torch.no_grad():
            mine_logits, ref_logits = mine(ids), ref(input_ids=ids).logits
        for h in hooks:
            h.remove()

        errs = [rel_err(x, y) for x, y in zip(mine_h, ref_h)]
        worst_i = max(range(len(errs)), key=errs.__getitem__)
        for i in (0, 1, len(errs) // 2, len(errs) - 1):
            print(f"  层 {i:2d}  {errs[i]:.3e}")
        bad = [i for i, e in enumerate(errs) if e > tol_h]
        print(f"  最大 {errs[worst_i]:.3e} 在层 {worst_i} | "
              f"超阈值 {len(bad)}/{len(errs)} 层 "
              f"{'ok' if not bad else '**' + str(bad[:5]) + '**'}")

        e_logits = rel_err(mine_logits, ref_logits)
        agree = (mine_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()
        print(f"  logits {e_logits:.3e} {'ok' if e_logits <= tol_l else '**超阈值**'}"
              f" | argmax 一致率 {agree:.4f}")

        ok &= (not bad) and e_logits <= tol_l
        del mine, ref
        if dev == "cuda":
            torch.cuda.empty_cache()

    ok &= check_long_autocast(ckpt, dev)
    print("\n" + ("对齐通过" if ok else "对齐失败 —— 模型写错了"))
    return 0 if ok else 1


def check_long_autocast(ckpt: Path, dev: str) -> bool:
    """长序列 + autocast —— 补上一个真实踩过的盲区。

    2026-07-27:RoPE 里的 `inv_freq @ position_ids` 没强制 float32,被 autocast
    降到了 bf16。bf16 尾数 8 位,整数超过 256 就不精确(257→256、1023→1024),
    位置编码错乱。代价是 seq_len=1024 上 loss 从 2.3331 涨到 2.7605。

    **上面那两轮 dtype 对齐测不出来**:序列只有 256,而且没开 autocast。
    这个函数专门盯这个组合。
    """
    from transformers import AutoModelForCausalLM

    print(f"\n{'=' * 58}\n=== 长序列 + autocast(RoPE 精度回归)\n{'=' * 58}")
    if not FIXTURE.exists():
        print("  跳过:缺 fixture")
        return True

    ids = torch.load(FIXTURE, weights_only=True)[:2, :1024].long().to(dev)
    mine = Qwen3ForCausalLM.from_pretrained(ckpt, device=dev,
                                            dtype=torch.bfloat16).eval()
    ref = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16):
        e = rel_err(mine(ids), ref(input_ids=ids).logits)
    del mine, ref
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 位置编码错乱时这个值会到 1e-1 量级,正常时与短序列同量级
    tol = 2e-2
    print(f"  seq={ids.shape[1]} autocast=bf16  logits 相对误差 {e:.3e}  "
          f"阈值 {tol:.0e}  {'ok' if e <= tol else '**超阈值 —— 查 RoPE 是否被降精度**'}")
    return e <= tol


if __name__ == "__main__":
    sys.exit(main())
