"""复现 CLAUDE.md 里公布的 SOTA 数字。

改了 src/ 或 prepare/ 之后跑这个 —— 拿**真实的 SOTA checkpoint** 跑完整评测,
对照记录的数字。这是判断有没有改坏的硬标准:单元级的对拍只能说明某个函数
没写错,说明不了整条链对。

    python test/test_reproduce_sota.py              # 三项都跑
    python test/test_reproduce_sota.py --only ppl   # 只跑 PPL(几秒)

三项各防一段链路,**不能互相替代**:

| | 走什么 | 防什么 |
|---|---|---|
| `ppl` | 自己的 forward | **模型实现**。阶段 3 换成纯 torch 后唯一能锚住它的 |
| `trans` | vLLM | 词表 / 数据 / 导出链路。BLEU 是这个项目的头号指标 |
| `mono` | vLLM + lm_eval | 六个 benchmark 的整体退化 |

**注意 `trans` 和 `mono` 测不到 `src/model.py`** —— vLLM 从 checkpoint 直接加载
权重,用的是它自己的 Qwen3 实现,根本不经过我们的代码。所以阶段 3 之后
只有 `ppl` 这一项能发现模型写错了。

耗时:ppl 几秒,trans 约 5 分钟,mono 约 36 分钟。

评测依赖两个 venv(原因见 `_attic/MANIFEST.md`「试过合并成单个 venv,不行」):

    python test/test_reproduce_sota.py --only ppl                        # .venv
    python test/test_reproduce_sota.py --python ~/.venv-eval/bin/python

这个文件在四层改造期间就位,目的是让每个阶段结束都有东西可跑。脚本位置
用多处试探(改造前在 evals/,改造后在 prepare/ 或 src/),所以搬目录不用改它。
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 期望值
#
# 来源:2026-05-24 训出 v18_p2_tie 那次,2026-07-27 在 .venv-eval 上逐位复现
# 通过(Δ=0.000000)。产物 eval_results/translate_wmt22/ 下的两份 json。
# CLAUDE.md 表现表里的数字以这里为可执行来源。

EXPECT_TRANS = {
    "zh-en": {"bleu": 20.4599, "comet": 0.7933},
    "en-zh": {"bleu": 36.0314, "comet": 0.8444},
}

EXPECT_MONO = {
    "lambada_openai": ("acc,none", 0.5768),
    "piqa":           ("acc_norm,none", 0.7367),
    "arc_challenge":  ("acc_norm,none", 0.5145),
    "hellaswag":      ("acc_norm,none", 0.6389),
    "ceval-valid":    ("acc,none", 0.6204),
    # gsm8k 恒在 0 附近 —— 81903 piece 词表打断了 Qwen3 的数值推理,两个 phase
    # 都救不回来。这是词表代价不是回归,所以容差放宽,只防「彻底崩掉」。
    "gsm8k":          ("exact_match,strict-match", 0.0349),
}

# PPL 锚点。2026-07-27 用当时的 transformers 路径(evals/eval_ppl.py +
# AutoModelForCausalLM)在固定切片 test/fixtures/ppl_slice.pt 上测得,连跑两次
# 逐位一致。
#
# **阶段 3 把 src/ 改成纯 torch 之后,这个数字是判断自写 Qwen3 对不对的硬标准。**
# 必须趁 transformers 还在链路里的时候抓 —— 改完再抓就没有参照了。
EXPECT_PPL_LOSS = 2.333104555907487      # PPL = 10.3099
PPL_FIXTURE = ROOT / "test" / "fixtures" / "ppl_slice.pt"
PPL_BATCH = 8

# 容差按实测噪声定,不是拍的。
#
# **vLLM 的贪心解码不可复现。** 2026-07-27 用同一 ckpt、同一条命令跑了 6 次:
#
#   zh-en BLEU  20.4081 – 20.5114   range 0.1033   sd 0.022
#   en-zh BLEU  36.0314 – 36.1651   range 0.1337   sd 0.056
#   zh-en COMET 0.7932  – 0.7942    range 0.0010   sd 0.0005
#   en-zh COMET 0.8441  – 0.8447    range 0.0005   sd 0.0002
#
# 原因是连续批处理 + chunked prefill + 异步调度,每次 batch 组成不同 → bf16
# 归约顺序不同 → 接近平局的 argmax 偶尔翻转。temperature=0 挡不住这个。
# COMET 稳两个数量级,因为它是连续打分,不被个别 token 翻转放大。
#
# 推论:**checkpoint 之间 0.1 量级的 BLEU 差异是噪声,不能当结论。**
# CLAUDE.md 里 v18 20.46 vs base 22.34(−1.88)远在噪声之上,那个结论不受影响。
#
# 容差取实测 range 的约 2 倍。这么松的 BLEU 只能当「链路没断」的检查,做不了
# 精密仪器 —— 精密的那一项是下面的 TOL_LOSS。
TOL_BLEU = 0.25
TOL_COMET = 0.003
TOL_ACC = 0.010
TOL_GSM8K = 0.020

# TOL_LOSS 是给阶段 3 用的,走的是自己的 forward,不经过 vLLM,所以能收很紧:
# 换成自己写的实现之后 bf16 累加顺序不同,不会再逐位一致,但正确时差异在 1e-4
# 量级。而 RoPE θ 写错、漏掉 q/k norm 这类 bug 会让 loss 跳一个数量级。
TOL_LOSS = 0.005

SKIPPED = -1     # 缺输入而没跑。绝不能和「跑了且通过」混为一谈

# 哪些项因为缺输入而没跑。**全都跳过时收尾必须说「什么都没测」** ——
# 否则新 clone 上会看到「全部通过」而其实一项没跑,那比失败更糟。
_SKIPPED: list[str] = []

TRANS_ARGS = dict(testset="wmt22", exemplar_set="wmt21", num_fewshot=5,
                  max_samples=1000)
MONO_TASKS = "lambada_openai:0,piqa:5,arc_challenge:25,hellaswag:10,ceval-valid:5,gsm8k:5"


def first_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_ckpt() -> Path | None:
    """SOTA checkpoint。传父目录(合并后的完整权重),不是 checkpoint-1500/。

    checkpoint-1500/ 里是 LoRA adapter;父目录才是合并好的 Qwen3ForCausalLM,
    也是 2026-05-24 产出 20.46 那次实际喂给 eval 的东西。
    """
    return first_existing(
        ROOT / "save" / "sota" / "v18_p2_tie",
        ROOT / "output" / "phase2_ckpt_v18_tie",
        # 从 HF 下的发布包 —— 新 clone 上通常只有这个
        ROOT / "data" / "downloads" / "Qwen3-1.7B-Base-ReTok",
        ROOT / "output" / "Qwen3-1.7B-Base-ReTok",
    )


def resolve_trans_script() -> Path | None:
    return first_existing(
        ROOT / "prepare" / "translate.py",
        ROOT / "evals" / "eval_pretrain_translate_vllm.py",
    )


def resolve_mono_script() -> Path | None:
    return first_existing(
        ROOT / "prepare" / "benchmark.py",
        ROOT / "evals" / "eval_with_piece_vllm.py",
    )


def resolve_ppl_script() -> Path | None:
    return first_existing(
        ROOT / "src" / "evaluate.py",
        ROOT / "evals" / "eval_ppl.py",
    )


def resolve_comet() -> Path | None:
    """COMET 模型目录。优先注册表登记的位置,再退回本机旧路径。

    旧路径是 v18 那次实跑用的,留作兼容 —— 但注册表(data/source.py)才是
    「从零重建时它应该在哪」的唯一说法。
    """
    sys.path.insert(0, str(ROOT))
    try:
        from data import source
        registered = source.get("comet").dir()
    except Exception:                                    # noqa: BLE001
        registered = None
    return first_existing(
        *(p for p in (registered,) if p is not None),
        Path("~/a6000/Summer-data/comet-wmt22-da"),
    )


def run(cmd: list[str]) -> bool:
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode == 0


def check(name: str, got: float, want: float, tol: float) -> bool:
    if got == SKIPPED:
        print(f"  {name:34} 跳过(缺输入)")
        return True
    d = got - want
    ok = abs(d) <= tol
    print(f"  {name:34} {got:9.4f}  期望 {want:8.4f}  Δ{d:+8.4f}  "
          f"{'ok' if ok else '不符 (容差 %.3f)' % tol}")
    return ok


def test_ppl(py: str) -> bool:
    """固定切片上的 next-token loss。阶段 3 之后这是唯一锚住 src/model.py 的项。"""
    print("\n=== 固定切片 PPL(经过模型 forward) ===")
    ckpt, script = resolve_ckpt(), resolve_ppl_script()
    if ckpt is None or script is None or not PPL_FIXTURE.exists():
        print(f"  跳过:ckpt={ckpt} script={script} fixture={PPL_FIXTURE.exists()}")
        _SKIPPED.append("ppl")
        return True

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)
    if not run([py, str(script), "--model_path", str(ckpt),
                "--valid_pt", str(PPL_FIXTURE), "--batch_size", str(PPL_BATCH),
                "--output_path", str(out)]):
        print("  评测脚本非零退出")
        return False

    got = json.loads(out.read_text())["loss"]
    out.unlink(missing_ok=True)
    return check("next-token loss", got, EXPECT_PPL_LOSS, TOL_LOSS)


def test_translate(py: str) -> bool:
    print("\n=== WMT22 翻译 (BLEU + COMET) ===")
    ckpt, script = resolve_ckpt(), resolve_trans_script()
    if ckpt is None or script is None:
        print(f"  跳过:ckpt={ckpt} script={script}")
        _SKIPPED.append("trans")
        return True

    comet = resolve_comet()
    if comet is None:
        print("  COMET 模型不在,只比 BLEU")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)
    cmd = [py, str(script), "--model_path", str(ckpt), "--direction", "both",
           "--output_path", str(out)]
    for k, v in TRANS_ARGS.items():
        cmd += [f"--{k}", str(v)]
    if comet is not None:
        # --compute_comet 必须配 --save_all_samples:COMET 要对全部译文打分,
        # 不加的话脚本只留 5 条,会 WARN 一句然后跳过 COMET(不报错)。
        cmd += ["--compute_comet", "--save_all_samples",
                "--comet_model_path", str(comet)]
    if not run(cmd):
        print("  评测脚本非零退出")
        return False

    res = json.loads(out.read_text())["results"]
    out.unlink(missing_ok=True)
    ok = True
    for d, want in EXPECT_TRANS.items():
        ok &= check(f"{d} BLEU", res[d]["bleu"], want["bleu"], TOL_BLEU)
        got_comet = res[d].get("comet", SKIPPED) if comet else SKIPPED
        ok &= check(f"{d} COMET", got_comet, want["comet"], TOL_COMET)
    return ok


def test_mono(py: str) -> bool:
    print("\n=== mono 六任务 ===")
    ckpt, script = resolve_ckpt(), resolve_mono_script()
    if ckpt is None or script is None:
        print(f"  跳过:ckpt={ckpt} script={script}")
        _SKIPPED.append("mono")
        return True

    with tempfile.TemporaryDirectory() as d:
        if not run([py, str(script), "--model_path", str(ckpt),
                    "--tasks", MONO_TASKS, "--output_dir", d]):
            print("  评测脚本非零退出")
            return False
        ok = True
        for task, (metric, want) in EXPECT_MONO.items():
            f = Path(d) / task / "result.json"
            if not f.exists():
                print(f"  {task:34} 结果文件没生成")
                ok = False
                continue
            got = json.loads(f.read_text())["results"][task][metric]
            tol = TOL_GSM8K if task == "gsm8k" else TOL_ACC
            ok &= check(f"{task} {metric.split(',')[0]}", got, want, tol)
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["ppl", "trans", "mono"],
                   help="只跑其中一项")
    p.add_argument("--python", default=sys.executable,
                   help="跑评测用的解释器。trans/mono 要 vllm,在 .venv-eval 里;"
                        "ppl 只要 torch,.venv 就行")
    a = p.parse_args()

    print(f"仓库    {ROOT}")
    print(f"解释器  {a.python}")
    print(f"ckpt    {resolve_ckpt()}")

    ok = True
    if a.only in (None, "ppl"):
        ok &= test_ppl(a.python)
    if a.only in (None, "trans"):
        ok &= test_translate(a.python)
    if a.only in (None, "mono"):
        ok &= test_mono(a.python)

    asked = {"ppl", "trans", "mono"} if a.only is None else {a.only}
    ran = asked - set(_SKIPPED)
    if not ran:
        print(f"\n**一项都没跑**(跳过 {sorted(_SKIPPED)})—— 这不是通过。")
        print("  缺 checkpoint 就先:make -C data download-retok_model")
        return 1
    tail = f"  (跳过 {sorted(_SKIPPED)})" if _SKIPPED else ""
    print("\n" + (f"通过 {sorted(ran)}{tail}" if ok else f"有不符项 —— 改坏了{tail}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
