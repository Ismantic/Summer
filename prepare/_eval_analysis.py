"""Final comparison analysis: all ckpts × all tasks (vLLM unified backend).

Generates:
- Side-by-side table of % loss vs base for each mono task
- Translation BLEU/COMET losses
- Group averages: EN mono, ZH mono, math, translation
- Comparison to ReTok / YuLan-Mini reference numbers
"""
import argparse
import json
import glob
import os
import sys

DIR = "~/Shiyu/Summer/eval_results/full"

# Default ckpt list (ordered chronologically / by experiment)
CKPTS = ["base", "v8_p1", "v10_p2", "v11_p2", "v12_p1", "v12_p2", "v15_p1", "v16_p2"]

# Task categories
EN_MONO = ["piqa", "arc_challenge", "hellaswag", "mmlu", "lambada_openai"]
ZH_MONO = ["cmmlu", "ceval-valid"]
MATH    = ["gsm8k"]
ALL_MONO = EN_MONO + ZH_MONO + MATH


def load_mono(tag, task):
    """Load result.json for a given (tag, task). Returns acc-like value or None."""
    # Try several candidate paths
    cands = [
        f"{DIR}/{tag}_vllm/{task}/result.json",
        f"{DIR}/{tag}/{task}/result.json",
    ] + glob.glob(f"{DIR}/{tag}/{task}/*/results_*.json")
    for path in cands:
        if not os.path.exists(path):
            continue
        try:
            d = json.load(open(path))
            results = d.get("results", {})
            # gsm8k -> exact_match,strict-match / exact_match,flexible-extract
            # mmlu / cmmlu -> group-level aggregate present as task name itself
            # Pick first non-stderr float metric
            for sub, m in results.items():
                if not isinstance(m, dict):
                    continue
                for key in (
                    "acc_norm,none", "acc,none",
                    "exact_match,strict-match",
                    "exact_match,flexible-extract",
                ):
                    if key in m and isinstance(m[key], (int, float)):
                        return m[key]
            # Fall through: average sub-task acc if no top-level (cmmlu/ceval groups)
            vals = []
            for sub, m in results.items():
                if isinstance(m, dict):
                    for key in ("acc_norm,none", "acc,none"):
                        if key in m and isinstance(m[key], (int, float)):
                            vals.append(m[key]); break
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
    return None


def load_translation(tag):
    """Return (zh_en_bleu, zh_en_comet, en_zh_bleu, en_zh_comet) from vLLM retro eval.
    v16 special-cased to use step2000 ckpt (mono uses 'v16_p2', translation eval was
    saved under 'v16_p2_step2000')."""
    cands = [
        f"{DIR}/{tag}_vllm/wmt22.json",
        f"{DIR}/{tag}/wmt22.json",
    ]
    if tag == "v16_p2":
        cands.append(f"{DIR}/v16_p2_step2000/wmt22.json")
    for path in cands:
        if os.path.exists(path):
            try:
                r = json.load(open(path))["results"]
                return (
                    r.get("zh-en", {}).get("bleu"),
                    r.get("zh-en", {}).get("comet"),
                    r.get("en-zh", {}).get("bleu"),
                    r.get("en-zh", {}).get("comet"),
                )
            except Exception:
                pass
    return None, None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=CKPTS)
    args = p.parse_args()

    # Load base metrics first
    base = {t: load_mono("base", t) for t in ALL_MONO}
    base_zh_b, base_zh_c, base_en_b, base_en_c = load_translation("base")

    # Header
    print("=" * 120)
    print("Mono benchmarks (% loss vs base, unified vLLM backend where possible)")
    print("=" * 120)
    cols = ["tag"] + ALL_MONO + ["EN avg", "ZH avg", "all avg"]
    fmt = "{:8s}" + "  {:>8s}" * (len(cols) - 1)
    print(fmt.format(*cols))
    print("-" * 120)

    for tag in args.ckpts:
        row = [tag]
        en_losses, zh_losses, all_losses = [], [], []
        for t in ALL_MONO:
            v = load_mono(tag, t)
            b = base.get(t)
            if v is None or b is None:
                row.append("--")
            elif tag == "base":
                row.append(f"{v:.4f}")
            else:
                loss = (v - b) / b * 100
                row.append(f"{loss:+.2f}%")
                all_losses.append(loss)
                if t in EN_MONO: en_losses.append(loss)
                if t in ZH_MONO: zh_losses.append(loss)
        # Averages
        if tag == "base":
            row += ["—", "—", "—"]
        else:
            row.append(f"{sum(en_losses)/len(en_losses):+.2f}%" if en_losses else "--")
            row.append(f"{sum(zh_losses)/len(zh_losses):+.2f}%" if zh_losses else "--")
            row.append(f"{sum(all_losses)/len(all_losses):+.2f}%" if all_losses else "--")
        print(fmt.format(*row))

    # Translation table
    print()
    print("=" * 80)
    print("Translation (vLLM, 1000-sample WMT22)")
    print("=" * 80)
    print(f"{'tag':10s}  {'zh-en BLEU':>12s}  {'zh-en COMET':>12s}  {'en-zh BLEU':>12s}  {'en-zh COMET':>12s}")
    for tag in args.ckpts:
        zb, zc, eb, ec = load_translation(tag)
        if tag == "base":
            print(f"  {tag:8s}  {zb:>12.2f}  {zc:>12.4f}  {eb:>12.2f}  {ec:>12.4f}" if zb else f"  {tag:8s} (missing)")
        else:
            parts = [f"  {tag:8s}"]
            for v, base_v in [(zb, base_zh_b), (zc, base_zh_c), (eb, base_en_b), (ec, base_en_c)]:
                if v and base_v:
                    parts.append(f"{(v-base_v)/base_v*100:>+11.2f}%")
                else:
                    parts.append(f"{'--':>12s}")
            print("  ".join(parts))

    # Reference rows
    print()
    print("=" * 80)
    print("Reference (论文报告数字)")
    print("=" * 80)
    print(f"  ReTok / Qwen1.5-0.5B   piqa +0.3%  arc -3.2%  hella -1.4%  mmlu -3.0%  avg(9 task) -1.1%")
    print(f"  YuLan-Mini             8-task avg includes gsm8k/math500/humaneval/mbpp + mmlu/arc/hella/ceval")


if __name__ == "__main__":
    main()
