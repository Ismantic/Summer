"""Leverage analysis on Phase 2's CUMULATIVE UPDATE matrix.

Aurora's row-uniform property is about the optimizer's UPDATE, not the
final weight. After 500 steps, the weight change from Phase 1 baseline
is tiny (~0.05% Frobenius), so the final-weight leverage distribution
is dominated by Qwen3's pretrained state and Phase 2's signal is buried.

This script computes leverage on the *update matrix* W_phase2 - W_phase1
for each MLP layer. Under Aurora's claim:
  - Aurora updates → near-uniform leverage on the update matrix
  - Muon updates → heavy-tailed leverage on the update matrix
Even with tiny update magnitude, the SHAPE of the distribution is the
mechanism-level signal.

Usage:
    python analyze_update_leverage.py \
        --base phase1_ckpt_v8 \
        --targets phase2_ckpt_v8_s2_aurora phase2_ckpt_v8_s2_muon5e5
"""
import argparse
import glob
import os

import numpy as np
import torch
from safetensors.torch import safe_open


def load_mlp(ckpt_dir):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if not files:
        raise RuntimeError(f"No safetensors in {ckpt_dir}")
    weights = {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if any(x in k for x in ("mlp.up_proj.weight",
                                        "mlp.gate_proj.weight",
                                        "mlp.down_proj.weight")):
                    weights[k] = st.get_tensor(k).float()
    return weights


def update_leverage(dW):
    """Leverage scores of the update matrix's row space."""
    m, n = dW.shape
    if m < n:
        dW = dW.T
        m, n = n, m
    U, S, _ = torch.linalg.svd(dW, full_matrices=False)
    eps = float(S.max()) * 1e-6 if float(S.max()) > 0 else 0.0
    valid = S > eps
    U = U[:, valid]
    lev = (U * U).sum(dim=-1).numpy()
    target = float(valid.sum()) / m
    return lev, target


def summarize(lev, target):
    cv = lev.std() / lev.mean() if lev.mean() > 0 else 0.0
    sorted_lev = np.sort(lev)
    n = len(sorted_lev)
    p10 = sorted_lev[: max(1, n // 10)].mean()
    p90 = sorted_lev[-max(1, n // 10):].mean()
    return dict(
        n=n, target=target, mean=lev.mean(), std=lev.std(), cv=cv,
        p10=p10, p90=p90, p10_over_p90=p10 / p90 if p90 > 0 else 0.0,
        min=lev.min(), max=lev.max(), max_over_target=lev.max() / target if target > 0 else 0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--targets", nargs="+", required=True)
    args = p.parse_args()

    root = "/home/tfbao/Shiyu/Summer/output"
    base = load_mlp(os.path.join(root, args.base))
    print(f"Base: {args.base} ({len(base)} MLP tensors)")

    results = {}
    for tag in args.targets:
        print(f"\n=== {tag} update leverage ===")
        target = load_mlp(os.path.join(root, tag))
        by_kind = {"up": [], "gate": [], "down": []}

        for k in base:
            if k not in target:
                continue
            dW = target[k] - base[k]
            mag = dW.norm() / base[k].norm()
            kind = "up" if "up_proj" in k else "gate" if "gate_proj" in k else "down"
            lev, t = update_leverage(dW)
            by_kind[kind].append((lev, t, mag))

        # per-kind aggregate
        all_lev = []
        for kind, items in by_kind.items():
            if not items:
                continue
            levs = np.concatenate([x[0] for x in items])
            target_avg = np.mean([x[1] for x in items])
            mag_avg = np.mean([x[2] for x in items])
            s = summarize(levs, target_avg)
            print(f"  {kind:5s} | dW_norm/W={mag_avg*100:.4f}%  "
                  f"CV={s['cv']:.3f}  p10/p90={s['p10_over_p90']:.3f}  "
                  f"max/target={s['max_over_target']:.2f}")
            all_lev.append(levs)

        cat_all = np.concatenate(all_lev)
        avg_target = np.mean([x[1] for kind in by_kind.values() for x in kind])
        s_all = summarize(cat_all, avg_target)
        avg_mag = np.mean([x[2] for kind in by_kind.values() for x in kind])
        print(f"  ALL   | dW_norm/W={avg_mag*100:.4f}%  "
              f"CV={s_all['cv']:.3f}  p10/p90={s_all['p10_over_p90']:.3f}  "
              f"max/target={s_all['max_over_target']:.2f}")
        results[tag] = s_all

    print("\n=== Comparison (update matrix leverage, all MLP) ===")
    print(f"{'tag':<35} {'CV':>8} {'p10/p90':>9} {'max/target':>12}")
    for tag, s in results.items():
        print(f"{tag:<35} {s['cv']:>8.3f} {s['p10_over_p90']:>9.3f} "
              f"{s['max_over_target']:>12.2f}")
    print("\nLower CV / higher p10/p90 = more uniform leverage = Aurora signal")


if __name__ == "__main__":
    main()
