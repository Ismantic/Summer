"""Leverage-score analysis — replicate Aurora paper Figure 19 logic on
our Phase 2 checkpoints.

For each MLP weight matrix W ∈ R^{m,n} (m ≥ n after orientation), the
leverage of row i is ℓᵢ = ||U_i,:||²₂ where U is the left-singular-vector
matrix of W. Uniform leverage is ℓ = n/m for all rows (Aurora's target).
Heavy-tailed leverage indicates "neuron death" — rows that barely
participate in the column span.

Usage:
    python analysis/analyze_leverage.py \
        phase1_ckpt_v8 \
        phase2_ckpt_v8_s2_aurora \
        phase2_ckpt_v8_s2_muon5e5

Output:
- Per-checkpoint summary of leverage distribution stats across all MLP
  layers (mean / std / coefficient of variation / min/max / fraction in
  bottom 10%).
- Comparison table at the end.

CPU-bound; safe to run alongside GPU experiments.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from safetensors.torch import safe_open


def find_mlp_weights(ckpt_dir):
    """Find safetensors file(s) and return dict of {name: tensor}
    for parameters matching MLP up_proj / gate_proj / down_proj.
    """
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
                    weights[k] = st.get_tensor(k)
    return weights


def leverage_scores(W):
    """Compute leverage scores for rows of W (after orienting so m >= n).

    Returns (leverage_array, target_uniform) where leverage_array.shape
    is (m,) and target_uniform = n/m (uniform-leverage value).
    """
    W = W.float().cpu()
    m, n = W.shape
    if m < n:
        W = W.T
        m, n = n, m
    # full SVD on R^{m,n}, m >= n. U has shape (m, n).
    # Using torch.linalg.svd in 'reduced' mode -> U (m,n), S (n,), Vh (n,n).
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    # Drop zero singular values (numerical zeros)
    eps = S.max() * 1e-7
    valid = S > eps
    U = U[:, valid]
    # Leverage of row i = ||U[i,:]||²
    lev = (U * U).sum(dim=-1).numpy()
    target = float(valid.sum()) / m
    return lev, target


def summarize(lev, target):
    """Return dict of distribution stats."""
    cv = lev.std() / lev.mean() if lev.mean() > 0 else 0.0
    sorted_lev = np.sort(lev)
    n = len(sorted_lev)
    p10 = sorted_lev[: max(1, n // 10)].mean()
    p90 = sorted_lev[-max(1, n // 10):].mean()
    dead_frac = (lev < 0.1 * target).mean()
    return dict(
        n_rows=n, target=target, mean=lev.mean(), std=lev.std(),
        cv=cv, min=lev.min(), max=lev.max(),
        p10=p10, p90=p90, p10_over_p90=p10 / p90 if p90 > 0 else 0.0,
        dead_frac=dead_frac,
    )


def analyze_ckpt(name, ckpt_dir):
    print(f"\n=== {name} ({ckpt_dir}) ===")
    weights = find_mlp_weights(ckpt_dir)
    print(f"  {len(weights)} MLP weight tensors")

    all_lev = []
    by_kind = {"up": [], "gate": [], "down": []}
    target_by_kind = {}

    for k, W in sorted(weights.items()):
        kind = "up" if "up_proj" in k else "gate" if "gate_proj" in k else "down"
        lev, target = leverage_scores(W)
        all_lev.append(lev)
        by_kind[kind].append(lev)
        target_by_kind[kind] = target

    # aggregate stats per kind
    agg = {}
    for kind, levs in by_kind.items():
        if not levs:
            continue
        cat = np.concatenate(levs)
        s = summarize(cat, target_by_kind[kind])
        agg[kind] = s
        print(f"  {kind:5s} | target={s['target']:.4f}  mean={s['mean']:.4f}  "
              f"std={s['std']:.4f}  CV={s['cv']:.3f}  "
              f"p10/p90={s['p10_over_p90']:.3f}  "
              f"dead<10%={100*s['dead_frac']:.1f}%  "
              f"max/target={s['max']/s['target']:.2f}")

    # all-MLP aggregate
    cat_all = np.concatenate(all_lev)
    # use mean target across kinds (all have same since shapes match)
    avg_target = np.mean(list(target_by_kind.values()))
    s_all = summarize(cat_all, avg_target)
    print(f"  ALL   | target={s_all['target']:.4f}  mean={s_all['mean']:.4f}  "
          f"std={s_all['std']:.4f}  CV={s_all['cv']:.3f}  "
          f"p10/p90={s_all['p10_over_p90']:.3f}  "
          f"dead<10%={100*s_all['dead_frac']:.1f}%")
    return agg, s_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", help="Ckpt names under output/")
    args = p.parse_args()

    output_root = "/home/tfbao/Shiyu/Summer/output"
    results = {}
    for name in args.ckpts:
        ckpt_dir = os.path.join(output_root, name)
        if not os.path.isdir(ckpt_dir):
            print(f"SKIP {name}: dir missing", file=sys.stderr)
            continue
        results[name] = analyze_ckpt(name, ckpt_dir)

    # final comparison
    if len(results) >= 2:
        print("\n=== Comparison (all MLP) ===")
        print(f"{'ckpt':<35} {'CV':>8} {'p10/p90':>9} {'dead<10%':>10}")
        for name, (_, s) in results.items():
            print(f"{name:<35} {s['cv']:>8.3f} {s['p10_over_p90']:>9.3f} "
                  f"{100*s['dead_frac']:>9.1f}%")
        print("\nLower CV / higher p10/p90 / lower dead% = more uniform leverage = Aurora-style")


if __name__ == "__main__":
    main()
