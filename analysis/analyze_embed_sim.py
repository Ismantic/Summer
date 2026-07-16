"""
Compare top-K cosine-similarity neighbors of selected token embeddings across
multiple model checkpoints, to see whether phase 1 training is moving the
multi-mapped/fallback rows towards semantically sensible neighborhoods.

Usage:
    python analysis/analyze_embed_sim.py \
        --tokenizer_dir /home/tfbao/new/Qwen3-0.6B-Base-new-tok \
        --frozen_ids /home/tfbao/Shiyu/Summer/output/frozen_ids.json \
        --models cold:/home/tfbao/new/Qwen3-0.6B-Base-new-tok \
                 v0:/home/tfbao/Shiyu/Summer/output/phase1_ckpt_v0 \
                 v1:/home/tfbao/Shiyu/Summer/output/phase1_ckpt_v1 \
                 v2:/home/tfbao/Shiyu/Summer/output/phase1_ckpt_v2 \
        --top_k 6 \
        --n_multi 8 --n_one2one 3
"""
import argparse
import json
import os
import random

import torch
from safetensors.torch import load_file


def load_embed(model_path: str) -> torch.Tensor:
    """Return embed_tokens weight as fp32 [V, D]."""
    sf = os.path.join(model_path, "model.safetensors")
    state = load_file(sf)
    for k, v in state.items():
        if "embed_tokens" in k:
            return v.float()
    raise KeyError(f"embed_tokens not found in {sf}; keys={list(state.keys())[:10]}")


def topk_neighbors(emb: torch.Tensor, query_ids: list[int], k: int):
    """Returns list[list[(neighbor_id, sim)]] for each query_id."""
    norm = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_n = emb / norm
    q = emb_n[torch.tensor(query_ids)]
    sims = q @ emb_n.T  # [Q, V]
    for i, qid in enumerate(query_ids):
        sims[i, qid] = -1.0  # exclude self
    vals, idx = sims.topk(k, dim=1)
    out = []
    for r in range(len(query_ids)):
        out.append(list(zip(idx[r].tolist(), vals[r].tolist())))
    return out


def piece_repr(piece: str) -> str:
    """Pretty-print piece, show ▁ as visible space prefix."""
    return piece.replace("▁", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--frozen_ids", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="tag:path tag:path ...")
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--n_multi", type=int, default=8,
                   help="How many multi-mapped tokens to sample as queries")
    p.add_argument("--n_one2one", type=int, default=3,
                   help="How many 1-to-1 mapped tokens as control")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load piece tokenizer for id <-> text
    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(os.path.join(args.tokenizer_dir, "piece.model"),
             cn_dict=os.path.join(args.tokenizer_dir, "dict.txt"))
    vocab_size = tok.vocab_size()

    frozen = set(json.load(open(args.frozen_ids)))
    multi_ids = [i for i in range(vocab_size) if i not in frozen and i > 100]
    one2one = [i for i in frozen if i > 100]
    print(f"vocab={vocab_size}, frozen(1-to-1)={len(frozen)}, multi+fallback={len(multi_ids)}")

    rng = random.Random(args.seed)
    # Pick "interesting" multi-mapped: avoid super-rare ids; sample uniformly
    query_multi = rng.sample(multi_ids, args.n_multi)
    query_one2one = rng.sample(one2one, args.n_one2one)
    queries = [(i, "multi") for i in query_multi] + [(i, "one2one") for i in query_one2one]

    # Load all model embeds
    embs = {}
    for spec in args.models:
        tag, path = spec.split(":", 1)
        print(f"Loading {tag}: {path}")
        embs[tag] = load_embed(path)
    tags = [s.split(":", 1)[0] for s in args.models]

    # For each query token, show neighbors per model side-by-side
    for qid, kind in queries:
        try:
            qtext = piece_repr(tok.id_to_piece(qid))
        except Exception:
            qtext = "?"
        print(f"\n[{kind:8s}] id={qid:>5d}  piece={qtext!r}")
        for tag in tags:
            nbrs = topk_neighbors(embs[tag], [qid], args.top_k)[0]
            parts = []
            for nid, sim in nbrs:
                try:
                    ntext = piece_repr(tok.id_to_piece(nid))
                except Exception:
                    ntext = "?"
                parts.append(f"{ntext!r}({sim:.2f})")
            print(f"  {tag:>4s}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
