"""
Compute perplexity on a pretokenized .pt validation set.

The valid file is shape [N, seq_len] int32 — same format as
pretokenize.py output. We compute mean cross-entropy over the *next-token*
positions (so on a seq_len=512 chunk, 511 prediction targets per row),
then PPL = exp(mean_loss).

Usage:
    python src/evaluate.py \
        --model_path output/phase2_ckpt_v18_tie \
        --valid_pt test/fixtures/ppl_slice.pt \
        --batch_size 16 \
        --output_path eval_results/ppl/phase1_v1.json
"""
import argparse
import json
import math
import os
import time

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import Qwen3ForCausalLM                          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--valid_pt", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Optional cap on number of batches (for quick smoke)")
    p.add_argument("--output_path", default=None)
    args = p.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path, device="cuda", dtype=torch.bfloat16)

    print(f"Loading valid set from {args.valid_pt}...")
    data = torch.load(args.valid_pt, weights_only=True).long()
    n_chunks, seq_len = data.shape
    print(f"  {n_chunks:,} chunks x {seq_len} = {n_chunks*seq_len:,} tokens")

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    n_batches = (n_chunks + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for bi in range(n_batches):
            if args.max_batches and bi >= args.max_batches:
                break
            start = bi * args.batch_size
            end = min(start + args.batch_size, n_chunks)
            batch = data[start:end].cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)  # [B, T, V]
            # next-token CE: predict token at position t+1 from logits at t
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:].contiguous()
            # token-level mean CE, then * num tokens for accumulation
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            n_tok = shift_labels.numel()
            total_loss += loss_per_token.item()
            total_tokens += n_tok
            if (bi + 1) % 20 == 0:
                avg = total_loss / total_tokens
                elapsed = time.time() - t0
                print(f"  batch {bi+1}/{n_batches} | "
                      f"avg loss {avg:.4f} (PPL {math.exp(avg):.2f}) | "
                      f"{elapsed:.0f}s")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    elapsed = time.time() - t0
    print(f"\nFinal: loss {avg_loss:.4f} | PPL {ppl:.2f} "
          f"| {total_tokens:,} tokens | {elapsed:.0f}s")

    out = {
        "model_path": args.model_path,
        "valid_pt": args.valid_pt,
        "n_tokens": total_tokens,
        "loss": avg_loss,
        "ppl": ppl,
        "time_s": elapsed,
    }
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
