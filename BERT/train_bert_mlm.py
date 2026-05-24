"""BERT MLM pretrain — 简化版,encoder-only,无 NSP。

复用 v18 数据(input_ids tensor [N, seqlen]),在线做 MLM masking(15%)。
"""
import argparse, os, json, time, math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup


class IntTensorDataset(Dataset):
    def __init__(self, pt_path, max_chunks=None):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        if max_chunks is not None and data.shape[0] > max_chunks:
            data = data[:max_chunks]
        self.data = data.to(torch.long)  # int32 → long for embedding
        print(f"Loaded {data.shape[0]:,} chunks × {data.shape[1]} from {pt_path}")
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, i): return self.data[i]


def mlm_mask_batch(input_ids, mask_token_id, vocab_size, prob=0.15, pad_id=81899):
    """In-place MLM masking — 15% selected,其中 80% → mask,10% → random,10% unchanged。"""
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    # 不 mask 已经是 pad 的位置(数据里其实不会有)
    not_pad = input_ids != pad_id
    probability_matrix = torch.full(labels.shape, prob)
    masked_indices = torch.bernoulli(probability_matrix).bool() & not_pad
    labels[~masked_indices] = -100  # 只对 masked 位置算 loss

    # 80% → [MASK]
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[replace_mask] = mask_token_id
    # 10% → random token
    replace_rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_mask
    random_words = torch.randint(0, vocab_size, labels.shape, dtype=input_ids.dtype)
    input_ids[replace_rand] = random_words[replace_rand]
    # 剩 10% unchanged
    return input_ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--max_chunks", type=int, default=None,
                   help="只用前 N chunks(等价限制 token 量)")
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # 读 mask_token_id
    with open(os.path.join(args.model_path, "mask_token_id.txt")) as f:
        mask_token_id = int(f.read().strip())
    print(f"mask_token_id = {mask_token_id}")

    # 加载模型
    model = BertForMaskedLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    vocab_size = model.config.vocab_size
    pad_id = model.config.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, vocab={vocab_size}, max_pos={model.config.max_position_embeddings}")

    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled")

    # 数据
    ds = IntTensorDataset(args.train_data, args.max_chunks)
    print(f"Dataset: {len(ds):,} chunks × {ds.data.shape[1]} = {ds.data.numel():,} tokens")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader) // args.gradient_accumulation_steps
    print(f"steps/epoch: {steps_per_epoch}, max_steps: {args.max_steps}")

    # 优化器 + scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=0.5,
    )
    # 手动设 min_lr_ratio 兼容
    min_lr = args.lr * args.min_lr_ratio

    step = 0
    accum = 0
    t0 = time.time()
    loss_acc = 0.0
    correct_acc = 0
    n_masked_acc = 0
    model.train()
    while step < args.max_steps:
        for input_ids in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            masked_ids, labels = mlm_mask_batch(input_ids, mask_token_id, vocab_size,
                                                prob=args.mlm_prob, pad_id=pad_id)
            out = model(input_ids=masked_ids, labels=labels)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            # accuracy on masked
            with torch.no_grad():
                preds = out.logits.argmax(-1)
                mask_pos = labels != -100
                correct_acc += ((preds == labels) & mask_pos).sum().item()
                n_masked_acc += mask_pos.sum().item()
            loss_acc += loss.item() * args.gradient_accumulation_steps
            accum += 1
            if accum >= args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                scheduler.step()
                # clip min_lr
                for g in optim.param_groups:
                    if g["lr"] < min_lr: g["lr"] = min_lr
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                if step % args.logging_steps == 0:
                    el = time.time() - t0
                    cur_lr = optim.param_groups[0]["lr"]
                    avg_loss = loss_acc / args.logging_steps
                    acc = correct_acc / max(1, n_masked_acc)
                    print(f"step {step}/{args.max_steps} | loss {avg_loss:.4f} | "
                          f"mlm_acc {acc:.4f} | lr [{cur_lr:.6f}] | {el:.1f}s", flush=True)
                    loss_acc = 0.0
                    correct_acc = 0
                    n_masked_acc = 0
                if step % args.save_steps == 0 or step >= args.max_steps:
                    save = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(save, exist_ok=True)
                    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                    raw.save_pretrained(save, safe_serialization=True)
                    # 拷 mask_token_id 给 inference 用
                    import shutil
                    shutil.copy2(os.path.join(args.model_path, "mask_token_id.txt"),
                                 os.path.join(save, "mask_token_id.txt"))
                    print(f"Saved checkpoint to {save}")
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    # 最终 save
    final = args.output_dir
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.save_pretrained(final, safe_serialization=True)
    print(f"\nFinal save to {final}")
    print(f"Training complete: {step} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
