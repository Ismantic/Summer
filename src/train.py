"""
ReTok 两阶段训练的入口。**只依赖 torch。**

  Phase 1  --freeze_transformer   只训 embed_tokens + lm_head
  Phase 2  --use_lora             transformer 走低秩旁路,embed 全参训

模型、LoRA、优化器、safetensors 读写都是自己实现的(src/ 下),不依赖
transformers 和 peft。分词器在 prepare/ —— src/ 不碰文本,只读预编码好的 id。

  --mode clm   预编码 .pt(v18 走的路径)
  --mode sft   JSONL,**目前不支持** —— 模型没有 attention_mask,padding 会算错

v18 SOTA 的配方见 prepare/Makefile 的 p1 / p2 两个 target。
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import json
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import Qwen3ForCausalLM
from src.optim import SingleDeviceMuonWithAuxAdam
from src.optim import aurora_update

IGNORE_INDEX = -100


def _copy_tokenizer_artifacts(src_dir, dst_dir):
    """把分词器产物拷到 checkpoint 旁边,让保存下来的目录能自足加载。

    只是拷文件,不加载也不使用分词器 —— 不违反「src/ 不碰文本」。
    """
    import shutil
    # 上游名 + 旧名都列上:从哪个 checkpoint 继续训就带哪一套。
    # 上游名(Summer-Tokenizer.*)与 PieceTokenizer 仓库 save/ 下同名;
    # piece.model / dict.txt 是改造前的名字,v18 的 ckpt 和已发布模型在用。
    artifacts = ["Summer-Tokenizer.pt", "Summer-Tokenizer.dict.txt",
                 "piece.model", "dict.txt", "token_mapping.json",
                 "tokenizer_config.json", "special_tokens_map.json"]
    for name in artifacts:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


class PreTokenizedDataset(Dataset):
    """Pre-tokenized dataset from a .pt file (shape: [N, seq_len], dtype: int32)."""
    def __init__(self, pt_file):
        self.data = torch.load(pt_file, weights_only=True).long()
        print(f"[PreTok] Loaded {self.data.shape[0]} chunks from {pt_file}, seq_len={self.data.shape[1]}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        tokens = self.data[index]
        return dict(input_ids=tokens, labels=tokens.clone(), attention_mask=torch.ones_like(tokens, dtype=torch.bool))


def collate_fn(batch, pad_token_id):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def build_optimizer(model, muon_lr, adam_lr, muon_momentum, weight_decay, use_aurora=False, moonshot_scaling=False):
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_count = sum(p.numel() for p in muon_params)
    adam_count = sum(p.numel() for p in adam_params)
    rule = "Aurora" if use_aurora else "Muon"
    if moonshot_scaling:
        rule = f"{rule}+MoonshotLR"
    print(f"{rule} params: {muon_count:,} | Adam params: {adam_count:,}  (wd={weight_decay})")

    if muon_params:
        param_groups = [
            dict(params=muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay, use_muon=True),
            dict(params=adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay, use_muon=False),
        ]
        update_fn = aurora_update if use_aurora else None
        return SingleDeviceMuonWithAuxAdam(param_groups, update_fn=update_fn, moonshot_scaling=moonshot_scaling)
    else:
        # No Muon params (e.g. freeze_transformer), use plain Adam
        return torch.optim.AdamW(adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay)


def train(args):
    # DDP setup (no-op when launched as single process)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = (local_rank == 0)
    import torch.distributed as dist
    if is_distributed:
        # Long NCCL timeout — inline eval can take ~60min, during which
        # rank-1 sits at dist.barrier(). Default 10min triggers SIGABRT.
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def log(*a, **kw):
        if is_main:
            print(*a, **kw)

    # pad id 从 token_mapping.json 直接读 —— src/ 不加载分词器(那是 prepare/ 的事)
    _map_file = os.path.join(args.model_path, "token_mapping.json")
    if os.path.exists(_map_file):
        with open(_map_file) as f:
            pad_token_id = json.load(f)["pad_id"]
    else:
        raise FileNotFoundError(
            f"{args.model_path} 里没有 token_mapping.json,取不到 pad_id。")
    # 自写的模型只做 is_causal 的因果注意力,**不接受 attention_mask**。
    # clm 模式喂的是预编码 .pt,每条都是满长度、没有 padding —— 这是 v18 走的
    # 路径,忽略 mask 是对的。sft 模式会 pad,忽略 mask 会让 padding 位参与注意力,
    # 算出来的东西是错的而且不报错,所以这里直接拦掉。
    if args.mode == "sft":
        raise NotImplementedError(
            "src/model.py 目前不支持 attention_mask,而 sft 模式会 padding。\n"
            "  要么给 Attention 加上 mask 支持,要么用 --mode clm。\n"
            "  v18 的复现路径是 clm + 预编码 .pt,不受影响。")

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path, device=device, dtype=torch.bfloat16)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze transformer if requested
    if args.freeze_transformer:
        for name, param in model.named_parameters():
            if "embed" not in name and "lm_head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"Frozen transformer: training {trainable:,} / {total:,} params "
            f"({trainable/total*100:.1f}%)")

    # Freeze specific embedding rows (one-to-one mapped tokens)
    if args.freeze_mapped_embeds:
        with open(args.freeze_mapped_embeds) as f:
            frozen_ids = json.load(f)
        frozen_mask = torch.zeros(model.config.vocab_size, 1, device=device, dtype=torch.bfloat16)
        frozen_mask[frozen_ids] = 1.0
        trainable_count = model.config.vocab_size - len(frozen_ids)
        log(f"Freezing {len(frozen_ids)} / {model.config.vocab_size} embedding rows, "
            f"training {trainable_count} rows ({trainable_count/model.config.vocab_size*100:.1f}%)")

        def _zero_frozen_grads(grad):
            return grad * (1.0 - frozen_mask)
        model.model.embed_tokens.weight.register_hook(_zero_frozen_grads)

    # LoRA —— Phase 2 的高效微调:transformer 走低秩旁路,embed_tokens 全参训。
    #
    # 自写实现(src/lora.py)替掉了 peft。v18 只用到 peft 的一个很小的子集,
    # 而 peft 的 ModulesToSave 会 deepcopy embed/lm_head —— 那正是 tie 被破坏
    # 的根源,也是当初要单开 --lora_tie_embed_head 的原因。自己实现之后不存在
    # 这个问题:根本不去动 embed 模块,只是把它解冻。
    if args.use_lora:
        from src.lora import apply_lora, count_lora_params

        if args.freeze_transformer:
            log("WARNING: --use_lora overrides --freeze_transformer")
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                              targets=tuple(args.lora_target.split(",")),
                              dropout=0.0)
        # embed_tokens 全参训;lm_head 与它共享 tensor,自动跟着更新
        model.model.embed_tokens.weight.requires_grad_(True)

        emb_w = model.model.embed_tokens.weight
        head_w = model.lm_head.weight
        if emb_w.data_ptr() == head_w.data_ptr():
            log("[tie] embed/lm_head 共享存储 ✓")
        elif args.lora_tie_embed_head:
            log("[tie] WARN: embed/lm_head 不再共享存储!")

        n_lora = count_lora_params(model)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"LoRA: {len(replaced)} 个模块 | LoRA 参数 {n_lora:,} | "
            f"可训练 {n_train:,} / {n_all:,} ({n_train / n_all * 100:.2f}%)")

    # Wrap with DDP after freeze logic + grad-hook registration (so the hook
    # binds to the underlying parameter, not the DDP-replicated one).
    raw_model = model
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True because freeze_transformer leaves
        # transformer params with requires_grad=False; DDP's autograd-graph
        # check needs this hint when not all params get grads each step.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=args.freeze_transformer)

    # torch.compile — kernel fusion, ~20-40% speedup. raw_model stays uncompiled
    # so checkpoint save / inline-eval CPU-offload operate on the original module.
    if args.compile:
        model = torch.compile(model)
        log("torch.compile enabled")

    # Dataset —— 只吃预编码好的 id。原始文本的分词在 prepare/encode_corpus.py。
    if not args.train_data.endswith(".pt"):
        raise ValueError(
            f"--train_data 必须是预编码的 .pt,收到 {args.train_data}\n"
            f"  先跑 `make -C prepare encode`(或 prepare/encode_corpus.py)。\n"
            f"  src/ 不碰文本 —— 分词属于 prepare/ 那一层。")
    dataset = PreTokenizedDataset(args.train_data)

    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    sampler = None
    if is_distributed and not is_iterable:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
    elif is_distributed and is_iterable:
        raise NotImplementedError("DDP + IterableDataset not supported; pretokenize first")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None and not is_iterable),
        sampler=sampler, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    # Optimizer (build on raw model so param names don't have DDP "module." prefix)
    optimizer = build_optimizer(raw_model, args.muon_lr, args.adam_lr, args.muon_momentum, args.weight_decay,
                                use_aurora=args.use_aurora, moonshot_scaling=args.moonshot_scaling)

    # LR scheduler: linear warmup, then decay to min_lr_ratio*peak floor.
    # Default schedule is cosine (Llama/Qwen standard); linear available for back-compat.
    # min_lr_ratio > 0 (e.g. 0.1) avoids the "decay-to-zero wastes last steps" pathology.
    import math
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, progress)
        if args.lr_schedule == "cosine":
            # Cosine from 1.0 to min_lr_ratio
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * factor
        else:
            return max(args.min_lr_ratio, 1.0 - progress * (1.0 - args.min_lr_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    # step counts optimizer updates, not forward passes
    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()
    interrupted = False

    def _move_optimizer_state(target):
        """Move all optimizer state tensors to target device."""
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(target, non_blocking=False)

    def _inline_eval(current_step):
        """Offload model+optimizer to CPU, run eval subprocess, move back."""
        if not args.inline_eval_cmd:
            return
        # All ranks offload
        raw_model.cpu()
        _move_optimizer_state("cpu")
        torch.cuda.empty_cache()
        if is_distributed:
            dist.barrier()
        # Only main rank runs eval command
        if is_main:
            cmd = args.inline_eval_cmd.replace("{step}", str(current_step))
            log(f"[inline_eval] running: {cmd}")
            t_ev = time.time()
            os.system(cmd)
            log(f"[inline_eval] done in {time.time()-t_ev:.0f}s")
        if is_distributed:
            dist.barrier()
        # All ranks move back
        raw_model.to(device)
        _move_optimizer_state(device)
        torch.cuda.empty_cache()

    import signal
    def _sigint_handler(sig, frame):
        nonlocal interrupted
        if interrupted:  # second Ctrl+C = force quit
            raise KeyboardInterrupt
        interrupted = True
        print(f"\nCtrl+C received at step {step}, saving checkpoint...")
    signal.signal(signal.SIGINT, _sigint_handler)

    epoch = 0
    while step < args.max_steps and not interrupted:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in dataloader:
            if step >= args.max_steps or interrupted:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 自写的模型只返回 logits,loss 在这里算(HF 是在模型内部算的)。
                # 口径与 Qwen3ForCausalLM 一致:左移一位的 next-token CE,
                # IGNORE_INDEX(-100) 处不计入。
                logits = model(batch["input_ids"])
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                    batch["labels"][:, 1:].reshape(-1),
                    ignore_index=IGNORE_INDEX)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.logging_steps == 0:
                    elapsed = time.time() - t0
                    lrs = [f"{g['lr']:.6f}" for g in optimizer.param_groups]
                    real_loss = loss.item() * args.gradient_accumulation_steps
                    log(f"step {step}/{args.max_steps} | loss {real_loss:.4f} | "
                        f"lr [{', '.join(lrs)}] | {elapsed:.1f}s")

                if args.save_steps > 0 and step % args.save_steps == 0:
                    if is_main:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                        os.makedirs(save_path, exist_ok=True)
                        # LoRA 训练时中途只存 adapter(几 MB),不存整个基座
                        if args.use_lora:
                            from src.checkpoint import save_safetensors
                            from src.lora import adapter_state_dict
                            save_safetensors(adapter_state_dict(raw_model),
                                             os.path.join(save_path,
                                                          "adapter_model.safetensors"),
                                             metadata={"format": "pt"})
                        else:
                            raw_model.save_pretrained(save_path)
                        _copy_tokenizer_artifacts(args.model_path, save_path)
                        print(f"Saved checkpoint to {save_path}")
                    if is_distributed:
                        dist.barrier()
                    # Inline eval (CPU-offload pattern) if requested
                    _inline_eval(step)

    # Save final model (only main rank).
    # LoRA: 合并回基座,存成标准 HF 布局,评测直接加载。
    if args.use_lora and args.output_dir and is_main:
        from src.lora import merge_lora
        n = merge_lora(raw_model)
        log(f"合并 {n} 个 LoRA 模块回基座,存成标准 HF 布局")

    if args.output_dir and is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        raw_model.save_pretrained(args.output_dir)
        _copy_tokenizer_artifacts(args.model_path, args.output_dir)
        print(f"Saved final model to {args.output_dir}")

    elapsed = time.time() - t0
    log(f"Training complete: {step} steps in {elapsed:.1f}s ({step/elapsed:.2f} steps/s)")
    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HY-MT fine-tuning with Muon optimizer")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True, help="JSONL for sft mode, or comma-separated text files for clm mode")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "clm"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model in torch.compile for kernel fusion (~20-40% speedup)")
    # LoRA —— Phase 2 在单卡上对 transformer 做高效微调(src/lora.py,非 peft)
    parser.add_argument("--use_lora", action="store_true",
                        help="transformer 走 LoRA 旁路;embed_tokens 全参训(lm_head 因 tie 自动同步)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_tie_embed_head", action="store_true",
                        help="保持 embed_tokens 和 lm_head 共享 weight (避免 PEFT 副作用 untie)")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj",
                        help="Comma-separated target_modules for LoRA")
    parser.add_argument("--freeze_transformer", action="store_true", help="Only train embed_tokens + lm_head")
    parser.add_argument("--freeze_mapped_embeds", type=str, default=None,
                        help="Path to JSON list of token IDs to freeze (one-to-one mapped tokens)")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--muon_lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--use_aurora", action="store_true",
                        help="Use Aurora update rule (leverage-uniform polar) instead of standard Muon")
    parser.add_argument("--moonshot_scaling", action="store_true",
                        help="Apply Moonshot's per-param LR scaling (lr *= 0.2 * sqrt(max(A,B))) to Muon/Aurora groups")
    parser.add_argument("--inline_eval_cmd", type=str, default="",
                        help="Shell command run on rank-0 at every save_steps. Model+optimizer "
                             "are moved to CPU (and CUDA cache cleared) before the command runs, "
                             "and moved back to GPU after it returns. The literal '{step}' in the "
                             "command is replaced with the current step number.")
    parser.add_argument("--resume_from", type=str, default="",
                        help="(unused placeholder for future resume support — current rolling "
                             "eval flow uses --inline_eval_cmd instead)")
    parser.add_argument("--lr_total_steps", type=int, default=0,
                        help="(unused placeholder; --max_steps drives cosine in current flow)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Floor for the LR schedule, as a fraction of peak LR. Default 0.1 matches "
                             "Llama/Qwen practice (peak/10). Set to 0 to recover old decay-to-zero behavior.")
    parser.add_argument("--lr_schedule", choices=["cosine", "linear"], default="cosine",
                        help="LR decay shape after warmup. Default cosine (Llama/Qwen standard). "
                             "Set 'linear' for older behavior.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)
