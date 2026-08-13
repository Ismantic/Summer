"""
训练入口。**只依赖 torch。** 两条路线共用它:

  ReTok(继续预训练,起点是 Qwen3-1.7B-Base 换完词表的模型)
    Phase 1  --freeze_transformer   只训 embed_tokens + lm_head
    Phase 2  --use_lora             transformer 走低秩旁路,embed 全参训

  Summer-0.5B(从零预训练,起点是随机初始化的 0.6B 架构)
    全参数训练,--lr_schedule wsd + --param_dtype float32 + --ce_chunk 4096

两条路线对精度和调度的要求相反,别把默认值弄混:**默认值仍然是 ReTok 那套**
(bf16 参数 + cosine),从零训练要显式加上面三个开关。理由写在各自的
add_argument 帮助里。

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
                 "tokenizer_config.json", "special_tokens_map.json",
                 # 漏了它 save/export.py 会拒绝导出;更糟的是没被拦住的时候,
                 # 生成时 eos/pad 走 HF 默认值,和这套 81903 词表对不上。
                 "generation_config.json"]
    for name in artifacts:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


class PreTokenizedDataset(Dataset):
    """预编码 .pt(形状 [N, seq_len],dtype int32)。**内存映射,不整份读进内存。**

    原来是 `torch.load(...).long()` —— 一次性读进内存,还顺手转成 int64 把体积
    翻倍。ReTok 的 1B token 是 4GB → 8GB,本机 61GB 撑得住;从零预训练要
    12B token(48GB → 96GB),**在训练开始前就 OOM,一步都没跑、什么都没存**。

    改成 mmap:页由内核按需调入,常驻的只有实际访问过的部分,int32→int64 的
    转换挪到逐条取数据的时候。多个 shard 用逗号分隔,当一份连续数据用 ——
    12B token 分 12 份,既是为了编码时能并行、断点续,也是为了单个文件别太大。
    """
    def __init__(self, pt_files, mask_files=None):
        paths = [p.strip() for p in str(pt_files).split(",") if p.strip()]
        # **掩码也 mmap,一一对应。** midtrain/SFT 只在助手回复上算 loss;
        # 没有掩码就是预训练那样整段算(mask 恒为 1)。
        mpaths = [p.strip() for p in str(mask_files or "").split(",") if p.strip()]
        if mpaths and len(mpaths) != len(paths):
            raise ValueError(
                f"掩码文件数({len(mpaths)})和数据文件数({len(paths)})不一致 —— "
                f"它们必须一一对应,错位会让 loss 算在错误的位置上而不报错。")
        self.masks = [torch.load(m, weights_only=True, mmap=True) for m in mpaths]
        self.shards, self.starts, total, seq_len = [], [], 0, None
        for p in paths:
            t = torch.load(p, weights_only=True, mmap=True)
            if seq_len is None:
                seq_len = t.shape[1]
            elif t.shape[1] != seq_len:
                raise ValueError(
                    f"shard 的 seq_len 不一致:{p} 是 {t.shape[1]},前面是 {seq_len}。"
                    f"混着用会让 position_ids 和 batch 形状对不上。")
            self.shards.append(t)
            self.starts.append(total)
            total += t.shape[0]
        self.total, self.seq_len = total, seq_len
        print(f"[PreTok] mmap {len(paths)} 个 shard | {total:,} chunks × {seq_len} "
              f"= {total * seq_len:,} token", flush=True)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        # 从后往前找所属 shard —— shard 数是十几个,线性扫比二分更简单也够快
        shard_i = 0
        for k, (start, shard) in enumerate(zip(reversed(self.starts),
                                               reversed(self.shards))):
            if index >= start:
                tokens = shard[index - start].long()
                shard_i = len(self.shards) - 1 - k
                local = index - start
                break
        labels = tokens.clone()
        if self.masks:
            # mask=0 的位置置成 IGNORE_INDEX —— 用户那段不算 loss。
            keep = self.masks[shard_i][local].bool()
            labels[~keep] = IGNORE_INDEX
        return dict(input_ids=tokens, labels=labels,
                    attention_mask=torch.ones_like(tokens, dtype=torch.bool))


def chunked_cross_entropy(logits, targets, chunk=0):
    """按行分块算 next-token CE。**显存瓶颈在这里,不在模型。**

    词表 81903 的时候 `logits.float()` 本身就是大头:batch 16 × seq 1023 时
    这一份 float 副本要 5.4GB,cross_entropy 内部的 log_softmax 再要一份。
    实测(0.5B / seq 1024 / 4090 24GB):

        整份算   batch 8 峰值 11.3GiB,batch 16 直接 OOM
        分块算   batch 8 峰值 10.1GiB,batch 16 能跑(16.8GiB),吞吐还 +8%

    数值口径与 `F.cross_entropy(logits.float(), targets)` 一致:每块用
    reduction="sum" 累加,最后除以有效 token 数。**除数是有效数不是总数** ——
    用总数会把 IGNORE_INDEX 的位置也算进分母,loss 系统性偏低而不报错。

    chunk<=0 时退回整份计算(短序列/小词表下没必要分块)。
    """
    n_valid = (targets != IGNORE_INDEX).sum()
    if chunk <= 0:
        return torch.nn.functional.cross_entropy(
            logits.float(), targets, ignore_index=IGNORE_INDEX)
    total = logits.new_zeros((), dtype=torch.float32)
    for i in range(0, logits.size(0), chunk):
        total = total + torch.nn.functional.cross_entropy(
            logits[i:i + chunk].float(), targets[i:i + chunk],
            ignore_index=IGNORE_INDEX, reduction="sum")
    return total / n_valid.clamp(min=1)


def collate_fn(batch, pad_token_id):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def build_optimizer(model, muon_lr, adam_lr, muon_momentum, weight_decay, use_aurora=False, moonshot_scaling=False,
                    adam_betas=(0.9, 0.95), adam_weight_decay=None, dmodel_lr_scale=False):
    """**后三个参数默认保持原行为。**

    它们是为了对齐 nanochat 加的(见 `docs/POSTTRAIN.md` 的 A 组),但
    `src/train.py` 同时在跑预训练和 ReTok 两条线,而那两条线**已经发布了权重**。
    直接改硬编码等于悄悄换掉它们的配方,**而且不会报错** —— 所以做成开关,
    由 Makefile 的 midtrain / sft-chat 两个 target 显式打开。

    - `adam_betas`:nanochat 用 (0.8, 0.95),我们原来是 (0.9, 0.95)
    - `adam_weight_decay`:nanochat 的 AdamW 组 wd 硬编码 0,wd 只给 Muon。
      None = 沿用 `weight_decay`(原行为)
    - `dmodel_lr_scale`:nanochat 把 AdamW 的 lr 乘 (d_model/768)^-0.5。
      d_model=1024 时是 0.866
    """
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
    adam_wd = weight_decay if adam_weight_decay is None else adam_weight_decay
    if dmodel_lr_scale:
        scale = (model.config.hidden_size / 768) ** -0.5
        adam_lr = adam_lr * scale
        print(f"AdamW lr × (d_model/768)^-0.5 = ×{scale:.4f} → {adam_lr:.3e}", flush=True)
    print(f"{rule} params: {muon_count:,} (wd={weight_decay}) | "
          f"Adam params: {adam_count:,} (wd={adam_wd}, betas={adam_betas})", flush=True)

    if muon_params:
        param_groups = [
            dict(params=muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay, use_muon=True),
            dict(params=adam_params, lr=adam_lr, betas=adam_betas, eps=1e-10, weight_decay=adam_wd, use_muon=False),
        ]
        update_fn = aurora_update if use_aurora else None
        return SingleDeviceMuonWithAuxAdam(param_groups, update_fn=update_fn, moonshot_scaling=moonshot_scaling)
    else:
        # No Muon params (e.g. freeze_transformer), use plain Adam
        return torch.optim.AdamW(adam_params, lr=adam_lr, betas=adam_betas, eps=1e-10, weight_decay=adam_wd)


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
        # **必须 flush。** 重定向到文件时 Python 的 stdout 是块缓冲(8KB),
        # 不是行缓冲 —— p1 要跑 28 小时,而 logging_steps=50 一条日志才 ~90 字节,
        # 攒满 8KB 要几千步。表现是「跑了半小时日志还是空的」,看着像卡死,
        # 实际 GPU 一直满载。2026-07-28 真踩了一次,白等 40 分钟。
        if is_main:
            kw.setdefault("flush", True)
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

    # 续训:权重从 checkpoint 目录加载,超参仍然从命令行来。
    # **不从 train_state 里恢复超参** —— WSD 的用法就是拿新的 max_steps 续跑,
    # 恢复旧超参会让「改主意」这件事失效。恢复的只有轨迹状态。
    resume_state = None
    load_dir = args.model_path
    if args.resume_from:
        load_dir = args.resume_from
        state_file = os.path.join(args.resume_from, "train_state.pt")
        if not os.path.exists(state_file):
            raise FileNotFoundError(
                f"{state_file} 不在 —— 这个 checkpoint 是老版本存的,只有权重。\n"
                f"  从它续训会丢掉优化器动量和数据位置,轨迹对不上。\n"
                f"  要么从带 train_state.pt 的 checkpoint 续,要么用 --model_path 重头开始。")
        resume_state = torch.load(state_file, weights_only=False, map_location="cpu")
        args.seed = resume_state["seed"]
        log(f"[resume] 从 {args.resume_from} 续:step {resume_state['step']}, "
            f"epoch {resume_state['epoch']}, 本 epoch 已吃 {resume_state['consumed']:,} chunk")

    torch.manual_seed(args.seed)
    param_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.param_dtype]
    model = Qwen3ForCausalLM.from_pretrained(
        load_dir, device=device, dtype=param_dtype)
    log(f"参数精度 {args.param_dtype}(计算恒为 bf16 autocast)")
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
    dataset = PreTokenizedDataset(args.train_data, args.loss_mask)

    # **验证集 = 留出尾部 N 行。**
    #
    # 不用重新编码:`prepare/chat.py` 是**全局 shuffle 之后才写盘**的,所以尾部
    # 那些行天然跨数据源分层,不会全来自同一个源。
    #
    # **它防的是训崩和过拟合,不是后训练的主判据。** v4 的实测:打包方式一改,
    # 中文停止率 +40 点、格式跟随 +32 点,而 loss 侧完全看不出改进(数据难度
    # 解释了全部差值)。拿 val loss 选 checkpoint 会把那一版判成平手。
    # 详见 docs/POSTTRAIN.md。
    n_train = len(dataset) - max(0, args.val_rows)
    if args.val_rows > 0:
        if n_train <= 0:
            raise ValueError(f"--val_rows {args.val_rows} 不小于总行数 {len(dataset)}")
        val_idx = list(range(n_train, len(dataset)))
        print(f"[val] 留出尾部 {len(val_idx):,} 行做验证,训练用前 {n_train:,} 行",
              flush=True)
    else:
        val_idx = []

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )

    # **数据顺序显式生成,不用 DataLoader 的 shuffle。**
    #
    # shuffle=True 时的排列藏在 DataLoader 内部,续训时没法「跳过前面吃过的
    # 那些 chunk」—— 只能从头再来一遍,于是断一次就重复喂一批数据。12B token
    # 的跑动里这不是小事:重复的部分等于白训,而且**看不出来**。
    #
    # 自己按 (seed, epoch) 生成排列,续训时切片跳过即可,轨迹与不中断完全一致。
    def epoch_order(epoch: int) -> list[int]:
        g = torch.Generator().manual_seed(args.seed * 1000003 + epoch)
        # 只在前 n_train 行里排列 —— 尾部留给验证,绝不能进训练
        return torch.randperm(n_train, generator=g).tolist()

    def make_loader(epoch: int, consumed: int):
        order = None if sampler is not None else epoch_order(epoch)[consumed:]
        if sampler is not None:
            sampler.set_epoch(epoch)
        return DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=sampler if sampler is not None else order,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        )

    @torch.no_grad()
    def eval_val() -> float:
        """验证集 loss。**sum / 有效 token 总数**,不是「每 batch 的均值再平均」——
        各行被监督的 token 数不一样,后者会给短行更大权重。"""
        model.eval()
        total, n_valid = 0.0, 0
        for i in range(0, len(val_idx), args.batch_size):
            rows = [dataset[j] for j in val_idx[i:i + args.batch_size]]
            b = collate_fn(rows, pad_token_id)
            ids = b["input_ids"].to(device)
            lab = b["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
            tgt = lab[:, 1:].reshape(-1)
            total += torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), tgt,
                ignore_index=IGNORE_INDEX, reduction="sum").item()
            n_valid += int((tgt != IGNORE_INDEX).sum())
        model.train()
        return total / max(n_valid, 1)

    # Optimizer (build on raw model so param names don't have DDP "module." prefix)
    optimizer = build_optimizer(raw_model, args.muon_lr, args.adam_lr, args.muon_momentum, args.weight_decay,
                                use_aurora=args.use_aurora, moonshot_scaling=args.moonshot_scaling,
                                adam_betas=(args.adam_beta1, args.adam_beta2),
                                adam_weight_decay=args.adam_weight_decay,
                                dmodel_lr_scale=args.dmodel_lr_scale)

    # **Muon 动量的预热。** nanochat 前 300 步把 momentum 从 0.85 拉到 0.95
    # (`base_train.py` 的 `get_muon_momentum`)。理由是刚开始梯度方向变化快,
    # 高动量会拖着走错方向;Muon 还要对动量缓冲做 Newton-Schulz 正交化,更敏感。
    # `--muon_momentum_warmup 0` 关掉,保持原行为(恒定 `--muon_momentum`)。
    def muon_momentum_at(step: int) -> float:
        w = args.muon_momentum_warmup
        if w <= 0:
            return args.muon_momentum
        f = min(1.0, step / w)
        return args.muon_momentum_start + f * (args.muon_momentum - args.muon_momentum_start)

    # LR scheduler: linear warmup, then decay to min_lr_ratio*peak floor.
    # Default schedule is cosine (Llama/Qwen standard); linear available for back-compat.
    # min_lr_ratio > 0 (e.g. 0.1) avoids the "decay-to-zero wastes last steps" pathology.
    import math
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        # WSD(warmup-stable-decay):中间一段恒定,只在最后 lr_decay_steps 步退火。
        #
        # 从零预训练要跑好几天,而**总步数在开跑的时候其实定不下来** —— 得看
        # 中途的 few-shot 翻译曲线才知道跑到哪儿够。cosine 把整条曲线绑死在
        # max_steps 上:想提前收尾,前面那些步就是按错的曲线走的;想延长,
        # 学习率已经衰到底了,续不动。
        #
        # WSD 的恒定段没有这个耦合:改主意时只要用新的 max_steps 续跑,
        # 前面走过的路一模一样,退火窗口自动落在新的末尾。
        if args.lr_schedule == "wsd":
            decay = args.lr_decay_steps or max(1, int(0.1 * args.max_steps))
            decay_start = args.max_steps - decay
            if step < decay_start:
                return 1.0
            p = min(1.0, (step - decay_start) / max(1, decay))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * (1.0 - p)
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
    # 日志用:一个 optimizer step 内所有 micro-batch 的 loss 之和(每个已经除过
    # accum,所以和就是均值),以及它的 EMA。
    step_loss_sum = 0.0
    loss_ema: float | None = None
    best_val: float | None = None
    epoch, consumed = 0, 0
    t0 = time.time()
    interrupted = False

    def _move_optimizer_state(target):
        """Move all optimizer state tensors to target device."""
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(target, non_blocking=False)

    if resume_state is not None:
        # **优化器状态不能不恢复。** Muon 的动量是 5 步 Newton-Schulz 正交化之后
        # 的累积量,丢掉它等于让模型在训练中途重新经历一次冷启动 —— loss 会
        # 明显跳一下再慢慢压回去,而日志上只显示「续训成功」。
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state(device)
        step = resume_state["step"]
        epoch, consumed = resume_state["epoch"], resume_state["consumed"]
        if args.reset_data_position:
            # 换数据集续训(比如退火段换成带平行语料的那份)必须清掉位置。
            # train_state 里的 consumed 是**旧数据集的索引**:拿 1024 万去切
            # 一个 460 万 chunk 的新数据集,切出来是空的 —— DataLoader 一条都
            # 不吐,for 循环直接结束、epoch 自增,然后才从头开始。不报错,
            # 只是白转一圈还把 epoch 记歪了。
            log(f"[resume] --reset_data_position:丢掉旧数据位置"
                f"(epoch {epoch} / 已吃 {consumed:,} chunk),从新数据集头部开始")
            epoch, consumed = 0, 0
        # scheduler 是 step 的纯函数,不用存 —— 快进到该在的位置就行。
        # 这样即使 max_steps 变了(WSD 提前收尾),退火窗口自动落在新末尾。
        for _ in range(step):
            scheduler.step()
        torch.set_rng_state(resume_state["cpu_rng"])
        if resume_state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(resume_state["cuda_rng"])
        lrs = ", ".join(f"{g['lr']:.3e}" for g in optimizer.param_groups)
        log(f"[resume] 优化器状态已恢复 | step {step} | lr [{lrs}]")

    def _save_train_state(save_path, cur_step, cur_epoch, cur_consumed):
        """权重旁边存下续训需要的全部状态。

        权重本身由 `save_pretrained` 存成 HF 布局(评测和发布只要那一份);
        这里存的是**只有续训才用得上**的东西,单独一个文件,不污染发布产物。
        """
        torch.save({
            "step": cur_step, "epoch": cur_epoch, "consumed": cur_consumed,
            "seed": args.seed,
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, os.path.join(save_path, "train_state.pt"))

    def _prune_checkpoints(keep_last, milestone):
        """删掉旧 checkpoint,只留最近几份和里程碑。

        每份 = 权重 2.1GB(fp32)+ 优化器状态 2.4GB(Muon 动量 + Adam 两个矩)。
        45,776 步每 2000 步存一次就是 22 份 ≈ 100GB —— 跑到一半磁盘满,
        训练进程写 checkpoint 失败,而**前面几天的进度只剩最后一份能用**。

        留里程碑是为了事后能回看轨迹(比如某一段 BLEU 掉了想回去查),
        只留最近几份的话那些点就永远取不回来了。
        """
        if keep_last <= 0:
            return
        import re
        import shutil
        dirs = []
        for name in os.listdir(args.output_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m:
                dirs.append((int(m.group(1)), os.path.join(args.output_dir, name)))
        dirs.sort()
        keep = {s for s, _ in dirs[-keep_last:]}
        if milestone > 0:
            keep |= {s for s, _ in dirs if s % milestone == 0}
        for s, path in dirs:
            if s not in keep:
                shutil.rmtree(path, ignore_errors=True)
                log(f"[prune] 删掉 checkpoint-{s}(保留最近 {keep_last} 份"
                    f"{f' + 每 {milestone} 步的里程碑' if milestone > 0 else ''})")

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
            ckpt = os.path.join(args.output_dir, f"checkpoint-{current_step}")
            cmd = (args.inline_eval_cmd
                   .replace("{step}", str(current_step))
                   .replace("{ckpt}", ckpt))
            log(f"[inline_eval] running: {cmd}")
            t_ev = time.time()
            rc = os.system(cmd)
            # **失败要说出来。** 中途评测挂了而训练继续跑,表现就是「曲线没更新」——
            # 很容易当成模型没进步,而不是评测根本没跑起来。
            log(f"[inline_eval] done in {time.time()-t_ev:.0f}s"
                + ("" if rc == 0 else f"  **退出码 {rc >> 8} —— 这次评测失败了**"))
        if is_distributed:
            dist.barrier()
        # All ranks move back
        raw_model.to(device)
        _move_optimizer_state(device)
        torch.cuda.empty_cache()

    import signal
    def _stop_handler(sig, frame):
        nonlocal interrupted
        if interrupted:  # 第二次直接退
            raise KeyboardInterrupt
        interrupted = True
        name = {signal.SIGINT: "SIGINT(Ctrl+C)", signal.SIGTERM: "SIGTERM"}.get(sig, str(sig))
        print(f"\n收到 {name},在 step {step} 存 checkpoint 后退出 ...", flush=True)
    # **SIGTERM 也要接。** 只接 SIGINT 的话,被 kill、被会话清理、被系统关机
    # 带走时都是硬杀 —— 退回上一个 save_steps 的 checkpoint,中间那些步全丢。
    # 2026-07-29 真踩了一次:step 5000 被 SIGTERM,只能从 checkpoint-4000 续,
    # 白跑 1000 步(3.5 小时)。
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    while step < args.max_steps and not interrupted:
        dataloader = make_loader(epoch, consumed)
        for batch in dataloader:
            if step >= args.max_steps or interrupted:
                break
            consumed += batch["input_ids"].shape[0]
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 自写的模型只返回 logits,loss 在这里算(HF 是在模型内部算的)。
                # 口径与 Qwen3ForCausalLM 一致:左移一位的 next-token CE,
                # IGNORE_INDEX(-100) 处不计入。
                logits = model(batch["input_ids"])
                loss = chunked_cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    batch["labels"][:, 1:].reshape(-1),
                    chunk=args.ce_chunk)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            step_loss_sum += loss.item()
            loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.muon_momentum_warmup > 0:
                    m = muon_momentum_at(step)
                    for g in optimizer.param_groups:
                        if g.get("use_muon"):
                            g["momentum"] = m
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # **loss 要取整个 step 的平均,不是最后一个 micro-batch 的。**
                # 原来写的是 `loss.item() * accum`,那只是把最后一个 micro-batch
                # 的值还原回去 —— accum=4 时报的是四分之一的数据,波动 ±0.25,
                # 曾经让我误判成 loss 在上升。EMA 是给趋势用的:单点波动比整段
                # 的下降幅度还大,**要拿 loss 下结论必须看 EMA**。
                # nanochat 也只打 EMA(mid_train.py 的 ema_beta = 0.9)。
                step_loss = step_loss_sum
                loss_ema = (step_loss if loss_ema is None
                            else 0.9 * loss_ema + 0.1 * step_loss)
                step_loss_sum = 0.0

                if step % args.logging_steps == 0:
                    elapsed = time.time() - t0
                    lrs = [f"{g['lr']:.6f}" for g in optimizer.param_groups]
                    # 动量在预热期内要**打出来** —— `src/optim.py` 每步从 group 里
                    # 读 `beta=group["momentum"]`,所以逐步改是生效的;但改错了
                    # 不会报错,所以让它可见。
                    mom = ""
                    if args.muon_momentum_warmup > 0 and step <= args.muon_momentum_warmup + 1:
                        # `step` 此时已自增,而动量是在自增**前**取的 —— 打 step-1
                        # 才是这一步真正用的值(第一步是 muon_momentum_start)
                        mom = f" | mom {muon_momentum_at(step - 1):.4f}"
                    log(f"step {step}/{args.max_steps} | loss {step_loss:.4f} | "
                        f"ema {loss_ema:.4f} | "
                        f"lr [{', '.join(lrs)}]{mom} | {elapsed:.1f}s")

                if val_idx and args.eval_steps > 0 and step % args.eval_steps == 0:
                    vl = eval_val()
                    best_val = vl if best_val is None else min(best_val, vl)
                    log(f"step {step}/{args.max_steps} | **val loss {vl:.4f}** | "
                        f"最优 {best_val:.4f}"
                        + ("  ← 新低" if vl == best_val else ""))

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
                        _save_train_state(save_path, step, epoch, consumed)
                        print(f"Saved checkpoint to {save_path}", flush=True)
                        _prune_checkpoints(args.keep_ckpt, args.ckpt_milestone)
                    if is_distributed:
                        dist.barrier()
                    # Inline eval (CPU-offload pattern) if requested
                    _inline_eval(step)

        # 一个 epoch 走完(12B token 的跑动里通常一次都走不完)
        epoch += 1
        consumed = 0

    # 被 Ctrl+C 打断也要留下能续的 checkpoint —— 否则「优雅退出」只保住权重,
    # 优化器动量和数据位置照样丢,下次续训是带着冷启动的假续训。
    if interrupted and is_main and args.output_dir and not args.use_lora:
        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        raw_model.save_pretrained(save_path)
        _copy_tokenizer_artifacts(args.model_path, save_path)
        _save_train_state(save_path, step, epoch, consumed)
        log(f"中断于 step {step},已存可续的 checkpoint:{save_path}")

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
        print(f"Saved final model to {args.output_dir}", flush=True)

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
    parser.add_argument("--val_rows", type=int, default=0,
                        help="留出**尾部**多少行做验证(0=不留)。chat.py 是全局 "
                             "shuffle 后才写盘的,所以尾部天然跨数据源分层")
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="每多少步测一次验证 loss。0=不测。nanochat 用 150")
    parser.add_argument("--muon_lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    # 下面五个是为了对齐 nanochat(docs/POSTTRAIN.md 的 A 组)。
    # **默认值一律保持原行为** —— 预训练和 ReTok 两条线也在用这个脚本,
    # 而它们已经发布了权重,改默认等于悄悄换掉它们的配方。
    parser.add_argument("--muon_momentum_warmup", type=int, default=0,
                        help="Muon 动量从 --muon_momentum_start 线性拉到 "
                             "--muon_momentum 的步数。0=关(恒定)。nanochat 用 300")
    parser.add_argument("--muon_momentum_start", type=float, default=0.85,
                        help="动量预热的起点,只在 --muon_momentum_warmup > 0 时生效")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="nanochat 用 0.8")
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=None,
                        help="AdamW 组单独的 wd。不给=沿用 --weight_decay(原行为)。"
                             "nanochat 硬编码 0,wd 只给 Muon")
    parser.add_argument("--dmodel_lr_scale", action="store_true",
                        help="AdamW 的 lr 乘 (d_model/768)^-0.5,对齐 nanochat。"
                             "d_model=1024 时是 ×0.866")
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
                        help="从 checkpoint-N/ 续训:恢复权重、优化器状态、step、"
                             "RNG 和数据顺序。跑几天的任务必须有这个 —— save_steps "
                             "只存权重,断了从头再来等于赌全程不断电。")
    parser.add_argument("--lr_total_steps", type=int, default=0,
                        help="(unused placeholder; --max_steps drives cosine in current flow)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Floor for the LR schedule, as a fraction of peak LR. Default 0.1 matches "
                             "Llama/Qwen practice (peak/10). Set to 0 to recover old decay-to-zero behavior.")
    parser.add_argument("--lr_schedule", choices=["cosine", "linear", "wsd"], default="cosine",
                        help="warmup 之后的形状。cosine 是 Llama/Qwen 标准(ReTok 两阶段用它);"
                             "wsd 恒定到末段再退火,适合总步数没定死的从零预训练。")
    parser.add_argument("--lr_decay_steps", type=int, default=0,
                        help="wsd 的退火窗口(最后多少步)。0 = max_steps 的 10%%。")
    parser.add_argument("--loss_mask", default=None,
                        help="与 --train_data 一一对应的掩码 .pt(uint8,1=算 loss)。"
                             "midtrain/SFT 用;不给就整段算 loss(预训练那样)。"
                             "由 prepare/chat.py 产出。")
    parser.add_argument("--seed", type=int, default=42,
                        help="初始化、数据顺序、dropout 的种子。--resume_from 会覆盖成"
                             "存下来的那份,保证续训和不中断跑出来的轨迹一致。")
    parser.add_argument("--param_dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="参数本身的精度(计算恒为 bf16 autocast)。"
                             "**从零预训练用 float32** —— bf16 的相对精度是 1/256,"
                             "退火末段 lr 降到 3e-5 时,单步更新只占权重的 0.15%%,"
                             "低于 bf16 的分辨率,会被直接舍掉。继续预训练(ReTok "
                             "两阶段)保持 bfloat16,与 v18 一致。")
    parser.add_argument("--reset_data_position", action="store_true",
                        help="续训时不沿用 train_state 里的数据位置。**换数据集续训"
                             "必须加这个** —— consumed 是旧数据集的索引,拿去切新的"
                             "会切空。优化器状态和 step 照常恢复。")
    parser.add_argument("--keep_ckpt", type=int, default=0,
                        help="只保留最近 N 份 checkpoint(0 = 全留,旧行为)。"
                             "从零预训练每份 4.5GB、要存 20 多次,不清理会撑爆磁盘。")
    parser.add_argument("--ckpt_milestone", type=int, default=10000,
                        help="步数是它的倍数的 checkpoint 不删 —— 事后回看轨迹要用。")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader 工作进程数。mmap 之后取数据只是页缺失 + "
                             "int32→int64,开 2 个足够把它和 GPU 计算重叠。")
    parser.add_argument("--ce_chunk", type=int, default=0,
                        help="分块算 CE 的块大小(行数)。0 = 不分块。词表 81903 时"
                             "logits.float() 是显存大头,4096 能把 batch 16 从 OOM 救回来。")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)
