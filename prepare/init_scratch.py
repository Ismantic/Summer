"""造一个随机初始化的起点 —— Summer-0.5B 从零预训练的第 0 步。

与 `retok.py` 是并列的两个入口,产出格式完全一样(HF 布局 + 分词器产物 +
token_mapping.json),区别只在权重从哪来:

    retok.py         Qwen3-1.7B-Base 的权重 → 换词表 → 嵌入按 mean-of-BBPE 映射
    init_scratch.py  没有任何已有权重 → 按 initializer_range 随机初始化

## 架构不抄,直接读 Qwen3-0.6B-Base 的 config.json

28 层 / 1024 / 3072 / GQA 16:8 / head_dim 128 这些数字**不写进这个文件**。
抄错一个不会报错,只会训出一个「差不多但不是那个架构」的模型,而且事后
无从对证。所以从 `data/source.py` 登记的 `qwen_base_06b` 读真的 config,
只覆盖两处:词表大小(151936 → 81903)和权重精度。

## 为什么产物叫 0.5B 而基座叫 0.6B

换掉词表之后 tie 的嵌入从 155.6M 降到 83.9M:

    Qwen3-0.6B-Base   151936 × 1024 + 440.5M = 596.0M
    Summer-0.5B        81903 × 1024 + 440.5M = 524.3M

transformer 部分一个参数没动,少掉的 71.8M 全在嵌入。**叫 0.6B 会误导** ——
差 12%,而且差在最容易被当成「同规模」的地方。

## 精度

默认存 float32。从零训练全程用 fp32 参数 + bf16 autocast:bf16 的相对精度是
1/256,退火末段单步更新只占权重的千分之一几,直接被舍掉。理由详见
`src/train.py --param_dtype` 的帮助。
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import source as _source                                # noqa: E402
from prepare.retok import SPECIAL_TOKENS, write_tokenizer_files    # noqa: E402
from prepare.tokenizer import resolve_assets                       # noqa: E402
from src.model import Qwen3Config, Qwen3ForCausalLM                # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_arch(arch_dir: str) -> dict:
    """读架构模板的 config.json。本地没有就报错,不去联网猜。"""
    path = os.path.join(arch_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 不在。先跑 `python data/download.py qwen_base_06b`。\n"
            f"  架构数字不写死在代码里 —— 抄错了不报错,只会训出个"
            f"「差不多但不是 Qwen3-0.6B-Base」的模型。")
    return json.loads(open(path).read())


def scale_arch(raw: dict, depth: int) -> dict:
    """把架构模板按 **nanochat 的 depth 惯例**缩放:`hidden = depth × 64`。

    用途是**给流程做预演** —— 拿一个小模型把整条链(编码 → 预训练 → midtrain →
    SFT → 评测)走通,再上大的,省掉在 0.5B 上返工 4 天。

    保留 Qwen3 的比例(`--arch_from` 那份模板给的):`intermediate = 3 × hidden`、
    GQA 2:1、`head_dim` 不变(128)。**只动 L 和 h**,别的照抄模板 ——
    「模型结构就用 Qwen3」。

    ## 小模型的数字有哪些不能当真

    词表 81903 是固定的,所以**模型越小,嵌入占比越高**:

        28L/1024H   524.3M   嵌入 16.0%   ← Summer-0.5B
        20L/1280H   596.4M        17.6%   ← nanochat d20 的对应鳞级
        12L/ 768H   169.1M        37.2%
        10L/ 640H   113.9M        46.0%

    16% 和 46% 不是同一种模型。**凡是和嵌入/词表耦合的结论都不能从小模型往上推**
    —— 特殊 token 学不学得动、tie 的影响、停止率、格式跟随,这几样恰好都是
    `docs/POSTTRAIN.md` 在追的。小模型用来验「链路跑不跑得通」,不是用来定配方。
    """
    if depth <= 0:
        return raw
    h = depth * 64
    hd = raw["head_dim"]

    # **注意力的头数不能用 `hidden // head_dim` 算。** Qwen3-0.6B 是
    # 16 heads × head_dim 128 = **2048**,而 hidden 只有 1024 —— q/o 的投影维度
    # 是 hidden 的 2 倍。按 hidden//head_dim 算会得出 8 个头(缩水一半),
    # 而且**不会报错**,只会训出个「差不多但不是 Qwen3」的模型。
    # 正确做法:保留模板的两个比例。
    q_ratio = raw["num_attention_heads"] * hd / raw["hidden_size"]      # 0.6B 是 2.0
    gqa = raw["num_attention_heads"] // raw["num_key_value_heads"]      # 0.6B 是 2
    mlp_ratio = raw["intermediate_size"] / raw["hidden_size"]           # 0.6B 是 3.0

    nh = round(q_ratio * h / hd)
    if nh < gqa or nh % gqa:
        raise ValueError(
            f"depth {depth} → hidden {h} 算出 {nh} 个头,不能被 GQA 比例 {gqa} 整除。"
            f"换一个 depth(hidden 需要是 {int(hd * gqa / q_ratio)} 的倍数)。")
    out = dict(raw)
    out.update(num_hidden_layers=depth, hidden_size=h,
               intermediate_size=int(mlp_ratio * h),
               num_attention_heads=nh,
               num_key_value_heads=nh // gqa)
    return out


def main(args):
    piece_model, cn_dict = (args.tokenizer_model, args.cn_dict)
    if not piece_model:
        piece_model, cn_dict = resolve_assets()

    import piece_tokenizer as pt
    tok = pt.Tokenizer()
    tok.load(piece_model, cn_dict)
    vocab_size = tok.vocab_size()
    special_token_ids = {}
    for t in SPECIAL_TOKENS:
        idx = tok.piece_to_id(t)
        if idx < 0:
            raise RuntimeError(f"词表里没有特殊 token {t}:{piece_model}")
        special_token_ids[t] = idx

    raw = load_arch(args.arch_from)
    if args.depth:
        base = (f"{raw['num_hidden_layers']}L/{raw['hidden_size']}H")
        raw = scale_arch(raw, args.depth)
        print(f"**按 depth {args.depth} 缩放**:{base} → "
              f"{raw['num_hidden_layers']}L/{raw['hidden_size']}H"
              f"(hidden = depth×64,nanochat 的惯例)")
        print(f"  小模型是给**流程预演**用的 —— 嵌入占比会高很多,"
              f"和嵌入/词表耦合的结论不能往上推,见 scale_arch 的说明")
    print(f"架构模板 {args.arch_from}")
    print(f"  {raw['num_hidden_layers']}L / {raw['hidden_size']}H / "
          f"{raw['intermediate_size']}I / GQA {raw['num_attention_heads']}:"
          f"{raw['num_key_value_heads']} / head_dim {raw['head_dim']}")

    fields = set(Qwen3Config.__dataclass_fields__)
    kwargs = {k: v for k, v in raw.items() if k in fields}
    kwargs["vocab_size"] = vocab_size                 # 151936 → 81903
    kwargs["torch_dtype"] = args.dtype
    kwargs["initializer_range"] = args.initializer_range
    cfg = Qwen3Config(**kwargs)

    print(f"词表 {raw['vocab_size']} → {vocab_size}  "
          f"(pad={special_token_ids['<pad>']})")
    model = Qwen3ForCausalLM(cfg).to(DTYPES[args.dtype])
    model.init_weights(seed=args.seed)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    n_all = sum(p.numel() for p in model.parameters())
    n_emb = model.model.embed_tokens.weight.numel()
    print(f"参数 {n_all:,}  (嵌入 {n_emb:,} 与 lm_head 绑权重 / "
          f"transformer {n_all - n_emb:,})")
    print(f"初始化 N(0, {args.initializer_range})、残差出口 ÷√(2×"
          f"{cfg.num_hidden_layers})、RMSNorm 全 1、seed {args.seed}")

    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    write_tokenizer_files(
        SimpleNamespace(new_tokenizer_path=piece_model, cn_dict=cn_dict),
        special_token_ids, args.output_path)

    # 与 retok.py 的产出同构 —— 训练和评测两边都按同一份 key 读 pad_id。
    mapping = {
        "base_vocab_size": vocab_size,
        "total_vocab_size": vocab_size,
        "special_tokens": special_token_ids,
        "bos_id": 1, "eos_id": 2, "unk_id": 0,
        "pad_id": special_token_ids["<pad>"],
        "user_id": special_token_ids["<user>"],
        "assistant_id": special_token_ids["<assistant>"],
        "system_id": special_token_ids["<system>"],
    }
    with open(os.path.join(args.output_path, "token_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # **generation_config.json 是必需的。** retok.py 那条线是靠 transformers 的
    # model.generation_config 顺带写出来的(retok.py:219),这条线自己造模型,
    # 没有那个对象 —— 漏了之后一路无感,直到 save/export.py 拒绝导出
    # (它把这个文件列为必需)。而真正的风险在漏了却没人拦的时候:生成时
    # eos/pad 走 HF 默认值,和这套 81903 词表对不上。
    gen_cfg = {
        "bos_token_id": mapping["bos_id"],
        "eos_token_id": mapping["eos_id"],
        "pad_token_id": mapping["pad_id"],
        "max_new_tokens": 2048,
    }
    with open(os.path.join(args.output_path, "generation_config.json"), "w") as f:
        json.dump(gen_cfg, f, indent=2)

    print(f"\n完成:{args.output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch_from", default=str(_source.get("qwen_base_06b").dir()),
                   help="架构模板目录(读它的 config.json)。默认 Qwen3-0.6B-Base。")
    p.add_argument("--output_path", required=True)
    p.add_argument("--tokenizer_model", default="",
                   help="留空则用 resolve_assets() 从 clone 的 PieceTokenizer 反查")
    p.add_argument("--cn_dict", default="")
    p.add_argument("--dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--initializer_range", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=0,
                   help="按 nanochat 惯例缩放架构:hidden = depth×64,层数 = depth。"
                        "0=不缩放(用 --arch_from 的原尺寸)。"
                        "**给流程预演用**,小模型的嵌入占比高很多,"
                        "和嵌入/词表耦合的结论不能往上推")
    main(p.parse_args())
