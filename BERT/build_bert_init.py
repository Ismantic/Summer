"""BERT-base init(独立可移植,不依赖 Summer 项目):
  - 12 layers / 768 hidden / 12 heads / 3072 ff
  - vocab = (piece vocab) + 1(<mask> 占末位 id)
  - max_position 1024
  - 随机初始化(BERT 原生,std=0.02)

用法:
  python build_bert_init.py \
      --piece_dir <含 piece.model + dict.txt + token_mapping.json 的目录> \
      --output_dir bert_init \
      --piece_vocab_size 81903

  piece_dir 提供的 5 个文件(piece.model / dict.txt / token_mapping.json /
  special_tokens_map.json / tokenizer_config.json)会拷到 output_dir 旁,
  方便后续 inference 加载。
"""
import argparse, os, shutil, torch
from transformers import BertConfig, BertForMaskedLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece_dir", required=True,
                    help="含 piece.model / dict.txt 的目录")
    ap.add_argument("--output_dir", default="bert_init")
    ap.add_argument("--piece_vocab_size", type=int, required=True,
                    help="piece tokenizer 的 vocab 大小(BERT vocab 会是此值+1,末位给 <mask>)")
    ap.add_argument("--pad_token_id", type=int, default=0,
                    help="piece 里 <pad> 的 id(BERT config 用)")
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--intermediate_size", type=int, default=3072)
    ap.add_argument("--max_position", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mask_token_id = args.piece_vocab_size       # 新 mask 占末位
    total_vocab = args.piece_vocab_size + 1     # piece + 1 mask

    os.makedirs(args.output_dir, exist_ok=True)

    config = BertConfig(
        vocab_size=total_vocab,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=args.max_position,
        type_vocab_size=1,             # MLM only,no NSP
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=args.pad_token_id,
    )

    torch.manual_seed(args.seed)
    model = BertForMaskedLM(config)
    model.init_weights()

    total = sum(p.numel() for p in model.parameters())
    print(f"BERT init:")
    print(f"  vocab_size: {total_vocab} (piece {args.piece_vocab_size} + 1 mask at id {mask_token_id})")
    print(f"  hidden: {config.hidden_size}, layers: {config.num_hidden_layers}, heads: {config.num_attention_heads}")
    print(f"  max_position: {config.max_position_embeddings}")
    print(f"  total params: {total/1e6:.1f}M")
    print(f"  embed: {model.bert.embeddings.word_embeddings.weight.numel()/1e6:.1f}M")

    model.save_pretrained(args.output_dir, safe_serialization=True)
    print(f"\nSaved to {args.output_dir}/")

    # 拷贝 piece tokenizer artifacts
    for f in ["piece.model", "dict.txt", "token_mapping.json",
              "special_tokens_map.json", "tokenizer_config.json"]:
        src = os.path.join(args.piece_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, f))
            print(f"  copied {f}")
        else:
            print(f"  WARN: {f} not in {args.piece_dir}")

    with open(os.path.join(args.output_dir, "mask_token_id.txt"), "w") as f:
        f.write(str(mask_token_id))
    print(f"\nmask_token_id = {mask_token_id} (recorded in mask_token_id.txt)")


if __name__ == "__main__":
    main()
