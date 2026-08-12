"""拿发布的 base 模型做续写,交互式。

    python -m save.generate                      # 默认用当前 SOTA ckpt
    python -m save.generate --model_path <dir>
    python -m save.generate --prompt "中国科学院"  # 单次,不进交互

**这是 base 模型,不是 chat 模型。** 它只会接着往下写,不会回答问题、不会
听指令 —— 这个项目不做指令微调(那是下游 Interpreter 的事)。

只依赖 torch + PieceTokenizer:模型走 `src/model.py`,分词走
`prepare/tokenizer.py`。没有 transformers,也没有 vllm。
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare.tokenizer import PieceTokenizerWrapper                # noqa: E402
from src.model import Qwen3ForCausalLM                             # noqa: E402


def resolve_ckpt() -> Path | None:
    for c in ("save/sota/v18_p2_tie", "output/phase2_ckpt_v18_tie",
              "save/releases/Qwen3-1.7B-Base-ReTok"):
        if (ROOT / c / "config.json").exists():
            return ROOT / c
    return None


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int, temperature: float,
             top_p: float, device: str) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device=device)
    stops = set(tok.stop_token_ids)
    out = []
    for _ in range(max_new):
        # 没有 KV cache —— 每步重算整个前缀。短续写够用,长文本会慢。
        logits = model(x)[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            srt, idx = torch.sort(probs, descending=True)
            keep = (torch.cumsum(srt, 0) - srt) < top_p
            srt, idx = srt[keep], idx[keep]
            nxt = int(idx[torch.multinomial(srt / srt.sum(), 1)])
        if nxt in stops:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None)
    p.add_argument("--prompt", default=None, help="给了就单次运行,不进交互")
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0 表示贪心")
    p.add_argument("--top_p", type=float, default=0.9)
    a = p.parse_args()

    ckpt = Path(a.model_path) if a.model_path else resolve_ckpt()
    if ckpt is None or not (ckpt / "config.json").exists():
        print("找不到模型目录。传 --model_path,或从 HF 下:")
        print("  huggingface-cli download Ismantic/Qwen3-1.7B-Base-ReTok "
              "--local-dir save/releases/Qwen3-1.7B-Base-ReTok")
        return 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 {ckpt}({dev})…")
    tok = PieceTokenizerWrapper(str(ckpt))
    model = Qwen3ForCausalLM.from_pretrained(ckpt, device=dev,
                                             dtype=torch.bfloat16)

    def run(text):
        print(f"\n{text}", end="", flush=True)
        print(generate(model, tok, text, a.max_new, a.temperature, a.top_p, dev))

    if a.prompt:
        run(a.prompt)
        return 0

    print("\nbase 模型续写。它不会回答问题,只会接着往下写。Ctrl-D 退出。")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text:
            run(text)


if __name__ == "__main__":
    sys.exit(main())
