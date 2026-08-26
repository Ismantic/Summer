"""vLLM-based translation eval, drop-in replacement for eval_pretrain_translate.py.

Uses vLLM's independent generation engine (PagedAttention) instead of
transformers.generate() — decouples eval from transformers version instability.

Tokenization still uses our piece_tokenizer (we pass prompt_token_ids directly).
COMET evaluation reuses the same code path as the transformers version.

Usage (mirrors eval_pretrain_translate.py):
    python prepare/translate.py \\
        --model_path /path/to/ckpt --output_path /path/out/wmt22.json \\
        --max_samples 1000 --batch_size 64 \\
        --compute_comet --save_all_samples
"""
import argparse
import json
import os
import time
import sys

import sacrebleu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)




# ---- 以下来自原 eval_pretrain_translate.py(非 vLLM 版)----
import argparse
import json
import os
import random
import sys
import time

import sacrebleu
import torch
from transformers import AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


PROMPT_HEADER = {
    "zh-en": "Translate Chinese to English.\n\n",
    "en-zh": "Translate English to Chinese.\n\n",
}
SRC_LABEL = {"zh-en": "Chinese", "en-zh": "English"}
TGT_LABEL = {"zh-en": "English", "en-zh": "Chinese"}



def _default_comet() -> str:
    """COMET 模型目录 —— 从 data/source.py 的注册表取,不写死本机路径。"""
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from data import source as _s
    return str(_s.get("comet").dir())

def load_tokenizer(model_path):
    from prepare.tokenizer import PieceTokenizerWrapper, has_piece_vocab
    if has_piece_vocab(model_path):
        return PieceTokenizerWrapper(model_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_pair(testset: str, direction: str):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def pick_exemplars(direction: str, n: int, seed: int = 0,
                   exemplar_set: str = "wmt21"):
    """Pull n (src, ref) pairs from a fixed earlier test set, deterministic."""
    src, ref = load_pair(exemplar_set, direction)
    rng = random.Random(seed)
    idxs = list(range(len(src)))
    rng.shuffle(idxs)
    picked = idxs[:n]
    return [(src[i], ref[i]) for i in picked]


def build_prompt(direction: str, exemplars, test_src: str) -> str:
    s_label = SRC_LABEL[direction]
    t_label = TGT_LABEL[direction]
    parts = [PROMPT_HEADER[direction]]
    for s, r in exemplars:
        parts.append(f"{s_label}: {s}\n{t_label}: {r}\n\n")
    parts.append(f"{s_label}: {test_src}\n{t_label}:")
    return "".join(parts)


def encode_prompts(tokenizer, prompts, add_bos=False):
    """Returns list[list[int]] of token id lists.

    **默认不加 BOS**,和 S0/S1/S2 的历史 BLEU/COMET 数字同协议——那几个
    模型是 `stream` 打包训的,训练时从没见过 BOS,加了反而是分布外。
    `bos_bestfit` 打包的模型(S0B 起)训练时每行都以 BOS 开头,这个多段
    拼接、`add_special_tokens=False` 的 few-shot 提示对它是彻底的分布外
    输入——2026-08-26 实测:S0B 不加 BOS 时 5-shot 翻译直接复读崩溃
    ("1, 2, \\n, \\n..."),补一个 BOS 就恢复成正常的(虽然还是不对题,但
    流畅的)续写。按训练用的打包方式选 `add_bos`,不是随便开。
    """
    out = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if add_bos:
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is None:
                raise ValueError("--add_bos 但 tokenizer 没有 bos_token_id")
            ids = [bos] + ids
        out.append(ids)
    return out


def trim_continuation(text: str, src_label: str) -> str:
    """Stop at the next 'English:' / 'Chinese:' header (newline OR space-separated)
    or first blank line. Models trained on diverse data sometimes emit space-only
    separators instead of newlines, so we match both."""
    stops = ("\n\n", "\n English:", "\n Chinese:",
             "\nEnglish:", "\nChinese:", "  English:", "  Chinese:",
             f"  {src_label}:", f"\n{src_label}:")
    cuts = [text.find(s) for s in stops]
    cuts = [c for c in cuts if c != -1]
    if cuts:
        text = text[:min(cuts)]
    return text.strip()


def compute_comet(sources, translations, references, comet_model_path, batch_size=16):
    """Compute COMET score using a local Unbabel/wmt22-comet-da snapshot."""
    from comet import load_from_checkpoint
    import glob
    # COMET package expects a .ckpt file inside the model dir
    ckpts = glob.glob(f"{comet_model_path}/**/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found under {comet_model_path}")
    model = load_from_checkpoint(ckpts[0])
    data = [{"src": s, "mt": t, "ref": r}
            for s, t, r in zip(sources, translations, references)]
    output = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
    return float(output.system_score), [float(x) for x in output.scores]


def evaluate_vllm(llm, tokenizer, direction, testset, exemplar_set, num_fewshot,
                  max_samples, max_new_tokens, seed, save_all_samples=False,
                  repetition_penalty=1.0, add_bos=False):
    from vllm import SamplingParams
    sources, references = load_pair(testset, direction)
    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]
    exemplars = pick_exemplars(direction, num_fewshot, seed=seed,
                                exemplar_set=exemplar_set) if num_fewshot > 0 else []
    prompts = [build_prompt(direction, exemplars, s) for s in sources]
    prompt_token_ids = encode_prompts(tokenizer, prompts, add_bos=add_bos)

    # 默认 rp=1.0(纯贪心),和历史上所有报过的 BLEU/COMET 数字同一协议,可以
    # 直接比——同样的理由见 stoprate.py 的 rp 说明:跑分协议和部署协议是
    # 两件事,混着比会把噪声当结论。
    sampling = SamplingParams(
        temperature=0.0,                     # greedy
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        repetition_penalty=repetition_penalty,
    )

    t0 = time.time()
    # vLLM 0.21 expects list of TokensPrompt objects (or dicts with prompt_token_ids).
    from vllm.inputs import TokensPrompt
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    outputs = llm.generate(
        inputs,
        sampling_params=sampling,
        use_tqdm=False,
    )
    elapsed = time.time() - t0

    # vLLM returns in same order as input
    translations = []
    for o in outputs:
        gen_ids = list(o.outputs[0].token_ids)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        translations.append(trim_continuation(text, SRC_LABEL[direction]))

    bleu = sacrebleu.corpus_bleu(
        translations, [references],
        tokenize="zh" if direction == "en-zh" else "13a",
    )
    n_save = len(sources) if save_all_samples else min(5, len(sources))
    return {
        "direction": direction,
        "testset": testset,
        "exemplar_set": exemplar_set,
        "num_fewshot": num_fewshot,
        "n_samples": len(sources),
        "bleu": bleu.score,
        "time_s": elapsed,
        "translations": translations if save_all_samples else None,
        "samples": [
            dict(src=sources[k], hyp=translations[k], ref=references[k])
            for k in range(n_save)
        ],
        "_sources": sources,
        "_references": references,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--testset", default="wmt22",
                   choices=["wmt17", "wmt18", "wmt19", "wmt20", "wmt21", "wmt22", "wmt23"])
    p.add_argument("--exemplar_set", default="wmt21")
    p.add_argument("--direction", default="both", choices=["zh-en", "en-zh", "both"])
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", default=None)
    p.add_argument("--compute_comet", action="store_true")
    p.add_argument("--comet_model_path", default=_default_comet())
    p.add_argument("--save_all_samples", action="store_true")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0=纯贪心(默认,和历史数字同协议)。诊断复读崩溃时"
                        "才临时调,报数要显式标出来,见 stoprate.py 的说明。")
    p.add_argument("--add_bos", action="store_true",
                   help="默认 False,和 S0/S1/S2 历史数字同协议。bos_bestfit "
                        "打包训出来的模型(S0B 起)不加这个跑 5-shot 会复读"
                        "崩溃,见 encode_prompts() 的说明。")
    p.add_argument("--vllm_dtype", default="bfloat16")
    p.add_argument("--gpu_mem_util", type=float, default=0.85,
                   help="Leave headroom for COMET model. 0.85 = use 85% of GPU.")
    args = p.parse_args()
    if args.exemplar_set == args.testset:
        raise ValueError("--exemplar_set must differ from --testset")

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)
    print(f"Loading vLLM engine from {args.model_path}...")

    from vllm import LLM
    llm = LLM(
        model=args.model_path,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        skip_tokenizer_init=True,   # we tokenize ourselves with piece_tokenizer
        trust_remote_code=True,
        enforce_eager=False,
    )

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    out = {"model_path": args.model_path, "results": {}}
    for d in directions:
        print(f"\n=== {d} (testset={args.testset}, {args.num_fewshot}-shot) ===")
        r = evaluate_vllm(llm, tokenizer, d, args.testset, args.exemplar_set,
                          args.num_fewshot, args.max_samples, args.max_new_tokens,
                          args.seed, save_all_samples=args.save_all_samples,
                          repetition_penalty=args.repetition_penalty,
                          add_bos=args.add_bos)
        print(f"  BLEU = {r['bleu']:.2f}  ({r['n_samples']} samples, {r['time_s']:.0f}s)")
        for s in r["samples"][:3]:
            print(f"    src: {s['src'][:80]}")
            print(f"    hyp: {s['hyp'][:80]}")
            print(f"    ref: {s['ref'][:80]}")
        out["results"][d] = r

    if args.compute_comet:
        print("\nFreeing vLLM model before loading COMET...")
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        for d in directions:
            r = out["results"][d]
            sources = r.pop("_sources")
            references = r.pop("_references")
            translations = r.get("translations") or [s["hyp"] for s in r["samples"]]
            if r.get("translations") is None:
                print(f"  WARN: {d} missing full translations (use --save_all_samples)")
                continue
            print(f"\n=== COMET {d} ===")
            t0 = time.time()
            system_score, per_sample = compute_comet(
                sources, translations, references, args.comet_model_path)
            print(f"  COMET = {system_score:.4f}  (time {time.time()-t0:.0f}s)")
            r["comet"] = system_score
            r["comet_per_sample"] = per_sample
    else:
        for d in directions:
            out["results"][d].pop("_sources", None)
            out["results"][d].pop("_references", None)

    print("\n=== Summary ===")
    for d, r in out["results"].items():
        line = f"  {d}: BLEU = {r['bleu']:.2f}"
        if "comet" in r:
            line += f"  COMET = {r['comet']:.4f}"
        print(line)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
