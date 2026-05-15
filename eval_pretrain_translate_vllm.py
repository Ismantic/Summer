"""vLLM-based translation eval, drop-in replacement for eval_pretrain_translate.py.

Uses vLLM's independent generation engine (PagedAttention) instead of
transformers.generate() — decouples eval from transformers version instability.

Tokenization still uses our piece_tokenizer (we pass prompt_token_ids directly).
COMET evaluation reuses the same code path as the transformers version.

Usage (mirrors eval_pretrain_translate.py):
    python eval_pretrain_translate_vllm.py \\
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

# Reuse pieces from the transformers eval script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pretrain_translate import (
    load_tokenizer,
    load_pair,
    pick_exemplars,
    build_prompt,
    encode_prompts,
    trim_continuation,
    compute_comet,
    SRC_LABEL,
)


def evaluate_vllm(llm, tokenizer, direction, testset, exemplar_set, num_fewshot,
                  max_samples, max_new_tokens, seed, save_all_samples=False):
    from vllm import SamplingParams
    sources, references = load_pair(testset, direction)
    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]
    exemplars = pick_exemplars(direction, num_fewshot, seed=seed,
                                exemplar_set=exemplar_set) if num_fewshot > 0 else []
    prompts = [build_prompt(direction, exemplars, s) for s in sources]
    prompt_token_ids = encode_prompts(tokenizer, prompts)

    sampling = SamplingParams(
        temperature=0.0,                     # greedy
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
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
    p.add_argument("--comet_model_path", default="/mnt/data/Summer-data/comet-wmt22-da")
    p.add_argument("--save_all_samples", action="store_true")
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
                          args.seed, save_all_samples=args.save_all_samples)
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
