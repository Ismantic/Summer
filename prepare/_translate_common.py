"""
Few-shot WMT translation BLEU for *base* (pretrained-only) models.

Uses completion-style prompts because base models haven't been chat-tuned:

    English: We meet at the cafe at 3 PM.
    Chinese: 我们下午三点在咖啡馆见面。

    English: The library closes at 9.
    Chinese: 图书馆九点关门。
    ...
    English: <test sentence>
    Chinese:

The model continues; we stop generation at the next "English:" / newline.
Few-shot exemplars come from a deterministic shuffle of the *previous year's*
test set (newstest2021 by default), so test/exemplar splits never overlap.

BLEU via sacrebleu (zh-tokenize for en->zh, 13a for zh->en).

Usage:
    python evals/eval_pretrain_translate.py \
        --model_path output/phase2_ckpt_v18_tie \
        --testset wmt22 --direction both --num_fewshot 5 \
        --max_samples 200 --batch_size 4 \
        --output_path eval_results/phase1_v0_translate/wmt22.json
"""
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
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from prepare.tokenizer import PieceTokenizerWrapper
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


def encode_prompts(tokenizer, prompts):
    """Returns list[list[int]] of token id lists, no special tokens added."""
    out = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        out.append(ids)
    return out


def left_pad(token_lists, pad_id, device):
    max_len = max(len(t) for t in token_lists)
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(t)) + t for t in token_lists],
        device=device, dtype=torch.long,
    )
    attn = input_ids.ne(pad_id)
    return input_ids, attn


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


def evaluate(model, tokenizer, direction, testset, exemplar_set, num_fewshot,
             max_samples, batch_size, max_new_tokens, seed, save_all_samples=False):
    sources, references = load_pair(testset, direction)
    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]
    exemplars = pick_exemplars(direction, num_fewshot, seed=seed,
                                exemplar_set=exemplar_set) if num_fewshot > 0 else []
    prompts = [build_prompt(direction, exemplars, s) for s in sources]

    pad_id = tokenizer.pad_token_id
    translations = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_ids = encode_prompts(tokenizer, batch_prompts)
        input_ids, attn = left_pad(batch_ids, pad_id, model.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
            )
        for j, row in enumerate(out):
            gen_ids = row[input_ids.shape[1]:].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            translations.append(trim_continuation(text, SRC_LABEL[direction]))
        if ((i // batch_size + 1) % 5) == 0:
            done = min(i + batch_size, len(prompts))
            print(f"  [{direction}] {done}/{len(prompts)} | {time.time()-t0:.0f}s")

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
        "time_s": time.time() - t0,
        "translations": translations if save_all_samples else None,
        "samples": [
            dict(src=sources[k], hyp=translations[k], ref=references[k])
            for k in range(n_save)
        ],
        "_sources": sources,
        "_references": references,
    }


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--testset", default="wmt22",
                   choices=["wmt17", "wmt18", "wmt19", "wmt20", "wmt21", "wmt22", "wmt23"])
    p.add_argument("--exemplar_set", default="wmt21",
                   help="Test set used as exemplar pool (must differ from --testset)")
    p.add_argument("--direction", default="both", choices=["zh-en", "en-zh", "both"])
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", default=None)
    p.add_argument("--compute_comet", action="store_true",
                   help="Compute COMET score (Unbabel/wmt22-comet-da) in addition to BLEU.")
    p.add_argument("--comet_model_path", default=_default_comet(),
                   help="Local snapshot of Unbabel/wmt22-comet-da")
    p.add_argument("--save_all_samples", action="store_true",
                   help="Save all translation samples in output JSON (else first 5).")
    args = p.parse_args()
    if args.exemplar_set == args.testset:
        raise ValueError("--exemplar_set must differ from --testset to avoid leakage")

    print(f"Loading model from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    out = {"model_path": args.model_path, "results": {}}
    for d in directions:
        print(f"\n=== {d} (testset={args.testset}, {args.num_fewshot}-shot) ===")
        r = evaluate(model, tokenizer, d, args.testset, args.exemplar_set,
                     args.num_fewshot, args.max_samples, args.batch_size,
                     args.max_new_tokens, args.seed,
                     save_all_samples=args.save_all_samples)
        print(f"  BLEU = {r['bleu']:.2f}  ({r['n_samples']} samples, {r['time_s']:.0f}s)")
        for s in r["samples"][:3]:
            print(f"    src: {s['src'][:80]}")
            print(f"    hyp: {s['hyp'][:80]}")
            print(f"    ref: {s['ref'][:80]}")
        out["results"][d] = r

    # COMET: compute AFTER generation model fully unloaded GPU (free for XLM-R loading)
    if args.compute_comet:
        print("\nFreeing translation model GPU mem before loading COMET...")
        del model
        import gc, torch as _torch
        gc.collect(); _torch.cuda.empty_cache()
        for d in directions:
            r = out["results"][d]
            sources = r.pop("_sources")
            references = r.pop("_references")
            translations = r.get("translations") or [s["hyp"] for s in r["samples"]]
            if r.get("translations") is None:
                print(f"  WARN: {d} missing full translations (use --save_all_samples for COMET on full set)")
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
