"""Pretty-print eval_results/full/<tag>/ into a comparison table.

Usage:
    python summarize_full.py base phase1_ckpt_v8 phase2_ckpt_v8_s2_aurora

For each tag, reads the standard eval_full.sh artifacts and prints
acc / acc_norm / BLEU / MMLU / PPL with stderr. base's mmlu and mono
tasks are looked up from the nested lm-eval output dir layout.
"""
import argparse
import glob
import json
import os
import re

FULL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results", "full")

MONO_TASKS = ["lambada_openai", "piqa", "arc_challenge", "hellaswag"]
PREFER_METRIC = ["acc_norm,none", "acc,none"]


def read_lmeval_json(tag, task):
    """Returns (value, stderr, metric_name) or (None, None, None)."""
    if tag == "base":
        # lm-eval CLI writes to <out>/<task>/<basename(model)>/results_*.json
        pattern = os.path.join(FULL, "base", task, "__home*", "results_*.json")
    else:
        # eval_with_piece writes directly to <out>/<task>/result.json
        pattern = os.path.join(FULL, tag, task, "result.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None, None
    with open(files[-1]) as f:
        d = json.load(f)
    if task not in d.get("results", {}):
        return None, None, None
    r = d["results"][task]
    for m in PREFER_METRIC:
        if m in r:
            sk = m.replace(",none", "_stderr,none")
            return r[m], r.get(sk, 0.0), m.split(",")[0]
    return None, None, None


def read_mmlu(tag):
    """Returns (acc, stderr) or (None, None)."""
    if tag == "base":
        pattern = os.path.join(FULL, "base", "mmlu", "__home*", "results_*.json")
    else:
        pattern = os.path.join(FULL, tag, "mmlu", "result.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    with open(files[-1]) as f:
        d = json.load(f)
    m = d.get("results", {}).get("mmlu", {})
    return m.get("acc,none"), m.get("acc_stderr,none")


def read_bleu(tag):
    """Returns (zh_en, en_zh) or (None, None)."""
    f = os.path.join(FULL, tag, "wmt22.json")
    if not os.path.isfile(f):
        # try the log
        flog = os.path.join(FULL, tag, "wmt22.log")
        if os.path.isfile(flog):
            with open(flog) as fp:
                txt = fp.read()
            m1 = re.search(r"zh-en: BLEU = ([\d.]+)", txt)
            m2 = re.search(r"en-zh: BLEU = ([\d.]+)", txt)
            return (float(m1.group(1)) if m1 else None,
                    float(m2.group(1)) if m2 else None)
        return None, None
    with open(f) as fp:
        d = json.load(fp)
    return d.get("zh-en", {}).get("BLEU"), d.get("en-zh", {}).get("BLEU")


def read_ppl(tag):
    f = os.path.join(FULL, tag, "ppl.json")
    if not os.path.isfile(f):
        return None
    with open(f) as fp:
        d = json.load(fp)
    return d.get("ppl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    args = p.parse_args()
    tags = args.tags

    print(f"{'metric':<22} " + " ".join(f"{t:>22}" for t in tags))
    print("-" * (22 + 23 * len(tags)))

    for task in MONO_TASKS:
        vals = [read_lmeval_json(t, task) for t in tags]
        row = f"{task:<22} "
        for v, se, met in vals:
            cell = f"{v:.4f}±{se:.4f}" if v is not None else "—"
            row += f" {cell:>22}"
        print(row)

    mmlu_vals = [read_mmlu(t) for t in tags]
    row = f"{'mmlu':<22} "
    for v, se in mmlu_vals:
        cell = f"{v:.4f}±{se:.4f}" if v is not None else "—"
        row += f" {cell:>22}"
    print(row)

    bleu_zh_vals = [read_bleu(t)[0] for t in tags]
    bleu_en_vals = [read_bleu(t)[1] for t in tags]
    row = f"{'BLEU zh-en':<22} "
    for v in bleu_zh_vals:
        cell = f"{v:.2f}" if v is not None else "—"
        row += f" {cell:>22}"
    print(row)
    row = f"{'BLEU en-zh':<22} "
    for v in bleu_en_vals:
        cell = f"{v:.2f}" if v is not None else "—"
        row += f" {cell:>22}"
    print(row)

    ppl_vals = [read_ppl(t) for t in tags]
    row = f"{'valid PPL':<22} "
    for v in ppl_vals:
        cell = f"{v:.2f}" if v is not None else "—"
        row += f" {cell:>22}"
    print(row)


if __name__ == "__main__":
    main()
