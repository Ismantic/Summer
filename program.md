# Summer — autonomous research program

Adapted from karpathy/autoresearch. Run this loop to autonomously iterate
on the Summer tokenizer-swap pretraining pipeline.

## Project context (read first)

Summer reproduces the ReTok methodology on `Qwen3-0.6B-Base`: swap
HuggingFace BBPE (~151K vocab) for a piece tokenizer (~65K) and recover
performance with continued pretraining. Two stages:

- **Phase 1**: freeze transformer, train embed/lm_head on diverse mix
- **Phase 2**: unfreeze, joint train all params (Muon for 2D, Adam for embed)

Current production checkpoint: `output/phase1_ckpt_v8` — Phase 1 only,
1B training tokens on 2B-pool diverse mix. Mono-LM benchmarks all
within ±10% of base (5/5 strict, 6/6 incl. ARC at boundary). BLEU
zh-en just -2.5% of base.

The hard problem now: **Phase 2 has been a no-op or destructive in every
prior attempt**. Goal of autonomous iteration is to find a Phase 2
recipe that adds real downstream gain without destroying transformer
priors.

## Setup (do once at start of session)

1. **Confirm baselines exist**:
   - `output/phase1_ckpt_v8` (Phase 1 ckpt — fixed start point)
   - `output/phase1_train_512_v7.pt` or `_v8.pt` (training data)
   - `output/valid_512.pt` (held-out valid set)
   - `/home/tfbao/new/Qwen3-0.6B-Base` (base model for eval)
2. **Read in-scope files**:
   - `CLAUDE.md` — project layout
   - `finetune_muon.py` — training loop (the file you'll touch most)
   - `muon.py` — Muon optimizer
   - `aurora.py` — Aurora optimizer (drop-in for Muon)
   - `pretokenize_v7.py` — data sampler (only touch if doing v9+ Phase 1)
   - `eval_pretrain_translate.py`, `eval_ppl.py`, `smoke_test.sh` — eval
3. **GPU check**: `nvidia-smi` — both A6000 must be free
4. **Initialize `experiments.tsv`** if it doesn't exist (see Logging below)
5. **Pick an experiment branch**: `git checkout -b autoresearch/<tag>` from current main

## Experimentation scope

**You CAN modify:**
- `finetune_muon.py` — training loop, optimizer flags
- `muon.py`, `aurora.py` — optimizer internals
- `run_v8_s2_*.sh` — hyperparameter scripts (one per experiment)
- `pretokenize_v7.py` SOURCES list & weights — for v9+ Phase 1 data mix changes
- New scripts for new ideas (e.g. KL-regularized Phase 2)

**You CANNOT modify:**
- `eval_pretrain_translate.py`, `eval_ppl.py`, `smoke_test.sh`, `eval_with_piece.py` — eval harnesses are ground truth
- Base model paths, test data paths
- The token mapping in `replace_tokenizer.py` (would invalidate ckpts)

**The default experiment unit: Phase 2 from `output/phase1_ckpt_v8`**
- 500 steps × eff_bs=128 × seq=512 = 32M training tokens
- Train ≈ 25 min, eval ≈ 30 min (smoke + BLEU + PPL), MMLU full ≈ 30 min
- Total ~85 min per experiment
- Skip MMLU full for clearly-failed runs (BLEU dropped or loss diverged); only run for promising candidates

## Default recipe (use unless empirically proven worse)

For any Phase 2 or from-scratch transformer training where MLPs are unfrozen
and trained ≥500 steps, **use Aurora as the default optimizer**, not vanilla
Muon. Rationale:

- 6% compute overhead is dominated by Aurora's protection against neuron
  death in tall matrices (Aurora paper, Tilde Research, 2026)
- Short-run ablations (v8_s2_aurora vs v8_s2_muon5e5 at 500 steps) showed
  parity, not Muon advantage — and Aurora's advantage accumulates with step
  count, so this is asymmetric risk-return
- Qwen3-0.6B-Base has MLP expansion 5.83× (in Aurora's sweet spot per
  Figure: monotonic gain with expansion factor)
- Caveats: MoE architectures benefit less (smaller per-expert m/n); for
  square-only matrix groups Aurora reduces exactly to Muon, so neutral

Concretely: `--use_aurora` should be the default in all new run scripts
until/unless an ablation in the same regime shows Muon strictly better.
Moonshot's per-param LR scaling (`--moonshot_scaling`) is orthogonal —
evaluate independently and stack if it adds value.

## Objective

Primary score (for keep/discard decisions):
```
score = 0.5 * (BLEU_zh_en + BLEU_en_zh) / base_BLEU_avg
        + 0.5 * (MMLU_full / base_MMLU)
```
where `base_BLEU_avg = (15.40 + 33.00) / 2 = 24.20` and `base_MMLU = 0.5244`.
Higher is better. Baseline (v8 Phase 1 only): ~0.91.

A run **wins** if its score is ≥ current best by ≥ 0.005 (≈ a real
non-noise improvement). Anything below `v7_s2_v2 baseline score - 0.01`
should be discarded.

Constraint: never destroy mono-LM beyond -15% on any single benchmark
(any such result is auto-discard regardless of BLEU gain).

## The experiment loop

```
LOOP FOREVER:
  1. Inspect git state and experiments.tsv. Identify best result so far.
  2. Form a hypothesis (one variable at a time when possible). Examples:
     - "Aurora at muon_lr=3e-5 instead of 5e-5 — sweet spot search"
     - "Add KL penalty vs base model to Phase 2 loss"
     - "Phase 2 with v7-data-shuffle to break narrow refinement"
     - "Aurora pp_iterations=3 instead of 2"
  3. Implement: edit finetune_muon.py / aurora.py / run_v8_s2_*.sh.
     Commit with a clear one-line message: `expN: <what changed>`
  4. Launch in background:
       bash run_v8_s2_<tag>.sh > output/<tag>_run.log 2>&1
     (always background; do NOT block the conversation on the run)
  5. Wait for the background-job completion notification.
  6. Pull results: BLEU from translate_wmt22/<tag>.log,
                   PPL from ppl/<tag>.json,
                   MMLU full from mmlu_full/<tag>/result.json (if run)
  7. Compute score. Compare to current best.
  8. KEEP path: log to experiments.tsv with status=keep, advance to
     try-on-top-of-this in the next iteration.
     DISCARD path: log with status=discard, git reset to the prior best.
     CRASH path: read tail of run log, log status=crash with the 1-line
     reason, attempt a quick fix only if obvious, otherwise abandon.
  9. NEVER STOP. The user may be asleep. Continue until interrupted.
```

## What scripts must print

Every `run_v8_s2_<tag>.sh` should produce a tag-named eval artifact set:
```
eval_results/translate_wmt22/<tag>.log    -> contains "zh-en: BLEU = X"
                                                       "en-zh: BLEU = Y"
eval_results/ppl/<tag>.json               -> json with "ppl" field
eval_results/smoke_<tag>/new-tok/*.log    -> per-task smoke (200 limit)
eval_results/mmlu_full/<tag>/result.json  -> json with "mmlu" key
                                             (only for promising runs)
```

A helper one-liner to extract the score for a tag:
```bash
TAG=phase2_v8_s2_aurora
ZHEN=$(grep -oP "zh-en: BLEU = \K[0-9.]+" eval_results/translate_wmt22/$TAG.log)
ENZH=$(grep -oP "en-zh: BLEU = \K[0-9.]+" eval_results/translate_wmt22/$TAG.log)
MMLU=$(jq '.results.mmlu."acc,none"' eval_results/mmlu_full/${TAG#phase2_}/result.json 2>/dev/null)
python3 -c "
zhen, enzh, mmlu = $ZHEN, $ENZH, $MMLU
score = 0.5 * (zhen+enzh)/24.20 + 0.5 * mmlu/0.5244
print(f'score={score:.4f}  zhen={zhen:.2f} enzh={enzh:.2f} mmlu={mmlu:.4f}')
"
```

## Logging — experiments.tsv

Tab-separated, do NOT commit (.gitignore it or just leave untracked).
Header + 7 columns:

```
commit	tag	score	bleu_zh_en	bleu_en_zh	mmlu	status	description
```

| field | meaning |
|---|---|
| commit | 7-char git hash |
| tag | run tag (phase2_v8_s2_<name>) |
| score | aggregate (formula above) — `0.0000` for crash |
| bleu_zh_en | zh→en BLEU — `0` for crash |
| bleu_en_zh | en→zh BLEU — `0` for crash |
| mmlu | full MMLU acc (or smoke abstract_algebra × 1.5 estimate if MMLU full skipped) |
| status | `keep` / `discard` / `crash` |
| description | one-line; no commas (tabs allowed in description? no — keep ASCII single-line) |

Example:
```
commit	tag	score	bleu_zh_en	bleu_en_zh	mmlu	status	description
db96edc	v8_baseline	0.9105	15.01	28.49	0.4813	keep	v8 phase1 only baseline
abc1234	v8_s2_aurora_5e5	0.9180	15.20	29.10	0.4790	keep	aurora muon_lr=5e-5 500 steps
def5678	v8_s2_aurora_8e5	0.8200	12.50	22.10	0.4500	discard	aurora muon_lr=8e-5 too high
0000000	v8_s2_kl	0.0000	0	0	0	crash	KL penalty - shape mismatch
```

## Idea reservoir (when you run out of obvious moves)

Tier 1 — Cheap (one-knob changes from v8 Phase 2):
- Aurora LR sweep: muon_lr ∈ {2e-5, 3e-5, 5e-5, 8e-5}
- Aurora pp_iterations ∈ {1, 2, 3, 4}
- Aurora pp_beta ∈ {0.3, 0.5, 0.7, 1.0}
- Warmup steps: try 50 / 200 / 500 (vs default 100)
- Muon momentum: try 0.9 / 0.95 / 0.99
- Gradient clipping: try 0.5 / 1.0 / 2.0
- Adam LR for embed: try 1e-5 / 5e-5 / 1e-4

Tier 2 — Structural (small code changes):
- KL regularization against base model logits in Phase 2 loss
- Asymmetric LR: attention vs MLP groups separate Muon LRs
- Data refresh: re-pretokenize a 500M-token subset, train Phase 2 on the *new* split
- Periodic re-anchoring: every N steps, blend current params back toward base by α (catastrophic forgetting mitigation)
- Layer-wise LR decay (lower LR for shallower layers, where world knowledge lives)

Tier 3 — Architectural (only if Tier 1+2 plateau):
- Adapter / LoRA on transformer instead of full unfreeze
- Embed-only Phase 2 (skip transformer entirely, longer Adam train on embed+head with v8 ckpt)
- Larger Phase 1 (v9: 4B pool, 2B training tokens, no Phase 2 needed if it just keeps scaling)

## Anti-patterns to avoid

- Multi-variable changes — one knob at a time so we can attribute gains
- Hand-tweaking eval params to look good — never. Eval is ground truth.
- Skipping the discard step — if it's worse, reset.
- "Just one more try" infinite chains — stop after 3 failed attempts on same hypothesis class
- Touching Phase 1 ckpt — v8 is fixed throughout this loop

## Stop conditions

Only stop when:
- User explicitly interrupts ("stop", "pause")
- Disk fills (< 10GB free in `/`)
- All Tier 1+2 ideas exhausted AND best score hasn't improved in 5 runs

Otherwise: **NEVER STOP**. The user is probably asleep.
