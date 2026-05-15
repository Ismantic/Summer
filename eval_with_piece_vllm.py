"""vLLM-backed lm-evaluation-harness wrapper with piece_tokenizer.

Drop-in replacement for eval_with_piece.py that uses vLLM PagedAttention
instead of transformers.generate. Provides ~5-10x speedup on loglikelihood
tasks (hellaswag, mmlu) by exploiting vLLM's batched continuous decoding.

Algorithm mirrors lm_eval.models.vllm_causallms.VLLM._loglikelihood_tokens:
1. Concat (context + continuation) token IDs
2. Run vLLM with prompt_logprobs=1, max_tokens=1
3. For each continuation token, fetch its logprob from prompt_logprobs[ctxlen:]
4. Sum → loglikelihood; check if argmax == continuation → is_greedy

Decoupled from lm_eval.models.vllm_causallms (which depends on ray + HF
AutoTokenizer); we register a fresh TemplateLM subclass that wires
piece_tokenizer encode/decode through vLLM's skip_tokenizer_init mode.

CLI mirrors eval_with_piece.py:
    python eval_with_piece_vllm.py --model_path ./HY-MT1.5-1.8B-new-tok \
        --task piqa --num_fewshot 5 --output_path ./eval_results/piqa
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub ray to avoid lm_eval import chain pulling it in. We don't use data
# parallelism (data_parallel_size=1 path doesn't actually call ray.*).
if "ray" not in sys.modules:
    sys.modules["ray"] = type(sys)("ray")

from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator

from tokenizer_wrapper import PieceTokenizerWrapper


def _load_tokenizer(model_path: str):
    """Auto-detect: piece.model present → piece tokenizer; else → HF AutoTokenizer.
    Returns object with encode/decode + bos/eos/pad token ids."""
    if os.path.exists(os.path.join(model_path, "piece.model")):
        return PieceTokenizerWrapper(model_path)

    # Fall back to HF tokenizer (e.g., for base Qwen3-0.6B-Base)
    from transformers import AutoTokenizer

    class _HFAdapter:
        def __init__(self, path):
            self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.bos_token_id = self._tok.bos_token_id
            self.eos_token_id = self._tok.eos_token_id
            self.pad_token_id = self._tok.pad_token_id or self._tok.eos_token_id
            self.vocab_size = self._tok.vocab_size

        def encode(self, text, add_special_tokens=False):
            return self._tok.encode(text, add_special_tokens=add_special_tokens)

        def decode(self, ids, skip_special_tokens=True):
            return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    return _HFAdapter(model_path)


@register_model("vllm_piece")
class VLLMPiece(TemplateLM):
    """Minimal TemplateLM that uses vLLM for generation/loglikelihood,
    with piece_tokenizer for encode/decode."""

    def __init__(
        self,
        pretrained: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        batch_size="auto",
        seed: int = 1234,
        **kwargs,
    ):
        super().__init__()
        self._tokenizer = _load_tokenizer(pretrained)
        self._max_length = int(max_model_len)
        self._batch_size = batch_size if batch_size == "auto" else int(batch_size)
        self.backend = "causal"

        from vllm import LLM
        self.model = LLM(
            model=pretrained,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization),
            skip_tokenizer_init=True,
            trust_remote_code=True,
            max_model_len=self._max_length,
            seed=int(seed),
        )

    # ---- required abstract properties ----
    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        bos = self._tokenizer.bos_token_id
        return bos if bos is not None else self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    # ---- tokenization ----
    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        if isinstance(string, list):
            return [self.tok_encode(s, add_special_tokens=add_special_tokens) for s in string]
        ids = self._tokenizer.encode(string, add_special_tokens=False)
        if add_special_tokens:
            ids = [self.prefix_token_id] + ids
        return ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, int):
            tokens = [tokens]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # ---- loglikelihood ----
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        max_ctx_len = self.max_length - 1  # vLLM needs ≥1 generation slot

        # Sort longest-first for better vLLM batching
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        # n=0 → one big chunk; vLLM handles continuous batching internally
        chunks = re_ord.get_batched(n=0, batch_fn=None)

        sampling = SamplingParams(
            temperature=0.0,
            prompt_logprobs=1,
            max_tokens=1,
            detokenize=False,
        )

        res = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="loglikelihood (vLLM)")
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for _, context_enc, continuation_enc in chunk:
                full = context_enc + continuation_enc
                # Left-truncate if too long, preserving the continuation
                inp = full[-max_ctx_len:]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - max_ctx_len
                )
                inputs.append(inp)
                ctxlens.append(ctxlen)

            prompts = [TokensPrompt(prompt_token_ids=inp) for inp in inputs]
            outputs = self.model.generate(
                prompts, sampling_params=sampling, use_tqdm=False
            )

            for output, ctxlen, inp in zip(outputs, ctxlens, inputs):
                res.append(self._parse_logprobs(inp, output, ctxlen))
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens, output, ctxlen):
        # prompt_logprobs is a list[dict|None] aligned with tokens (prefix=None)
        cont = output.prompt_logprobs

        def _to_float(lp):
            return getattr(lp, "logprob", lp)

        cont = [
            {tk: _to_float(lp) for tk, lp in d.items()} if d is not None else None
            for d in cont
        ]
        total = sum(
            d.get(t)
            for t, d in zip(tokens[ctxlen:], cont[ctxlen:])
        )
        is_greedy = True
        for t, d in zip(tokens[ctxlen:], cont[ctxlen:]):
            if d:
                top = max(d, key=d.get)
                if top != t:
                    is_greedy = False
                    break
        return total, is_greedy

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        # Not needed for our mono benchmarks. Implement only if asked.
        raise NotImplementedError(
            "loglikelihood_rolling not implemented for VLLMPiece (no current task needs it)"
        )

    # ---- generation ----
    def generate_until(self, requests, disable_tqdm=False):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        res = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="generate_until (vLLM)")

        # vLLM batches well — collect all then dispatch in one call per gen_kwargs
        # signature. Most tasks use the same gen_kwargs across requests.
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_tokens = int(gen_kwargs.get("max_gen_toks", 256))

            ctx_enc = self.tok_encode(context)[-self.max_length + max_tokens:]
            sampling = SamplingParams(
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                top_p=float(gen_kwargs.get("top_p", 1.0)),
                max_tokens=max_tokens,
                stop=until if until else None,
                stop_token_ids=[self.eot_token_id],
            )
            output = self.model.generate(
                [TokensPrompt(prompt_token_ids=ctx_enc)],
                sampling_params=sampling,
                use_tqdm=False,
            )
            gen_ids = list(output[0].outputs[0].token_ids)
            text = self.tok_decode(gen_ids, skip_special_tokens=True)
            # Belt-and-braces: also slice at first stop seq in case vLLM
            # doesn't catch a multi-token stop string
            for stop in until:
                if stop in text:
                    text = text.split(stop)[0]
            res.append(text)
            pbar.update(1)
        pbar.close()
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="vllm_piece",
        model_args=(
            f"pretrained={args.model_path},"
            f"dtype=bfloat16,"
            f"max_model_len={args.max_model_len},"
            f"gpu_memory_utilization={args.gpu_memory_utilization}"
        ),
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    out = {
        "task": args.task,
        "num_fewshot": args.num_fewshot,
        "results": results["results"],
    }
    print(json.dumps(out, indent=2, default=str))

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
