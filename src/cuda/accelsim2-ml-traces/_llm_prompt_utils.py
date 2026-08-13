"""Shared prompt construction for vLLM-based LLM trace drivers.

Produces a token-ID list of EXACT length so prefill size is reproducible
across tokenizers (Llama BPE, Mixtral SentencePiece, Qwen BPE, ...) and
matches the number declared in the workload's args line.

Feed the returned list to vLLM via {"prompt_token_ids": ids} (or
TokensPrompt) so vLLM does not re-tokenize or auto-insert BOS.
"""
from typing import List

BASE_PROMPT = (
    "Explain in detail the theory of general relativity, "
    "including the mathematical framework of tensor calculus, "
    "the equivalence principle, geodesics in curved spacetime, "
    "the Einstein field equations, and their solutions such as "
    "the Schwarzschild metric and Friedmann equations. "
    "Discuss the experimental evidence supporting general relativity, "
    "including gravitational lensing, frame dragging, and gravitational waves. "
)


def build_exact_prompt_ids(tokenizer, prompt_tokens: int) -> List[int]:
    """Return a list of token IDs of length exactly ``prompt_tokens``.

    BOS is counted in ``prompt_tokens`` so the value passed in matches
    what the model actually sees at prefill time.
    """
    body_ids = tokenizer.encode(BASE_PROMPT, add_special_tokens=False)
    if not body_ids:
        raise RuntimeError("Base prompt tokenized to zero tokens.")

    bos_id = getattr(tokenizer, "bos_token_id", None)
    bos = [bos_id] if bos_id is not None else []

    target = prompt_tokens - len(bos)
    if target <= 0:
        raise ValueError(
            f"prompt_tokens={prompt_tokens} is too small (BOS={len(bos)})"
        )

    reps = (target + len(body_ids) - 1) // len(body_ids)
    ids = bos + (body_ids * reps)[:target]

    if len(ids) != prompt_tokens:
        raise AssertionError(
            f"built {len(ids)} tokens, expected {prompt_tokens}"
        )
    return ids
