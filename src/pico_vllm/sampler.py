"""What this file teaches: how logits become the next token id.

Key idea: divide by temperature (<1 sharpens, >1 flattens), keep only the top_k scores, softmax,
then draw one id. temperature 0 is greedy argmax. The engine calls this once per step for
the whole running batch.

Read next: model/llama_style.py.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def next_token_probs(logits: Tensor, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
    """logits (B, V) -> probs (B, V). Each row sums to 1; zero outside the top_k.

    temperature must be > 0 here; temperature == 0 (greedy) is handled in sample_next.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 here, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    logits = logits / temperature  # <1 sharpens, >1 flattens
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]  # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return F.softmax(logits, dim=-1)


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """logits (B, V) -> next ids (B, 1).

    temperature == 0 means greedy: always the highest-scoring token, no randomness.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = next_token_probs(logits, temperature, top_k)
    return torch.multinomial(probs, num_samples=1, generator=generator)
