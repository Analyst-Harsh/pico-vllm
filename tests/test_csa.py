"""F6 step 1 target: CausalSelfAttention (batched heads) in src/pico_vllm/modern.py."""

import pytest
import torch

from pico_vllm.model import GPT, MultiHeadAttention
from pico_vllm.modern import CausalSelfAttention, ModernGPT

B, T, C, NH, CTX = 2, 6, 16, 4, 8


@pytest.fixture
def pair() -> tuple[MultiHeadAttention, CausalSelfAttention]:
    """A looped MHA and a batched CSA holding exactly the same weights."""
    torch.manual_seed(1337)
    mha = MultiHeadAttention(C, NH, CTX).eval()
    csa = CausalSelfAttention(C, NH, CTX).eval()
    with torch.no_grad():
        q = torch.cat([h.query.weight for h in mha.heads])  # (C, C): head 0 rows, head 1 rows…
        k = torch.cat([h.key.weight for h in mha.heads])
        v = torch.cat([h.value.weight for h in mha.heads])
        csa.c_attn.weight.copy_(torch.cat([q, k, v]))  # (3C, C) = [q | k | v]
        csa.proj.load_state_dict(mha.proj.state_dict())
    return mha, csa


def test_output_shape(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    _, csa = pair
    assert csa(torch.randn(B, T, C)).shape == (B, T, C)


def test_same_output_as_looped_heads(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    mha, csa = pair
    x = torch.randn(B, T, C)
    assert torch.allclose(csa(x), mha(x), atol=1e-6)


def test_same_param_count(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    mha, csa = pair
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(csa) == count(mha)


def test_causal(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    _, csa = pair
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(csa(x)[:, :3], csa(x2)[:, :3])


def test_modern_gpt_matches_gpt_shape_and_params() -> None:
    args = {"vocab_size": 65, "n_embd": C, "n_head": NH, "n_layer": 2, "ctx_len": CTX}
    gpt, modern = GPT(**args), ModernGPT(**args)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(modern) == count(gpt)
    assert not any(isinstance(m, MultiHeadAttention) for m in modern.modules())
    logits, _ = modern(torch.zeros((B, T), dtype=torch.long))
    assert logits.shape == (B, T, 65)
