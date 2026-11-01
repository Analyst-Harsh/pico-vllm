"""F5 step 1a target: MultiHeadAttention in src/pico_vllm/model.py."""

import pytest
import torch

from pico_vllm.model import MultiHeadAttention

B, T, C, NH, CTX = 2, 6, 16, 4, 8


@pytest.fixture
def mha() -> MultiHeadAttention:
    torch.manual_seed(1337)
    return MultiHeadAttention(n_embd=C, n_head=NH, ctx_len=CTX).eval()


def test_output_shape(mha: MultiHeadAttention) -> None:
    assert mha(torch.randn(B, T, C)).shape == (B, T, C)


def test_heads_split_the_width(mha: MultiHeadAttention) -> None:
    assert len(mha.heads) == NH
    assert all(h.key.out_features == C // NH for h in mha.heads)


def test_is_concat_of_heads_then_proj(mha: MultiHeadAttention) -> None:
    x = torch.randn(B, T, C)
    expected = mha.proj(torch.cat([h(x) for h in mha.heads], dim=-1))
    assert torch.allclose(mha(x), expected)


def test_causal(mha: MultiHeadAttention) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(mha(x)[:, :3], mha(x2)[:, :3])


def test_rejects_width_not_divisible_by_heads() -> None:
    with pytest.raises((AssertionError, ValueError)):
        MultiHeadAttention(n_embd=10, n_head=4, ctx_len=CTX)
