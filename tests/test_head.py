"""F4 step 3 target: Head (one head of causal self-attention) in src/pico_vllm/model.py."""

import pytest
import torch
from torch.nn import functional as F

from pico_vllm.model.attention import Head

B, T, C, HS, CTX = 2, 6, 16, 4, 8


@pytest.fixture
def head() -> Head:
    torch.manual_seed(1337)
    return Head(n_embd=C, head_size=HS, ctx_len=CTX).eval()


def test_output_shape(head: Head) -> None:
    x = torch.randn(B, T, C)
    assert head(x).shape == (B, T, HS)


def test_causal_future_does_not_change_past(head: Head) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)  # change tokens 3.. only
    out, out2 = head(x), head(x2)
    assert torch.allclose(out[:, :3], out2[:, :3])  # positions 0..2 never saw the future
    assert not torch.allclose(out[:, 3:], out2[:, 3:])


def test_matches_reference_attention(head: Head, monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.randn(B, T, C)
    q, k, v = head.query(x), head.key(x), head.value(x)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def banned(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("write attention yourself; don't call the reference")

    monkeypatch.setattr(F, "scaled_dot_product_attention", banned)
    assert torch.allclose(head(x), expected, atol=1e-6)
