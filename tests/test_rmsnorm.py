"""F6 step 2 target: RMSNorm in src/pico_vllm/modern.py."""

import torch
from torch import nn

from pico_vllm.modern import RMSNorm

B, T, C = 2, 6, 16


def test_rms_is_one_per_token() -> None:
    torch.manual_seed(1337)
    out = RMSNorm(C)(torch.randn(B, T, C) * 5 + 3)  # big, off-centre input
    rms = out.pow(2).mean(dim=-1).sqrt()  # (B, T)
    assert torch.allclose(rms, torch.ones(B, T), atol=1e-4)


def test_matches_torch_rmsnorm() -> None:
    torch.manual_seed(1337)
    ours, ref = RMSNorm(C), nn.RMSNorm(C, eps=1e-6)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(C))  # a trained-looking weight, not all ones
        ref.weight.copy_(ours.weight)
    x = torch.randn(B, T, C)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_half_the_params_of_layernorm() -> None:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(RMSNorm(C)) == C  # weight only
    assert count(nn.LayerNorm(C)) == 2 * C  # weight + bias
