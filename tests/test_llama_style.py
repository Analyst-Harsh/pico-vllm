"""Llama-style changes: RMSNorm, and ModernGPT built from GQA blocks."""

import torch
from torch import nn

from pico_vllm.model.attention import MultiHeadAttention
from pico_vllm.model.gpt import GPT
from pico_vllm.model.llama_style import ModernGPT, RMSNorm

B, T, C, NH, CTX = 2, 6, 16, 4, 8
HS = C // NH


def count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---- RMSNorm --------------------------------------------------------------------------------


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


def test_rmsnorm_half_the_params_of_layernorm() -> None:
    assert count(RMSNorm(C)) == C  # weight only
    assert count(nn.LayerNorm(C)) == 2 * C  # weight + bias


# ---- ModernGPT ------------------------------------------------------------------------------


def test_modern_gpt_matches_gpt_shape_and_params() -> None:
    args = {"vocab_size": 65, "n_embd": C, "n_head": NH, "n_layer": 2, "ctx_len": CTX}
    gpt, modern = GPT(**args), ModernGPT(**args)
    assert count(modern) == count(gpt)
    assert not any(isinstance(m, MultiHeadAttention) for m in modern.modules())
    logits, _ = modern(torch.zeros((B, T), dtype=torch.long))
    assert logits.shape == (B, T, 65)


def test_modern_gpt_rmsnorm_replaces_every_layernorm() -> None:
    args = {"vocab_size": 65, "n_embd": C, "n_head": NH, "n_layer": 2, "ctx_len": CTX}
    gpt, rms = GPT(**args), ModernGPT(**args, rmsnorm=True)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in rms.modules())
    assert count(gpt) - count(rms) == (2 * 2 + 1) * C  # 5 norms lose their bias


def test_modern_gpt_n_kv_head_shrinks_every_layer() -> None:
    args = {"vocab_size": 65, "n_embd": C, "n_head": NH, "n_layer": 2, "ctx_len": CTX}
    mha, gqa = ModernGPT(**args), ModernGPT(**args, n_kv_head=2)
    assert count(mha) - count(gqa) == 2 * (2 * (NH - 2) * HS * C)  # 2 layers
    assert all(b.sa.n_kv_head == 2 for b in gqa.blocks)
