"""F5 step 1d: GPT in src/pico_vllm/model.py."""

import math

import pytest
import torch

from pico_vllm.model import GPT

V, C, NH, NL, CTX, B = 11, 16, 4, 2, 8, 2


@pytest.fixture
def gpt() -> GPT:
    torch.manual_seed(1337)
    return GPT(V, n_embd=C, n_head=NH, n_layer=NL, ctx_len=CTX).eval()


def test_logits_shape_and_finite_loss(gpt: GPT) -> None:
    idx = torch.randint(V, (B, CTX))
    logits, loss = gpt(idx, idx)
    assert logits.shape == (B, CTX, V)
    assert torch.isfinite(loss)


def test_step0_loss_is_near_uniform(gpt: GPT) -> None:
    # small init -> near-equal logits -> loss ~ ln(V)
    idx = torch.randint(V, (B, CTX))
    _, loss = gpt(idx, torch.randint(V, (B, CTX)))
    assert abs(loss.item() - math.log(V)) < 0.1


def test_causal(gpt: GPT) -> None:
    idx = torch.randint(V, (B, CTX))
    idx2 = idx.clone()
    idx2[:, 5:] = (idx2[:, 5:] + 1) % V
    assert torch.allclose(gpt(idx)[0][:, :5], gpt(idx2)[0][:, :5])


def test_param_count_matches_formula(gpt: GPT) -> None:
    per_block = 12 * C * C + 10 * C  # 2 LayerNorms + attention + feed-forward
    expected = V * C + CTX * C + NL * per_block + 2 * C + (C * V + V)
    assert sum(p.numel() for p in gpt.parameters()) == expected


def test_generate_past_ctx_len(gpt: GPT) -> None:
    out = gpt.generate(torch.zeros((B, 1), dtype=torch.long), max_new_tokens=2 * CTX)
    assert out.shape == (B, 1 + 2 * CTX)


def test_generate_greedy_is_deterministic(gpt: GPT) -> None:
    idx = torch.zeros((B, 1), dtype=torch.long)
    a = gpt.generate(idx, max_new_tokens=10, temperature=0.0)
    b = gpt.generate(idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_respects_top_k(gpt: GPT) -> None:
    out = gpt.generate(torch.zeros((B, 1), dtype=torch.long), max_new_tokens=5, top_k=1)
    assert torch.equal(out, gpt.generate(torch.zeros((B, 1), dtype=torch.long), 5, 0.0))
