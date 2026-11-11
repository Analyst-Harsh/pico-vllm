"""The GPT-2-style transformer: FeedForward, Block, GPT."""

import math

import pytest
import torch

from pico_vllm.model.gpt import GPT, Block, FeedForward

B, T, C, NH, CTX = 2, 6, 16, 4, 8
V, NL = 11, 2


# ---- FeedForward: a per-token MLP -----------------------------------------------------------


@pytest.fixture
def ffn() -> FeedForward:
    torch.manual_seed(1337)
    return FeedForward(C).eval()


def test_ffn_output_shape(ffn: FeedForward) -> None:
    assert ffn(torch.randn(B, T, C)).shape == (B, T, C)


def test_hidden_layer_is_4x_wider(ffn: FeedForward) -> None:
    widths = [m.out_features for m in ffn.modules() if isinstance(m, torch.nn.Linear)]
    assert widths == [4 * C, C]


def test_each_token_is_processed_alone(ffn: FeedForward) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 2] = torch.randn(B, C)  # change token 2 only
    out, out2 = ffn(x), ffn(x2)
    others = [i for i in range(T) if i != 2]
    assert torch.allclose(out[:, others], out2[:, others])  # past AND future untouched
    assert not torch.allclose(out[:, 2], out2[:, 2])


# ---- Block: attention then feed-forward, pre-norm, residuals --------------------------------


@pytest.fixture
def block() -> Block:
    torch.manual_seed(1337)
    return Block(n_embd=C, n_head=NH, ctx_len=CTX).eval()


def test_block_output_shape(block: Block) -> None:
    assert block(torch.randn(B, T, C)).shape == (B, T, C)


def test_block_causal(block: Block) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(block(x)[:, :3], block(x2)[:, :3])


def test_residual_block_with_silent_sublayers_is_identity(block: Block) -> None:
    # zero the last layer of each sub-layer -> both add 0 -> the residual path returns x
    last_layers = [block.sa.proj, block.ffwd.net[2]]
    with torch.no_grad():
        for layer in last_layers:
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(B, T, C)
    assert torch.allclose(block(x), x)


# ---- GPT: embeddings -> Blocks -> lm_head ---------------------------------------------------


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


def test_gpt_causal(gpt: GPT) -> None:
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
