"""Attention, step by step: each version must equal the one before it."""

import pytest
import torch
from torch.nn import functional as F

from pico_vllm.model.attention import (
    CausalSelfAttention,
    GroupedQueryAttention,
    Head,
    MultiHeadAttention,
)

B, T, C, NH, CTX = 2, 6, 16, 4, 8
HS = C // NH


def count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---- Head: one head of causal self-attention ------------------------------------------------


@pytest.fixture
def head() -> Head:
    torch.manual_seed(1337)
    return Head(n_embd=C, head_size=HS, ctx_len=CTX).eval()


def test_head_output_shape(head: Head) -> None:
    x = torch.randn(B, T, C)
    assert head(x).shape == (B, T, HS)


def test_head_causal_future_does_not_change_past(head: Head) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)  # change tokens 3.. only
    out, out2 = head(x), head(x2)
    assert torch.allclose(out[:, :3], out2[:, :3])  # positions 0..2 never saw the future
    assert not torch.allclose(out[:, 3:], out2[:, 3:])


def test_head_matches_reference_attention(head: Head, monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.randn(B, T, C)
    q, k, v = head.query(x), head.key(x), head.value(x)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def banned(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("write attention yourself; don't call the reference")

    monkeypatch.setattr(F, "scaled_dot_product_attention", banned)
    assert torch.allclose(head(x), expected, atol=1e-6)


# ---- MultiHeadAttention: n_head Heads in a loop ---------------------------------------------


@pytest.fixture
def mha() -> MultiHeadAttention:
    torch.manual_seed(1337)
    return MultiHeadAttention(n_embd=C, n_head=NH, ctx_len=CTX).eval()


def test_mha_output_shape(mha: MultiHeadAttention) -> None:
    assert mha(torch.randn(B, T, C)).shape == (B, T, C)


def test_mha_heads_split_the_width(mha: MultiHeadAttention) -> None:
    assert len(mha.heads) == NH
    assert all(h.key.out_features == C // NH for h in mha.heads)


def test_mha_is_concat_of_heads_then_proj(mha: MultiHeadAttention) -> None:
    x = torch.randn(B, T, C)
    expected = mha.proj(torch.cat([h(x) for h in mha.heads], dim=-1))
    assert torch.allclose(mha(x), expected)


def test_mha_causal(mha: MultiHeadAttention) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(mha(x)[:, :3], mha(x2)[:, :3])


def test_mha_rejects_width_not_divisible_by_heads() -> None:
    with pytest.raises((AssertionError, ValueError)):
        MultiHeadAttention(n_embd=10, n_head=4, ctx_len=CTX)


# ---- CausalSelfAttention: every head in one batched matmul ----------------------------------


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


def test_csa_output_shape(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    _, csa = pair
    assert csa(torch.randn(B, T, C)).shape == (B, T, C)


def test_csa_equals_looped_heads(
    pair: tuple[MultiHeadAttention, CausalSelfAttention],
) -> None:
    mha, csa = pair
    x = torch.randn(B, T, C)
    assert torch.allclose(csa(x), mha(x), atol=1e-6)


def test_csa_same_param_count(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    mha, csa = pair
    assert count(csa) == count(mha)


def test_csa_causal(pair: tuple[MultiHeadAttention, CausalSelfAttention]) -> None:
    _, csa = pair
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(csa(x)[:, :3], csa(x2)[:, :3])


# ---- GroupedQueryAttention: query heads share k/v heads -------------------------------------


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_gqa_output_shape(n_kv_head: int) -> None:
    gqa = GroupedQueryAttention(C, NH, n_kv_head, CTX)
    assert gqa(torch.randn(B, T, C)).shape == (B, T, C)


def test_gqa_no_grouping_is_exactly_mha() -> None:
    torch.manual_seed(1337)
    csa = CausalSelfAttention(C, NH, CTX).eval()
    gqa = GroupedQueryAttention(C, NH, NH, CTX).eval()
    gqa.load_state_dict(csa.state_dict())  # same shapes when n_kv_head == n_head
    x = torch.randn(B, T, C)
    assert torch.allclose(gqa(x), csa(x), atol=1e-6)


def test_gqa_query_heads_in_a_group_share_one_kv_head() -> None:
    # one k/v head (MQA) and identical q weights for every head -> every head's output is equal
    torch.manual_seed(1337)
    gqa = GroupedQueryAttention(C, NH, 1, CTX).eval()
    with torch.no_grad():
        q_w = gqa.c_attn.weight[:HS].clone()  # head 0's query rows
        gqa.c_attn.weight[:C] = q_w.repeat(NH, 1)
        gqa.proj.weight.copy_(torch.eye(C))  # proj = identity, so we see the raw head outputs
        gqa.proj.bias.zero_()
    heads = gqa(torch.randn(B, T, C)).view(B, T, NH, HS)
    for h in range(1, NH):
        assert torch.allclose(heads[:, :, h], heads[:, :, 0], atol=1e-6)


@pytest.mark.parametrize("n_kv_head", [2, 1])
def test_gqa_causal(n_kv_head: int) -> None:
    torch.manual_seed(1337)
    gqa = GroupedQueryAttention(C, NH, n_kv_head, CTX).eval()
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(gqa(x)[:, :3], gqa(x2)[:, :3])


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_gqa_params_drop_by_the_kv_weights_saved(n_kv_head: int) -> None:
    saved = 2 * (NH - n_kv_head) * HS * C  # k and v rows no longer needed
    assert (
        count(GroupedQueryAttention(C, NH, n_kv_head, CTX))
        == count(CausalSelfAttention(C, NH, CTX)) - saved
    )


def test_gqa_kv_heads_must_divide_query_heads() -> None:
    with pytest.raises(ValueError, match="n_kv_head"):
        GroupedQueryAttention(C, NH, 3, CTX)


def test_gqa_matches_repeat_interleave_reference() -> None:
    # the fold must give what copying k/v to every head would (Llama's head -> kv mapping)
    torch.manual_seed(1337)
    nkv = 2
    gqa = GroupedQueryAttention(C, NH, nkv, CTX).eval()
    x = torch.randn(B, T, C)
    with torch.no_grad():
        q, k, v = gqa.c_attn(x).split([C, nkv * HS, nkv * HS], dim=-1)
        q = q.view(B, T, NH, HS).transpose(1, 2)
        k = k.view(B, T, nkv, HS).transpose(1, 2).repeat_interleave(NH // nkv, dim=1)
        v = v.view(B, T, nkv, HS).transpose(1, 2).repeat_interleave(NH // nkv, dim=1)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = gqa.proj(ref.transpose(1, 2).reshape(B, T, C))
        assert torch.allclose(gqa(x), ref, atol=1e-6)
