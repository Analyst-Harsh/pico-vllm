"""F6 step 3 target: GroupedQueryAttention in src/pico_vllm/modern.py."""

import pytest
import torch

from pico_vllm.model import CausalSelfAttention
from pico_vllm.modern import GroupedQueryAttention, ModernGPT

B, T, C, NH, CTX = 2, 6, 16, 4, 8
HS = C // NH


def count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_output_shape(n_kv_head: int) -> None:
    gqa = GroupedQueryAttention(C, NH, n_kv_head, CTX)
    assert gqa(torch.randn(B, T, C)).shape == (B, T, C)


def test_no_grouping_is_exactly_mha() -> None:
    torch.manual_seed(1337)
    csa = CausalSelfAttention(C, NH, CTX).eval()
    gqa = GroupedQueryAttention(C, NH, NH, CTX).eval()
    gqa.load_state_dict(csa.state_dict())  # same shapes when n_kv_head == n_head
    x = torch.randn(B, T, C)
    assert torch.allclose(gqa(x), csa(x), atol=1e-6)


def test_query_heads_in_a_group_share_one_kv_head() -> None:
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
def test_causal(n_kv_head: int) -> None:
    torch.manual_seed(1337)
    gqa = GroupedQueryAttention(C, NH, n_kv_head, CTX).eval()
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(gqa(x)[:, :3], gqa(x2)[:, :3])


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_params_drop_by_the_kv_weights_saved(n_kv_head: int) -> None:
    saved = 2 * (NH - n_kv_head) * HS * C  # k and v rows no longer needed
    assert (
        count(GroupedQueryAttention(C, NH, n_kv_head, CTX))
        == count(CausalSelfAttention(C, NH, CTX)) - saved
    )


def test_kv_heads_must_divide_query_heads() -> None:
    with pytest.raises(ValueError, match="n_kv_head"):
        GroupedQueryAttention(C, NH, 3, CTX)


def test_modern_gpt_n_kv_head_shrinks_every_layer() -> None:
    args = {"vocab_size": 65, "n_embd": C, "n_head": NH, "n_layer": 2, "ctx_len": CTX}
    mha, gqa = ModernGPT(**args), ModernGPT(**args, n_kv_head=2)
    assert count(mha) - count(gqa) == 2 * (2 * (NH - 2) * HS * C)  # 2 layers
    assert all(b.sa.n_kv_head == 2 for b in gqa.blocks)


def test_matches_repeat_interleave_reference() -> None:
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
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = gqa.proj(ref.transpose(1, 2).reshape(B, T, C))
        assert torch.allclose(gqa(x), ref, atol=1e-6)
