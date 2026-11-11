"""v9 B1: KV cache. Cached results must equal recomputing everything."""

import pytest
import torch

from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.serving.kv_cache import KVCache

V, C, NH, NL, CTX = 65, 16, 4, 2, 32


def tiny(n_kv_head: int | None = None) -> ModernGPT:
    torch.manual_seed(1337)
    model = ModernGPT(V, C, NH, NL, CTX, rmsnorm=True, n_kv_head=n_kv_head).eval()
    for p in model.parameters():  # GPT init (std 0.02) gives near-flat logits; make them peaky
        torch.nn.init.normal_(p, std=0.5)
    return model


def test_cache_update_writes_and_returns_the_prefix() -> None:
    cache = KVCache(n_layer=1, batch_size=2, n_kv_head=2, ctx_len=8, head_size=3)
    k1, v1 = torch.randn(2, 2, 3, 3), torch.randn(2, 2, 3, 3)
    k, v = cache.update(0, 0, k1, v1)
    assert k.shape == (2, 2, 3, 3)
    assert torch.equal(k, k1)
    k2, v2 = torch.randn(2, 2, 1, 3), torch.randn(2, 2, 1, 3)
    k, v = cache.update(0, 3, k2, v2)
    assert k.shape == (2, 2, 4, 3)
    assert torch.equal(k[:, :, :3], k1)
    assert torch.equal(v[:, :, 3:], v2)


@pytest.mark.parametrize("n_kv_head", [None, 2, 1])
def test_prefill_then_decode_matches_full_forward(n_kv_head: int | None) -> None:
    model = tiny(n_kv_head)
    idx = torch.randint(0, V, (2, 10))
    full, _ = model(idx)  # (2, 10, V), no cache
    cache = model.new_cache(2, "cpu")
    first, _ = model(idx[:, :6], cache=cache, start_pos=0)  # prefill 6
    steps = [model(idx[:, t : t + 1], cache=cache, start_pos=t)[0] for t in range(6, 10)]
    cached = torch.cat([first, *steps], dim=1)
    assert torch.allclose(cached, full, atol=1e-5)


@pytest.mark.parametrize("n_kv_head", [None, 2])
def test_greedy_generate_cached_equals_generate(n_kv_head: int | None) -> None:
    model = tiny(n_kv_head)
    prompt = torch.randint(0, V, (2, 5))
    slow = model.generate(prompt, max_new_tokens=20, temperature=0)
    fast = model.generate_cached(prompt, max_new_tokens=20, temperature=0)
    assert torch.equal(fast, slow)


def test_past_ctx_len_raises() -> None:
    model = tiny()
    with pytest.raises(ValueError, match="ctx_len"):
        model.generate_cached(torch.zeros(1, 30, dtype=torch.long), max_new_tokens=5)
