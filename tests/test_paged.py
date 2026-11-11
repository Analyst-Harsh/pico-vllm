"""v9 B2: paged KV cache. Paged must equal contiguous; blocks are never handed out twice."""

import pytest
import torch

from pico_vllm.modern import ModernGPT
from pico_vllm.serving.block_allocator import BlockAllocator, OutOfBlocksError
from pico_vllm.serving.kv_cache import KVCache
from pico_vllm.serving.paged_kv_cache import PagedKVCache

V, C, NH, NKV, NL, CTX, PAGE = 65, 16, 4, 2, 2, 32, 4
HS = C // NH


def tiny() -> ModernGPT:
    torch.manual_seed(1337)
    model = ModernGPT(V, C, NH, NL, CTX, rmsnorm=True, n_kv_head=NKV).eval()
    for p in model.parameters():  # peaky logits so greedy is sensitive to any cache bug
        torch.nn.init.normal_(p, std=0.5)
    return model


def paged(batch_size: int, num_blocks: int = 32) -> PagedKVCache:
    return PagedKVCache(NL, num_blocks, PAGE, NKV, HS, batch_size)


def test_allocator_never_hands_out_a_block_twice() -> None:
    alloc = BlockAllocator(4)
    got = [alloc.allocate() for _ in range(4)]
    assert sorted(got) == [0, 1, 2, 3]
    assert alloc.num_free == 0
    with pytest.raises(OutOfBlocksError):
        alloc.allocate()
    alloc.free(got[1])
    assert alloc.num_free == 1
    assert alloc.allocate() == got[1]  # reused only after it was freed


def test_allocator_rejects_double_free() -> None:
    alloc = BlockAllocator(2)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(ValueError, match="not allocated"):
        alloc.free(b)


def test_paged_update_matches_contiguous_across_page_boundaries() -> None:
    torch.manual_seed(0)
    flat, pages = KVCache(NL, 2, NKV, CTX, HS), paged(2)
    for start, T in [(0, 6), (6, 1), (7, 1), (8, 3)]:  # prefill crosses a page, then decode
        k, v = torch.randn(2, NKV, T, HS), torch.randn(2, NKV, T, HS)
        fk, fv = flat.update(1, start, k, v)
        pk, pv = pages.update(1, start, k, v)
        assert torch.equal(pk, fk)
        assert torch.equal(pv, fv)
    assert [len(t) for t in pages.block_tables.values()] == [
        3,
        3,
    ]  # 11 positions / page 4 -> 3 pages


def test_paged_generate_equals_contiguous_generate() -> None:
    model = tiny()
    prompt = torch.randint(0, V, (2, 5))
    flat = model.generate_cached(prompt, max_new_tokens=20, temperature=0)
    cache = paged(2)
    out = model.generate_cached(prompt, max_new_tokens=20, temperature=0, cache=cache)
    assert torch.equal(out, flat)
    # 24 positions written (the last sampled token is never fed back): 6 pages each, not 32
    assert [len(t) for t in cache.block_tables.values()] == [6, 6]
    used = cache.block_tables[0] + cache.block_tables[1]
    assert len(set(used)) == len(used)  # no block shared between the two sequences
    cache.free()
    assert cache.allocator.num_free == 32


def test_running_out_of_blocks_raises() -> None:
    model = tiny()
    with pytest.raises(OutOfBlocksError):  # needs 6 pages per sequence, only 4 in the pool
        model.generate_cached(
            torch.zeros(1, 5, dtype=torch.long), 20, temperature=0, cache=paged(1, num_blocks=4)
        )
