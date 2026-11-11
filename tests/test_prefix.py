"""v9 B4: prefix caching. Reused pages must give the same outputs as computing everything."""

import pytest
import torch

from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.serving.block_allocator import BlockAllocator
from pico_vllm.serving.engine import Engine

V, C, NH, NKV, NL, CTX, PAGE = 65, 16, 4, 2, 2, 32, 4
SYSTEM = [3, 1, 4, 1, 5, 9, 2, 6]  # a shared 2-page "system prompt"


def tiny() -> ModernGPT:
    torch.manual_seed(1337)
    model = ModernGPT(V, C, NH, NL, CTX, rmsnorm=True, n_kv_head=NKV).eval()
    for p in model.parameters():  # peaky logits so greedy is sensitive to any reuse bug
        torch.nn.init.normal_(p, std=0.5)
    return model


def alone(model: ModernGPT, prompt: list[int], n: int) -> list[int]:
    out = model.generate_cached(torch.tensor([prompt]), n, temperature=0)
    return out[0, len(prompt) :].tolist()


def engine(model: ModernGPT, num_blocks: int = 64, max_num_seqs: int = 4) -> Engine:
    return Engine(model, num_blocks, PAGE, max_num_seqs, prefix_caching=True)


def test_allocator_ref_counts() -> None:
    alloc = BlockAllocator(3)
    b = alloc.allocate()
    alloc.share(b)  # a second user
    alloc.free(b)
    assert alloc.refs(b) == 1  # still in use by one
    assert alloc.num_free == 2
    alloc.free(b)
    assert alloc.num_free == 3  # last user gone: back in the pool
    with pytest.raises(ValueError, match="not allocated"):
        alloc.free(b)


def test_concurrent_requests_share_prefix_pages() -> None:
    model = tiny()
    eng = engine(model)
    a = eng.add_request([*SYSTEM, 10, 11], 5)
    b = eng.add_request([*SYSTEM, 20, 21, 22], 5)
    eng.step()  # both admitted in the same step, one after the other
    ta, tb = eng.cache.block_tables[a.seq_id], eng.cache.block_tables[b.seq_id]
    assert tb[:2] == ta[:2]  # the system prompt's 2 pages are the same physical pages
    assert tb[2:] != ta[2:]  # the different tails are not
    assert all(eng.cache.allocator.refs(blk) == 2 for blk in ta[:2])
    eng.run()
    assert a.output == alone(model, a.prompt, 5)
    assert b.output == alone(model, b.prompt, 5)
    assert eng.prefill_cached == 8  # b skipped the 8 system tokens
    assert eng.cache.allocator.num_free == 64


def test_freed_prefix_pages_are_revived() -> None:
    model = tiny()
    eng = engine(model)
    first = eng.add_request([*SYSTEM, 10], 3, arrival_step=0)
    later = eng.add_request([*SYSTEM, 30, 31], 4, arrival_step=10)  # first is long gone by then
    eng.run()
    assert first.finish_step is not None
    assert first.finish_step < 10
    assert eng.prefill_cached == 8
    assert later.output == alone(model, later.prompt, 4)


def test_at_least_one_token_is_always_prefilled() -> None:
    model = tiny()
    eng = engine(model)
    eng.add_request(list(SYSTEM), 3)  # prompt = exactly 2 full pages
    again = eng.add_request(list(SYSTEM), 3, arrival_step=10)
    eng.run()
    assert eng.prefill_cached == 4  # only 1 page reused: the last token must still be computed
    assert again.output == alone(model, again.prompt, 3)


def test_eviction_under_memory_pressure_keeps_outputs_correct() -> None:
    # a tiny pool forces freed named pages to be overwritten; reuse must never read stale data
    model = tiny()
    eng = engine(model, num_blocks=6, max_num_seqs=2)
    prompts = [[*SYSTEM, i, i + 1] for i in range(10, 40, 5)]
    seqs = [eng.add_request(p, 8, arrival_step=3 * i) for i, p in enumerate(prompts)]
    seqs.append(eng.add_request([7, 7, 7, 7, 8, 8, 8, 8, 1], 8, arrival_step=1))
    eng.run()
    for s in seqs:
        assert s.output == alone(model, s.prompt, 8)
    assert eng.cache.allocator.num_free == 6
