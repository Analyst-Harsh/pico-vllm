"""v9 B3: the engine. Batched, interleaved, preempted: outputs must equal one-at-a-time greedy."""

import random

import pytest
import torch

from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.serving.engine import Engine
from pico_vllm.serving.sequence import Status

V, C, NH, NKV, NL, CTX, PAGE = 65, 16, 4, 2, 2, 32, 4


def tiny() -> ModernGPT:
    torch.manual_seed(1337)
    model = ModernGPT(V, C, NH, NL, CTX, rmsnorm=True, n_kv_head=NKV).eval()
    for p in model.parameters():  # peaky logits so greedy is sensitive to any batching bug
        torch.nn.init.normal_(p, std=0.5)
    return model


def requests(n: int = 8, seed: int = 0) -> list[tuple[list[int], int, int]]:
    """(prompt, max_new_tokens, arrival_step): different lengths, staggered arrivals."""
    rng = random.Random(seed)  # noqa: S311  (a seeded workload, not security)
    return [
        (
            [rng.randrange(V) for _ in range(rng.randint(2, 9))],
            rng.randint(1, 14),
            rng.randint(0, 6),
        )
        for _ in range(n)
    ]


def alone(model: ModernGPT, prompt: list[int], n: int) -> list[int]:
    out = model.generate_cached(torch.tensor([prompt]), n, temperature=0)
    return out[0, len(prompt) :].tolist()


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_outputs_equal_one_at_a_time(policy: str) -> None:
    model = tiny()
    eng = Engine(model, num_blocks=64, page_size=PAGE, max_num_seqs=3, policy=policy)
    reqs = requests()
    seqs = [eng.add_request(p, n, a) for p, n, a in reqs]
    eng.run()
    for s, (p, n, _) in zip(seqs, reqs, strict=True):
        assert s.status is Status.FINISHED
        assert s.output == alone(model, p, n)
    assert max(st.running for st in eng.stats) <= 3  # never more than max_num_seqs
    assert eng.cache.allocator.num_free == 64  # every page came back


def test_preemption_keeps_outputs_correct() -> None:
    # 3 sequences that each grow to 5 pages, in an 8-page pool: someone must be preempted
    model = tiny()
    eng = Engine(model, num_blocks=8, page_size=PAGE, max_num_seqs=3)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
    seqs = [eng.add_request(p, 14) for p in prompts]
    eng.run()
    assert eng.preemptions > 0
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output == alone(model, p, 14)
    assert eng.cache.allocator.num_free == 8


def test_continuous_frees_slots_for_late_arrivals() -> None:
    # one long request, then a short one arriving at step 1: continuous serves it at once,
    # static makes it wait for the long one to finish
    ttft = {}
    for policy in ("continuous", "static"):
        eng = Engine(tiny(), num_blocks=64, page_size=PAGE, max_num_seqs=2, policy=policy)
        eng.add_request([1, 2, 3], 20, arrival_step=0)
        short = eng.add_request([4, 5, 6], 3, arrival_step=1)
        eng.run()
        ttft[policy] = short.ttft
    assert ttft["continuous"] == 1
    assert ttft["static"] > 15


def test_request_larger_than_pool_is_rejected() -> None:
    eng = Engine(tiny(), num_blocks=2, page_size=PAGE)
    with pytest.raises(ValueError, match="pages"):
        eng.add_request([1, 2, 3, 4, 5, 6], 10)  # 16 tokens = 4 pages > 2
