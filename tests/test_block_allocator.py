"""The block allocator: a free list of KV pages, with ref counts for shared pages."""

import pytest

from pico_vllm.serving.block_allocator import BlockAllocator, OutOfBlocksError


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
