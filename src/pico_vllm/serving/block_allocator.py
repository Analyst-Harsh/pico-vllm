"""What this file teaches: the free list behind paged KV memory.

Key idea: KV memory is cut into fixed-size blocks (pages). A sequence takes one only when its last
is full and returns it the moment it finishes. A ref count lets several sequences share a
block that holds the same prompt text.

Read next: serving/paged_kv_cache.py.
"""

from collections import deque


class OutOfBlocksError(RuntimeError):
    """No free KV block left. The engine reacts by waiting or preempting."""


class BlockAllocator:
    """Hands out KV block ids from a fixed pool.

    Ref counts (for prefix caching): a block shared by several sequences returns to the pool
    only when its last user frees it. Freed blocks queue oldest-first, so allocate() hands out
    the least recently freed one: freed blocks that still hold a cached prefix live longest.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))  # allocate() takes the left end (oldest freed)
        self._refs: dict[int, int] = {}  # block -> how many sequences use it

    @property
    def num_free(self) -> int:
        return len(self._free)

    def refs(self, block: int) -> int:
        return self._refs.get(block, 0)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError(f"all {self.num_blocks} KV blocks are in use")
        block = self._free.popleft()
        self._refs[block] = 1
        return block

    def share(self, block: int) -> None:
        """One more user for this block; a freed block is taken back out of the pool."""
        if block not in self._refs:
            self._free.remove(block)  # Simplification: O(num_blocks) scan; vLLM links free blocks
        self._refs[block] = self._refs.get(block, 0) + 1

    def free(self, block: int) -> None:
        if block not in self._refs:  # never allocated, or freed twice
            raise ValueError(f"block {block} is not allocated")
        self._refs[block] -= 1
        if self._refs[block] == 0:
            del self._refs[block]
            self._free.append(block)  # newest freed at the right end
