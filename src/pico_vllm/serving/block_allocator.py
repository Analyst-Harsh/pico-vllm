"""v9 B2: the block allocator behind the paged KV cache."""

from collections import deque


class OutOfBlocksError(RuntimeError):
    """No free KV block left. B3's scheduler reacts by waiting or preempting."""


class BlockAllocator:
    """Hands out KV block ids from a fixed pool. v9 B2 (the §5 interview piece).

    B4 adds ref counts: a block shared by several sequences returns to the pool only when its
    last user frees it. Freed blocks queue oldest-first, so allocate() hands out the least
    recently freed one: freed blocks that still hold a cached prefix live as long as possible.
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
        """One more user for this block; a freed block is taken back out of the pool (B4)."""
        if block not in self._refs:
            self._free.remove(block)  # ponytail: O(num_blocks) scan; vLLM links free blocks
        self._refs[block] = self._refs.get(block, 0) + 1

    def free(self, block: int) -> None:
        if block not in self._refs:  # never allocated, or freed twice
            raise ValueError(f"block {block} is not allocated")
        self._refs[block] -= 1
        if self._refs[block] == 0:
            del self._refs[block]
            self._free.append(block)  # newest freed at the right end
