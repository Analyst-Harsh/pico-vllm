"""v9 B1: a contiguous KV cache. Every sequence gets ctx_len slots per layer up front.

B2 replaces this with pages (blocks handed out on demand) to stop reserving unused slots.
"""

import torch
from torch import Tensor


class KVCache:
    """Per-layer k and v buffers of shape (n_layer, B, n_kv_head, ctx_len, head_size)."""

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_kv_head: int,
        ctx_len: int,
        head_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layer, batch_size, n_kv_head, ctx_len, head_size)
        # ponytail: reserves ctx_len slots per sequence even if it stops early; B2 pages it
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def update(self, layer: int, start_pos: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write k, v (B, n_kv, T, hs) at positions [start_pos, start_pos + T).

        Returns every cached k, v for this layer so far: (B, n_kv, start_pos + T, hs).
        """
        end = start_pos + k.size(2)
        self.k[layer, :, :, start_pos:end] = k
        self.v[layer, :, :, start_pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]


class OutOfBlocksError(RuntimeError):
    """No free KV block left. B3's scheduler reacts by waiting or preempting."""


class BlockAllocator:
    """Hands out KV block ids from a fixed pool. v9 B2 (the §5 interview piece)."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks))  # a stack: free() pushes, allocate() pops
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError(f"all {self.num_blocks} KV blocks are in use")
        block = self._free.pop()
        self._used.add(block)
        return block

    def free(self, block: int) -> None:
        if block not in self._used:  # never allocated, or freed twice
            raise ValueError(f"block {block} is not allocated")
        self._used.remove(block)
        self._free.append(block)


class PagedKVCache:
    """KV cache in fixed-size pages, handed out on demand. v9 B2.

    One shared pool per layer: (num_blocks, page_size, n_kv_head, head_size). Each sequence
    (batch row) has a block table: its i-th page lives in physical block table[i]. The same
    block id is used in every layer's pool, like vLLM.
    """

    def __init__(
        self,
        n_layer: int,
        num_blocks: int,
        page_size: int,
        n_kv_head: int,
        head_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layer, num_blocks, page_size, n_kv_head, head_size)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.page_size = page_size
        self.allocator = BlockAllocator(num_blocks)
        self.block_tables: list[list[int]] = [[] for _ in range(batch_size)]

    def _slots(self, row: int, start: int, end: int) -> tuple[Tensor, Tensor]:
        """Physical (block, offset) of positions [start, end) of one sequence."""
        table = self.block_tables[row]
        while len(table) * self.page_size < end:  # a new page only when the last one is full
            table.append(self.allocator.allocate())
        pos = torch.arange(start, end)
        blocks = torch.tensor(table, device=self.k.device)[pos // self.page_size]
        return blocks, (pos % self.page_size).to(self.k.device)

    def update(self, layer: int, start_pos: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Same contract as KVCache.update: write k, v (B, n_kv, T, hs); return all so far."""
        end = start_pos + k.size(2)
        ks, vs = [], []
        for row in range(k.size(0)):
            blocks, offsets = self._slots(row, start_pos, end)
            self.k[layer, blocks, offsets] = k[row].transpose(0, 1)  # (T, n_kv, hs) into pages
            self.v[layer, blocks, offsets] = v[row].transpose(0, 1)
            blocks, offsets = self._slots(row, 0, end)  # gather every position back, in order
            ks.append(self.k[layer, blocks, offsets].transpose(0, 1))  # (n_kv, end, hs)
            vs.append(self.v[layer, blocks, offsets].transpose(0, 1))
        # ponytail: gathers (copies) the whole history every step; a PagedAttention kernel
        # reads pages in place. DESIGN "Gaps vs vLLM": attention kernel.
        return torch.stack(ks), torch.stack(vs)

    def free(self) -> None:
        """Return every sequence's blocks to the pool."""
        for table in self.block_tables:
            for block in table:
                self.allocator.free(block)
            table.clear()
