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

    def update(
        self, layer: int, start_pos: int | Tensor, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Write k, v (B, n_kv, T, hs) at positions [start_pos, start_pos + T).

        Returns every cached k, v for this layer so far: (B, n_kv, start_pos + T, hs).
        """
        if isinstance(start_pos, Tensor):
            raise TypeError("the contiguous cache needs one start_pos for the whole batch")
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
    """KV cache in fixed-size pages, handed out on demand. v9 B2, keyed by sequence in B3.

    One shared pool per layer: (num_blocks, page_size, n_kv_head, head_size). Each sequence
    has a block table: its i-th page lives in physical block table[i]. The same block id is
    used in every layer's pool, like vLLM. bind() says which sequence each batch row is.
    """

    def __init__(
        self,
        n_layer: int,
        num_blocks: int,
        page_size: int,
        n_kv_head: int,
        head_size: int,
        batch_size: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layer, num_blocks, page_size, n_kv_head, head_size)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.page_size = page_size
        self.allocator = BlockAllocator(num_blocks)
        self.block_tables: dict[int, list[int]] = {}  # sequence id -> its pages, in order
        self.bind(list(range(batch_size)))  # generate_cached: row i is sequence i

    def bind(self, seq_ids: list[int]) -> None:
        """The next forward's batch rows are these sequences (row i = seq_ids[i])."""
        self.rows = seq_ids
        for seq_id in seq_ids:
            self.block_tables.setdefault(seq_id, [])

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.page_size)  # ceil

    def _slots(self, seq_id: int, start: int, end: int) -> tuple[Tensor, Tensor]:
        """Physical (block, offset) of positions [start, end) of one sequence."""
        table = self.block_tables[seq_id]
        while len(table) * self.page_size < end:  # a new page only when the last one is full
            table.append(self.allocator.allocate())
        pos = torch.arange(start, end)
        blocks = torch.tensor(table, device=self.k.device)[pos // self.page_size]
        return blocks, (pos % self.page_size).to(self.k.device)

    def update(
        self, layer: int, start_pos: int | Tensor, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Same contract as KVCache.update, but each row may start at its own position.

        Rows with a shorter history are padded with zeros at the end; attention's mask hides
        them because they sit past that row's own position.
        """
        B, n_kv, T, hs = k.shape
        ks, vs = [], []
        for row, seq_id in enumerate(self.rows):
            start = int(start_pos[row]) if isinstance(start_pos, Tensor) else start_pos
            blocks, offsets = self._slots(seq_id, start, start + T)
            self.k[layer, blocks, offsets] = k[row].transpose(0, 1)  # (T, n_kv, hs) into pages
            self.v[layer, blocks, offsets] = v[row].transpose(0, 1)
            blocks, offsets = self._slots(seq_id, 0, start + T)  # every position back, in order
            ks.append(self.k[layer, blocks, offsets].transpose(0, 1))  # (n_kv, start + T, hs)
            vs.append(self.v[layer, blocks, offsets].transpose(0, 1))
        # ponytail: gathers (copies) every history each step and pads to the longest; a
        # PagedAttention kernel reads pages in place. DESIGN "Gaps vs vLLM": attention kernel.
        longest = max(x.size(1) for x in ks)
        k_out, v_out = k.new_zeros(B, n_kv, longest, hs), v.new_zeros(B, n_kv, longest, hs)
        for row, (kr, vr) in enumerate(zip(ks, vs, strict=True)):
            k_out[row, :, : kr.size(1)] = kr
            v_out[row, :, : vr.size(1)] = vr
        return k_out, v_out

    def free_seq(self, seq_id: int) -> None:
        """Return one sequence's pages to the pool (B3: the moment it finishes)."""
        for block in self.block_tables.pop(seq_id, []):
            self.allocator.free(block)

    def free(self) -> None:
        """Return every sequence's pages to the pool."""
        for seq_id in list(self.block_tables):
            self.free_seq(seq_id)
