"""v9 B1: a contiguous KV cache. Every sequence gets ctx_len slots per layer up front.

B2 replaces this with pages (blocks handed out on demand) to stop reserving unused slots.
"""

from collections import deque

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


class PagedKVCache:
    """KV cache in fixed-size pages, handed out on demand. v9 B2, keyed by sequence in B3.

    One shared pool per layer: (num_blocks, page_size, n_kv_head, head_size). Each sequence
    has a block table: its i-th page lives in physical block table[i]. The same block id is
    used in every layer's pool, like vLLM. bind() says which sequence each batch row is.

    prefix_caching (B4): every full page gets a name, hash(previous page's name, its tokens),
    so equal names mean equal text from the very start. A new sequence reuses the pages of
    its longest named prefix (shared while in use, revived if freed but not yet overwritten).
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
        prefix_caching: bool = False,
    ) -> None:
        shape = (n_layer, num_blocks, page_size, n_kv_head, head_size)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.page_size = page_size
        self.allocator = BlockAllocator(num_blocks)
        self.block_tables: dict[int, list[int]] = {}  # sequence id -> its pages, in order
        self.bind(list(range(batch_size)))  # generate_cached: row i is sequence i
        # B4 prefix caching
        self.prefix_caching = prefix_caching
        self.cached: dict[int, int] = {}  # page name -> block holding that text
        self.block_name: dict[int, int] = {}  # block -> its page name (while it holds it)
        self.seq_names: dict[int, list[int]] = {}  # sequence -> names of its full pages

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
            table.append(self._allocate())
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
        """Return one sequence's pages to the pool (B3: the moment it finishes).

        Named pages keep their name and content (B4): until overwritten, they can be revived.
        """
        for block in self.block_tables.pop(seq_id, []):
            self.allocator.free(block)
        self.seq_names.pop(seq_id, None)

    # ---- B4: prefix caching -----------------------------------------------------------------

    def _allocate(self) -> int:
        block = self.allocator.allocate()
        if block in self.block_name:  # overwriting a freed named page: forget its name (evict)
            del self.cached[self.block_name.pop(block)]
        return block

    def _names(self, tokens: list[int], n_pages: int, start: list[int] | None = None) -> list[int]:
        """Names of the first n_pages full pages of tokens, chained from the start."""
        names = list(start or [])
        for i in range(len(names), n_pages):
            parent = names[-1] if names else None
            page = tuple(tokens[i * self.page_size : (i + 1) * self.page_size])
            # ponytail: Python's hash; production uses a strong hash (vLLM: optional SHA-256)
            names.append(hash((parent, page)))
        return names

    def lookup(self, tokens: list[int]) -> list[int]:
        """Blocks holding the longest cached prefix of tokens (full pages only).

        At least one token is left out, so the prefill still has a token to predict from.
        """
        if not self.prefix_caching:
            return []
        blocks = []
        for name in self._names(tokens, (len(tokens) - 1) // self.page_size):
            if name not in self.cached:
                break  # a miss: everything after it differs too
            blocks.append(self.cached[name])
        return blocks

    def admission_cost(self, tokens: list[int], length: int) -> int:
        """Free pages a sequence would take to hold `length` tokens, given its prefix hits.

        Hits on pages in use cost nothing; hits on freed pages still take them off the pile.
        """
        live = sum(self.allocator.refs(b) > 0 for b in self.lookup(tokens))
        return self.blocks_needed(length) - live

    def attach_prefix(self, seq_id: int, tokens: list[int]) -> int:
        """Point a new sequence's table at its cached prefix pages; return tokens covered."""
        hits = self.lookup(tokens)
        for block in hits:
            self.allocator.share(block)
        self.block_tables[seq_id] = list(hits)
        self.seq_names[seq_id] = self._names(tokens, len(hits))
        return len(hits) * self.page_size

    def commit(self, seq_id: int, cached_tokens: list[int]) -> None:
        """Name the sequence's newly filled pages so later requests can find them."""
        if not self.prefix_caching:
            return
        table = self.block_tables[seq_id]
        old = self.seq_names.get(seq_id, [])
        names = self._names(cached_tokens, len(cached_tokens) // self.page_size, old)
        for i in range(len(old), len(names)):
            if names[i] not in self.cached:  # first copy wins; a duplicate stays unnamed
                self.cached[names[i]] = table[i]
                self.block_name[table[i]] = names[i]
        self.seq_names[seq_id] = names

    def free(self) -> None:
        """Return every sequence's pages to the pool."""
        for seq_id in list(self.block_tables):
            self.free_seq(seq_id)
