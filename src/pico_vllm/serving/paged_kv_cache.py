"""What this file teaches: PagedAttention's memory layout, and prefix caching.

Key idea: one shared pool of pages per layer; each sequence has a block table (its i-th page
lives in block table[i]), so memory grows a page at a time and is never reserved for the
longest reply. With prefix caching, every full page gets a name chained from the start of the text,
so a new request whose prompt starts the same way reuses those pages instead of recomputing.

Read next: serving/sequence.py, then serving/engine.py.
"""

import torch
from torch import Tensor

from pico_vllm.serving.block_allocator import BlockAllocator


class PagedKVCache:
    """KV cache in fixed-size pages, handed out on demand, keyed by sequence id.

    One shared pool per layer: (num_blocks, page_size, n_kv_head, head_size). Each sequence
    has a block table: its i-th page lives in physical block table[i]. The same block id is
    used in every layer's pool, like vLLM. bind() says which sequence each batch row is.

    prefix_caching: every full page gets a name, hash(previous page's name, its tokens),
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
        # prefix caching
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
            self.v[layer, blocks, offsets] = v[row].transpose(0, 1)  # (T, n_kv, hs)
            blocks, offsets = self._slots(seq_id, 0, start + T)  # every position back, in order
            ks.append(self.k[layer, blocks, offsets].transpose(0, 1))  # (n_kv, start + T, hs)
            vs.append(self.v[layer, blocks, offsets].transpose(0, 1))  # (n_kv, start + T, hs)
        # Simplification: gathers (copies) every history each step and pads to the longest; a
        # PagedAttention kernel reads pages in place. DESIGN "Gaps vs vLLM": attention kernel.
        longest = max(x.size(1) for x in ks)
        k_out, v_out = k.new_zeros(B, n_kv, longest, hs), v.new_zeros(B, n_kv, longest, hs)
        for row, (kr, vr) in enumerate(zip(ks, vs, strict=True)):
            k_out[row, :, : kr.size(1)] = kr
            v_out[row, :, : vr.size(1)] = vr
        return k_out, v_out

    def free_seq(self, seq_id: int) -> None:
        """Return one sequence's pages to the pool (the moment it finishes).

        Named pages keep their name and content: until overwritten, they can be revived.
        """
        for block in self.block_tables.pop(seq_id, []):
            self.allocator.free(block)
        self.seq_names.pop(seq_id, None)

    # ---- prefix caching ---------------------------------------------------------------------

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
            # Simplification: Python's hash; production uses a strong hash (vLLM: optional SHA-256)
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
