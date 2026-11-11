"""What this file teaches: the KV cache, the first serving efficiency.

Key idea: a token's keys and values never change once computed, so store them per layer and only run
the newest token through the model at each step. This contiguous version reserves ctx_len
slots per sequence up front, which wastes the tail of every short reply.

Read next: serving/block_allocator.py (the first piece of the paged fix).
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
        # Simplification: reserves ctx_len slots per sequence even if it stops early;
        # PagedKVCache hands out pages on demand instead
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
