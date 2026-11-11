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
