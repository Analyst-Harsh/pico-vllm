"""F6: Llama-style building blocks (batched heads, then RMSNorm and GQA).

model.py keeps the F4-F5 GPT unchanged, so old checkpoints and experiments still load.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm import model


class CausalSelfAttention(nn.Module):
    """All heads in one go: heads are a tensor dimension, not a Python loop. F6 step 1.

    Same maths as MultiHeadAttention; one big matmul instead of n_head small ones.
    """

    tril: Tensor

    def __init__(self, n_embd: int, n_head: int, ctx_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        # one Linear makes q, k and v for every head: output = [q | k | v], each n_embd wide
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(ctx_len, ctx_len)))

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd)."""
        B, T, C = x.shape
        hs = C // self.n_head
        # 1. one matmul makes q, k, v for every head at once
        q, k, v = self.c_attn(x).split(C, dim=-1)  # each (B, T, C)
        # 2. cut C into n_head slices of hs, then move heads next to batch
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # 3. same as Head, but every head in one batched matmul
        wei = q @ k.transpose(-2, -1) * hs**-0.5  # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # 4. weighted average of values, per head
        out = wei @ v  # (B, nh, T, hs)
        # 5. heads back side by side = the torch.cat in MultiHeadAttention
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # 6. mix heads, same as before
        return self.dropout(self.proj(out))


class ModernBlock(model.Block):
    """F5 Block with the looped heads swapped for CausalSelfAttention. Steps 2-3 edit here."""

    def __init__(self, n_embd: int, n_head: int, ctx_len: int, dropout: float = 0.0) -> None:
        super().__init__(n_embd, n_head, ctx_len, dropout)
        self.sa = CausalSelfAttention(n_embd, n_head, ctx_len, dropout)


class ModernGPT(model.GPT):
    """F5 GPT built from ModernBlocks. Same forward, generate and init."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        ctx_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, n_embd, n_head, n_layer, ctx_len, dropout)
        self.blocks = nn.Sequential(
            *(ModernBlock(n_embd, n_head, ctx_len, dropout) for _ in range(n_layer))
        )
        self.apply(self._init_weights)  # the new blocks need the GPT-2 init too
