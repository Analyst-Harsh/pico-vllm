"""What this file teaches: attention, as one progression of four classes.

Key idea: each token builds a query, compares it with every earlier token's key (softmax of q·k),
and takes the weighted average of their values. The four versions compute the same thing:
  Head                  one head
  MultiHeadAttention    n_head Heads in a Python loop, outputs concatenated
  CausalSelfAttention   every head in one batched matmul (heads are a tensor dimension)
  GroupedQueryAttention several query heads share one k/v head: a smaller KV cache

Read next: model/gpt.py (attention stacked into a transformer).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm.serving.kv_cache import KVCache
from pico_vllm.serving.paged_kv_cache import PagedKVCache


class Head(nn.Module):
    """One head of causal self-attention: each token takes a weighted average of past values."""

    tril: Tensor  # registered buffer; annotated so type checkers know it's a tensor

    def __init__(self, n_embd: int, head_size: int, ctx_len: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # a buffer, not a parameter: moves with .to(device) but is never trained
        self.register_buffer("tril", torch.tril(torch.ones(ctx_len, ctx_len)))

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, C) -> out (B, T, head_size)."""
        _, T, _ = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)  # (B, T, hs)
        return wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)


class MultiHeadAttention(nn.Module):
    """n_head small Heads in parallel; concat their outputs, then project back to n_embd."""

    def __init__(self, n_embd: int, n_head: int, ctx_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(Head(n_embd, head_size, ctx_len) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)  # mixes what the heads found
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd)."""
        # n_head outputs of (B, T, hs), side by side -> (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class CausalSelfAttention(nn.Module):
    """All heads in one go: heads are a tensor dimension, not a Python loop.

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
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
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


def positions(start_pos: int | Tensor, T: int, device: torch.device) -> Tensor:
    """Positions of a T-token chunk: (1, T) for one int start, (B, T) for a start per row."""
    # (B, 1) for a start per row, (1, 1) for one shared start
    starts = start_pos.view(-1, 1) if isinstance(start_pos, Tensor) else torch.tensor([[start_pos]])
    return starts.to(device) + torch.arange(T, device=device)


class GroupedQueryAttention(nn.Module):
    """GQA: n_head query heads share n_kv_head key/value heads.

    n_kv_head == n_head is model.CausalSelfAttention (MHA); n_kv_head == 1 is multi-query (MQA).
    Fewer k/v heads -> a smaller KV cache when serving. Llama 3 8B: 32 query heads, 8 k/v heads.
    """

    tril: Tensor

    def __init__(
        self, n_embd: int, n_head: int, n_kv_head: int, ctx_len: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head={n_head} must be divisible by n_kv_head={n_kv_head}")
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_size = n_embd // n_head
        kv_dim = n_kv_head * self.head_size  # k (and v) width: smaller than n_embd when grouped
        # output = [q (n_embd) | k (kv_dim) | v (kv_dim)]; same shape as c_attn when not grouped
        self.c_attn = nn.Linear(n_embd, n_embd + 2 * kv_dim, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(ctx_len, ctx_len)))

    def forward(
        self,
        x: Tensor,
        cache: KVCache | PagedKVCache | None = None,
        layer: int = 0,
        start_pos: int | Tensor = 0,
    ) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd). x holds positions [start_pos, start_pos + T).

        start_pos is one int for the whole batch, or a (B,) tensor: one start per sequence
        (the engine's batched decode).
        """
        B, T, C = x.shape
        hs, nh, nkv = self.head_size, self.n_head, self.n_kv_head
        g = nh // nkv  # query heads per k/v head (the group size)
        # 1. one matmul, uneven split: full-width q, narrow k and v
        # q (B, T, C); k, v (B, T, nkv*hs)
        q, k, v = self.c_attn(x).split([C, nkv * hs, nkv * hs], dim=-1)
        # 2. k, v into their nkv heads -- never copied after this (what the KV cache stores)
        k = k.view(B, T, nkv, hs).transpose(1, 2)  # (B, nkv, T, hs)
        v = v.view(B, T, nkv, hs).transpose(1, 2)  # (B, nkv, T, hs)
        if cache is not None:  # store the new k, v; attend over everything cached so far
            k, v = cache.update(layer, start_pos, k, v)  # (B, nkv, Tk, hs), Tk = start_pos + T
        Tk = k.size(2)  # key positions: T without a cache, start_pos + T with one
        # 3. fold each group's g query heads into one tall stack of g*T rows.
        #    Head h = kv * g + j, so heads 0,1 -> kv 0 and heads 2,3 -> kv 1 (Llama's layout).
        q = q.view(B, T, nkv, g, hs).permute(0, 2, 3, 1, 4)  # (B, nkv, g, T, hs)
        q = q.reshape(B, nkv, g * T, hs)  # (B, nkv, g*T, hs); copies q only (1 token at decode)
        # 4. attention: every row of the stack meets its group's one k/v head
        wei = q @ k.transpose(-2, -1) * hs**-0.5  # (B, nkv, g*T, Tk)
        # mask rows = this chunk's positions, columns = every key position up to the last one
        mask = self.tril[positions(start_pos, T, x.device)][..., :Tk]  # (B or 1, T, Tk)
        # a shorter sequence's padded keys sit past its own position, so this hides them too
        # view as (B, nkv, g, T, Tk) so the (B or 1, T, Tk) mask broadcasts over the heads
        wei = wei.view(B, nkv, g, T, Tk).masked_fill(mask[:, None, None] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1).view(B, nkv, g * T, Tk)  # (B, nkv, g*T, Tk)
        out = wei @ v  # (B, nkv, g*T, hs)
        # 5. unfold back to nh heads, heads side by side, mix
        # (B, nkv, g*T, hs) -> (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        out = out.view(B, nh, T, hs).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))
