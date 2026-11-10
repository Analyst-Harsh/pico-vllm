"""F6: Llama-style building blocks (GQA, RMSNorm) on top of model.py.

model.py keeps the F4-F5 GPT unchanged, so old checkpoints and experiments still load.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm import model


class GroupedQueryAttention(nn.Module):
    """GQA: n_head query heads share n_kv_head key/value heads. F6 step 3.

    n_kv_head == n_head is model.CausalSelfAttention (MHA); n_kv_head == 1 is multi-query (MQA).
    Fewer k/v heads -> a smaller KV cache in v9. Llama 3 8B: 32 query heads, 8 k/v heads.
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

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd)."""
        B, T, C = x.shape
        hs, nh, nkv = self.head_size, self.n_head, self.n_kv_head
        g = nh // nkv  # query heads per k/v head (the group size)
        # 1. one matmul, uneven split: full-width q, narrow k and v
        q, k, v = self.c_attn(x).split([C, nkv * hs, nkv * hs], dim=-1)
        # 2. k, v into their nkv heads -- never copied after this (what v9 caches)
        k = k.view(B, T, nkv, hs).transpose(1, 2)  # (B, nkv, T, hs)
        v = v.view(B, T, nkv, hs).transpose(1, 2)
        # 3. fold each group's g query heads into one tall stack of g*T rows.
        #    Head h = kv * g + j, so heads 0,1 -> kv 0 and heads 2,3 -> kv 1 (Llama's layout).
        q = q.view(B, T, nkv, g, hs).permute(0, 2, 3, 1, 4)  # (B, nkv, g, T, hs)
        q = q.reshape(B, nkv, g * T, hs)  # copies q only (1 token at decode)
        # 4. attention: every row of the stack meets its group's one k/v head
        wei = q @ k.transpose(-2, -1) * hs**-0.5  # (B, nkv, g*T, T)
        wei = wei.view(B, nkv, g, T, T).masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1).view(B, nkv, g * T, T)
        out = wei @ v  # (B, nkv, g*T, hs)
        # 5. unfold back to nh heads, heads side by side, mix
        out = out.view(B, nh, T, hs).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class RMSNorm(nn.Module):
    """LayerNorm without the mean: only rescale each token to root-mean-square 1. F6 step 2.

    Llama uses this. One learned weight per channel, no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # starts as "change nothing"

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, dim) -> (B, T, dim)."""
        # What it computes (hand-written, ~6 kernels; EXPERIMENTS 2026-11-10):
        #   inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, 1)
        #   return x * inv_rms * self.weight
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)  # same maths, one call


class ModernBlock(model.Block):
    """F5 Block with batched GQA attention (step 1/3) and optional RMSNorm (step 2)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        ctx_len: int,
        dropout: float = 0.0,
        rmsnorm: bool = False,
        n_kv_head: int | None = None,
    ) -> None:
        super().__init__(n_embd, n_head, ctx_len, dropout)
        # n_kv_head None -> n_head: plain MHA, same weights as CausalSelfAttention, so step-1/2
        # checkpoints still load
        self.sa = GroupedQueryAttention(n_embd, n_head, n_kv_head or n_head, ctx_len, dropout)
        if rmsnorm:  # step 2; off by default so step-1 checkpoints still load
            self.ln1, self.ln2 = RMSNorm(n_embd), RMSNorm(n_embd)


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
        rmsnorm: bool = False,
        n_kv_head: int | None = None,
    ) -> None:
        super().__init__(vocab_size, n_embd, n_head, n_layer, ctx_len, dropout)
        self.blocks = nn.Sequential(
            *(
                ModernBlock(n_embd, n_head, ctx_len, dropout, rmsnorm, n_kv_head)
                for _ in range(n_layer)
            )
        )
        if rmsnorm:
            self.ln_f = RMSNorm(n_embd)
        self.apply(self._init_weights)  # the new blocks need the GPT-2 init too
