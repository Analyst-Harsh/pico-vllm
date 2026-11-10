"""F6: Llama-style building blocks (GQA, RMSNorm) on top of model.py.

model.py keeps the F4-F5 GPT unchanged, so old checkpoints and experiments still load.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm import model
from pico_vllm.kv_cache import KVCache, PagedKVCache
from pico_vllm.sample import sample_next


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

    def forward(
        self,
        x: Tensor,
        cache: KVCache | PagedKVCache | None = None,
        layer: int = 0,
        start_pos: int = 0,
    ) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd). x holds positions [start_pos, start_pos + T)."""
        B, T, C = x.shape
        hs, nh, nkv = self.head_size, self.n_head, self.n_kv_head
        g = nh // nkv  # query heads per k/v head (the group size)
        # 1. one matmul, uneven split: full-width q, narrow k and v
        q, k, v = self.c_attn(x).split([C, nkv * hs, nkv * hs], dim=-1)
        # 2. k, v into their nkv heads -- never copied after this (what v9 caches)
        k = k.view(B, T, nkv, hs).transpose(1, 2)  # (B, nkv, T, hs)
        v = v.view(B, T, nkv, hs).transpose(1, 2)
        if cache is not None:  # B1: store the new k, v; attend over everything cached so far
            k, v = cache.update(layer, start_pos, k, v)  # (B, nkv, Tk, hs), Tk = start_pos + T
        Tk = k.size(2)  # key positions: T without a cache, start_pos + T with one
        # 3. fold each group's g query heads into one tall stack of g*T rows.
        #    Head h = kv * g + j, so heads 0,1 -> kv 0 and heads 2,3 -> kv 1 (Llama's layout).
        q = q.view(B, T, nkv, g, hs).permute(0, 2, 3, 1, 4)  # (B, nkv, g, T, hs)
        q = q.reshape(B, nkv, g * T, hs)  # copies q only (1 token at decode)
        # 4. attention: every row of the stack meets its group's one k/v head
        wei = q @ k.transpose(-2, -1) * hs**-0.5  # (B, nkv, g*T, Tk)
        # mask rows = this chunk's positions, columns = every key position up to the last one
        mask = self.tril[start_pos : start_pos + T, :Tk]  # (T, Tk)
        wei = wei.view(B, nkv, g, T, Tk).masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1).view(B, nkv, g * T, Tk)
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

    def forward(
        self,
        x: Tensor,
        cache: KVCache | PagedKVCache | None = None,
        layer: int = 0,
        start_pos: int = 0,
    ) -> Tensor:
        """Same as Block, plus the cache handed down to attention."""
        x = x + self.sa(self.ln1(x), cache, layer, start_pos)
        return x + self.ffwd(self.ln2(x))


class ModernGPT(model.GPT):
    """F5 GPT built from ModernBlocks, plus a KV cache (v9 B1). Same init and generate."""

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

    def forward(  # type: ignore[override]  # adds cache and start_pos
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: KVCache | PagedKVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) holds positions [start_pos, start_pos + T). No cache: start_pos is 0."""
        _, T = idx.shape
        if start_pos + T > self.ctx_len:
            raise ValueError(f"positions up to {start_pos + T} exceed ctx_len={self.ctx_len}")
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)  # real positions
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for layer, block in enumerate(self.blocks):
            x = block(x, cache, layer, start_pos)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        return logits, F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    def new_cache(self, batch_size: int, device: torch.device | str) -> KVCache:
        """An empty KVCache sized for this model."""
        sa = self.blocks[0].sa
        return KVCache(
            len(self.blocks), batch_size, sa.n_kv_head, self.ctx_len, sa.head_size, device
        )

    @torch.no_grad()
    def generate_cached(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        cache: KVCache | PagedKVCache | None = None,
    ) -> Tensor:
        """Prefill the prompt once, then feed one token per step. v9 B1.

        Learned positions: prompt + new tokens must fit in ctx_len (no cropping like generate).
        cache: an empty cache to fill (B2 passes a PagedKVCache); default a contiguous one.
        """
        B, T = idx.shape
        if T + max_new_tokens > self.ctx_len:
            raise ValueError(f"prompt {T} + {max_new_tokens} new > ctx_len={self.ctx_len}")
        cache = cache or self.new_cache(B, idx.device)
        logits, _ = self(idx, cache=cache, start_pos=0)  # prefill: whole prompt, fills the cache
        for i in range(max_new_tokens):
            idx_next = sample_next(logits[:, -1, :], temperature, top_k, generator)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            if i < max_new_tokens - 1:  # decode: only the new token goes through the model
                logits, _ = self(idx_next, cache=cache, start_pos=T + i)
        return idx
