"""F6: Llama-style building blocks (GQA, RMSNorm) on top of model.py.

model.py keeps the F4-F5 GPT unchanged, so old checkpoints and experiments still load.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm.model.attention import GroupedQueryAttention, positions
from pico_vllm.model.gpt import GPT, Block
from pico_vllm.sampler import sample_next
from pico_vllm.serving.kv_cache import KVCache
from pico_vllm.serving.paged_kv_cache import PagedKVCache


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


class ModernBlock(Block):
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
        start_pos: int | Tensor = 0,
    ) -> Tensor:
        """Same as Block, plus the cache handed down to attention."""
        x = x + self.sa(self.ln1(x), cache, layer, start_pos)
        return x + self.ffwd(self.ln2(x))


class ModernGPT(GPT):
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
        start_pos: int | Tensor = 0,
    ) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) holds positions [start_pos, start_pos + T). No cache: start_pos is 0.

        start_pos may be a (B,) tensor when the sequences in the batch are at different points.
        """
        _, T = idx.shape
        pos = positions(start_pos, T, idx.device)  # (B or 1, T): real positions
        if int(pos.max()) >= self.ctx_len:
            raise ValueError(f"positions up to {int(pos.max()) + 1} exceed ctx_len={self.ctx_len}")
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
