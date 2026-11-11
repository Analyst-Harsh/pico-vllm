"""What this file teaches: a GPT-2-style transformer, trained from scratch.

Key idea: a Block is communicate (attention, tokens talk) then compute (feed-forward, each token
alone), both with pre-norm and a residual. Stack n_layer Blocks between the embeddings and
lm_head and you have a GPT.

Read next: sampler.py (turning logits into the next token), then model/llama_style.py.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm.model.attention import MultiHeadAttention
from pico_vllm.sampler import sample_next


class FeedForward(nn.Module):
    """Per-token MLP: widen 4x, ReLU, narrow back. Tokens don't talk here; each thinks alone."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # widen
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # narrow back (the "projection")
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd)."""
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (feed-forward).

    Pre-norm: LayerNorm goes *before* each sub-layer, and each result is added back to x.
    """

    def __init__(self, n_embd: int, n_head: int, ctx_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, ctx_len, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, n_embd) -> (B, T, n_embd)."""
        x = x + self.sa(self.ln1(x))  # tokens talk; the residual keeps the original x
        return x + self.ffwd(self.ln2(x))  # each token thinks; residual again


class GPT(nn.Module):
    """Token + position embeddings -> n_layer Blocks -> final LayerNorm -> lm_head."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        ctx_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(ctx_len, n_embd)
        self.blocks = nn.Sequential(
            *(Block(n_embd, n_head, ctx_len, dropout) for _ in range(n_layer))
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final norm before predicting
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small random starts (std 0.02, as in GPT-2) keep early logits near uniform."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) -> logits (B, T, vocab_size), and loss if targets (B, T) are given."""
        _, T = idx.shape
        tok = self.token_embedding_table(idx)  # (B, T, C)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.blocks(tok + pos)  # (B, T, C)  gather -> think, n_layer times
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        # cross_entropy wants (N, V) logits and (N,) targets: flatten batch and time
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))  # (B*T, V), (B*T,)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Crop to ctx_len, predict, sample, append. Recomputes the whole window every token;
        ModernGPT.generate_cached removes that work with a KV cache."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx_len :])
            idx_next = sample_next(logits[:, -1, :], temperature, top_k, generator)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
