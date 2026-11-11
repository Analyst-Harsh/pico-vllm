"""What this file teaches: the two warm-up language models.

Key idea: the bigram predicts the next character from the current one alone (one table lookup);
adding a single attention head lets each token look back at its past, and the loss drops.

Read next: model/attention.py (that head, then the versions real models use).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pico_vllm.model.attention import Head


class SingleHeadLanguageModel(nn.Module):
    """Token + position embeddings -> one attention head -> lm_head."""

    def __init__(self, vocab_size: int, n_embd: int, ctx_len: int) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(ctx_len, n_embd)
        self.sa_head = Head(n_embd, head_size=n_embd, ctx_len=ctx_len)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) -> logits (B, T, vocab_size), and loss if targets (B, T) are given."""
        _, T = idx.shape
        tok = self.token_embedding_table(idx)  # (B, T, C)  what each token is
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C) where
        x = self.sa_head(tok + pos)  # (B, T, C)  each token mixes in its past
        logits = self.lm_head(x)  # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        # cross_entropy wants (N, V) logits and (N,) targets: flatten batch and time
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))  # (B*T, V), (B*T,)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """Same loop as the bigram, but only the last ctx_len tokens fit the position table."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx_len :])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)  # (B, T + 1)
        return idx


class BigramLanguageModel(nn.Module):
    """Next token depends only on the current token: one table lookup gives the logits."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) -> logits (B, T, vocab_size), and loss if targets (B, T) are given."""
        logits = self.token_embedding_table(idx)  # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        # cross_entropy wants (N, V) logits and (N,) targets: flatten batch and time
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))  # (B*T, V), (B*T,)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """idx (B, T) -> (B, T + max_new_tokens), sampling one token at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, V): only the last position
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
