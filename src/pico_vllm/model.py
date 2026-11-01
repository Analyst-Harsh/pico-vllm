"""F4: language models, starting with the bigram baseline."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class SingleHeadLanguageModel(nn.Module):
    """Token + position embeddings -> one attention head -> lm_head. F4 step 4."""

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
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """Same loop as the bigram, but only the last ctx_len tokens fit the position table."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx_len :])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
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
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
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
