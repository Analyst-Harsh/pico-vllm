"""F4 step 1: text -> token ids -> (x, y) training batches."""

import torch
from torch import Tensor


class CharTokenizer:
    """Character-level tokenizer: every unique character in the text is one token."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def train_val_split(data: Tensor) -> tuple[Tensor, Tensor]:
    """First 90% is train, last 10% is val. No shuffling."""
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data: Tensor, B: int, T: int, device: str | torch.device) -> tuple[Tensor, Tensor]:
    """B random windows of length T from data. y is x shifted one token to the right."""
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x.to(device), y.to(device)
