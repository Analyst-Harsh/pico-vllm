"""F4 step 1 target: src/pico_vllm/data.py."""

from pathlib import Path

import pytest
import torch

from pico_vllm.data import CharTokenizer, get_batch, train_val_split

TEXT = "hello world\nto be or not"


def test_roundtrip() -> None:
    tok = CharTokenizer(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_sorted_unique_chars() -> None:
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))
    assert tok.encode(sorted(set(TEXT))[0]) == [0]  # smallest char gets id 0


def test_split_is_90_10_and_in_order() -> None:
    data = torch.arange(100)
    train, val = train_val_split(data)
    assert len(train) == 90
    assert len(val) == 10
    assert torch.equal(torch.cat([train, val]), data)  # no shuffling


def test_get_batch_shapes_and_shift() -> None:
    data = torch.arange(50)  # ids == positions, so the shift is easy to check
    x, y = get_batch(data, B=4, T=8, device="cpu")
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    assert torch.equal(y, x + 1)  # y is x shifted by one


def test_get_batch_stays_in_bounds() -> None:
    data = torch.arange(9)  # exactly T + 1 tokens: the only valid window starts at 0
    x, y = get_batch(data, B=3, T=8, device="cpu")
    assert torch.equal(x[0], data[:8])
    assert torch.equal(y[0], data[1:])


@pytest.mark.skipif(not Path("data/input.txt").exists(), reason="run `make data` first")
def test_shakespeare_vocab_size() -> None:
    text = Path("data/input.txt").read_text()
    assert CharTokenizer(text).vocab_size == 65  # your prediction
