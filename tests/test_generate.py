"""generate.py: a saved checkpoint loads back, and the cache changes the cost, not the text."""

from pathlib import Path

import pytest
import torch

from pico_vllm.generate import compare, load, positions_processed, run
from pico_vllm.model.llama_style import ModernGPT

CFG = {"T": 16, "n_embd": 16, "n_head": 4, "n_kv_head": 2, "n_layer": 2, "dropout": 0.0}
CFG |= {"modern": True, "rmsnorm": True}
CHARS = dict(enumerate("\n abcdefghijklmnopqrstuvwxyz"))


def save_tiny(path: Path) -> ModernGPT:
    torch.manual_seed(1337)
    model = ModernGPT(len(CHARS), 16, 4, 2, 16, rmsnorm=True, n_kv_head=2)
    for p in model.parameters():  # peaky logits so greedy is sensitive to any cache bug
        torch.nn.init.normal_(p, std=0.5)
    torch.save({"model": model.state_dict(), "config": CFG, "chars": CHARS}, path)
    return model


def test_load_round_trips_a_checkpoint(tmp_path: Path) -> None:
    saved = save_tiny(tmp_path / "tiny.pt")
    model, chars = load(tmp_path / "tiny.pt", "cpu")
    assert chars == CHARS
    for a, b in zip(saved.state_dict().values(), model.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_missing_checkpoint_explains_how_to_train(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Train it first"):
        load(tmp_path / "nope.pt", "cpu")


def test_compare_gives_identical_text_and_less_work(tmp_path: Path) -> None:
    save_tiny(tmp_path / "tiny.pt")
    model, _ = load(tmp_path / "tiny.pt", "cpu")
    r = compare(model, torch.tensor([[1, 2, 3]]), 12)
    assert r["identical"]
    assert len(r["ids"]) == 12
    assert r["positions"] == (positions_processed(3, 12, 16, False), 3 + 12 - 1)
    assert r["positions"][0] > r["positions"][1]


def test_positions_processed() -> None:
    assert positions_processed(1, 3, 16, cached=False) == 1 + 2 + 3
    assert positions_processed(1, 3, 2, cached=False) == 1 + 2 + 2  # window capped at ctx_len
    assert positions_processed(1, 3, 16, cached=True) == 3


def test_run_streams_each_token_as_it_is_picked(tmp_path: Path) -> None:
    save_tiny(tmp_path / "tiny.pt")
    model, _ = load(tmp_path / "tiny.pt", "cpu")
    seen: list[int] = []
    ids, _ = run(model, torch.tensor([[1, 2, 3]]), 8, cached=True, on_token=seen.append)
    assert seen == ids
    assert len(ids) == 8
    full = model.generate_cached(torch.tensor([[1, 2, 3]]), 8, temperature=0)
    assert full[0, 3:].tolist() == ids  # streaming yields exactly what generate returns
