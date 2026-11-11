"""F4 step 2 target: BigramLanguageModel in src/pico_vllm/model.py."""

import pytest
import torch

from pico_vllm.model.baselines import BigramLanguageModel, SingleHeadLanguageModel

V, B, T = 10, 2, 5


@pytest.fixture
def model() -> BigramLanguageModel:
    torch.manual_seed(1337)
    return BigramLanguageModel(V)


def test_forward_without_targets(model: BigramLanguageModel) -> None:
    idx = torch.randint(V, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, V)  # one score per vocab token, at every position
    assert loss is None


def test_forward_with_targets_gives_finite_scalar_loss(model: BigramLanguageModel) -> None:
    idx = torch.randint(V, (B, T))
    targets = torch.randint(V, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, V)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_generate_appends_tokens(model: BigramLanguageModel) -> None:
    idx = torch.zeros((B, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=7)
    assert out.shape == (B, 1 + 7)
    assert torch.equal(out[:, :1], idx)  # the prompt is kept
    assert out.min() >= 0
    assert out.max() < V


def test_single_head_model_shapes_and_long_generation() -> None:
    torch.manual_seed(1337)
    m = SingleHeadLanguageModel(V, n_embd=8, ctx_len=4)
    idx = torch.randint(V, (B, 4))
    logits, loss = m(idx, idx)
    assert logits.shape == (B, 4, V)
    assert torch.isfinite(loss)
    out = m.generate(idx[:, :1], max_new_tokens=10)  # runs past ctx_len: must crop
    assert out.shape == (B, 11)
