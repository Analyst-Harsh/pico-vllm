"""The sampler: temperature, top-k, greedy (sampler.py)."""

import pytest
import torch

from pico_vllm.sampler import next_token_probs, sample_next

LOGITS = torch.tensor(
    [
        [1.0, 3.0, 0.5, 2.0, -1.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
    ]
)


def test_probs_rows_are_valid_distributions() -> None:
    for t in (0.5, 1.0, 2.0):
        p = next_token_probs(LOGITS, temperature=t)
        assert p.shape == LOGITS.shape
        assert (p >= 0).all()
        assert torch.allclose(p.sum(-1), torch.ones(2))


def test_temperature_one_is_plain_softmax() -> None:
    assert torch.allclose(next_token_probs(LOGITS), LOGITS.softmax(-1))


def test_low_temperature_sharpens_high_flattens() -> None:
    top = LOGITS.argmax(-1, keepdim=True)
    p_low = next_token_probs(LOGITS, temperature=0.5).gather(-1, top)
    p_mid = next_token_probs(LOGITS, temperature=1.0).gather(-1, top)
    p_high = next_token_probs(LOGITS, temperature=2.0).gather(-1, top)
    assert (p_low > p_mid).all()
    assert (p_mid > p_high).all()


def test_top_k_zeroes_everything_outside_the_k() -> None:
    p = next_token_probs(LOGITS, top_k=2)
    assert torch.equal(p[0] > 0, torch.tensor([False, True, False, True, False]))  # 3.0, 2.0
    assert torch.allclose(p.sum(-1), torch.ones(2))


def test_top_k_larger_than_vocab_is_harmless() -> None:
    assert torch.allclose(next_token_probs(LOGITS, top_k=100), LOGITS.softmax(-1))


def test_temperature_zero_is_greedy() -> None:
    assert torch.equal(sample_next(LOGITS, temperature=0.0), torch.tensor([[1], [2]]))


def test_samples_never_leave_top_k() -> None:
    g = torch.Generator().manual_seed(1337)
    ids = torch.cat([sample_next(LOGITS, top_k=2, generator=g) for _ in range(500)], dim=1)
    assert set(ids[0].tolist()) <= {1, 3}


def test_sample_shape_and_seeded_reproducibility() -> None:
    a = sample_next(LOGITS, generator=torch.Generator().manual_seed(7))
    b = sample_next(LOGITS, generator=torch.Generator().manual_seed(7))
    assert a.shape == (2, 1)
    assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs", [{"temperature": -1.0}, {"top_k": 0}])
def test_rejects_bad_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"temperature|top_k"):
        next_token_probs(LOGITS, **kwargs)
