"""F5 step 1b target: FeedForward in src/pico_vllm/model.py."""

import pytest
import torch

from pico_vllm.model.gpt import FeedForward

B, T, C = 2, 5, 16


@pytest.fixture
def ffn() -> FeedForward:
    torch.manual_seed(1337)
    return FeedForward(C).eval()


def test_output_shape(ffn: FeedForward) -> None:
    assert ffn(torch.randn(B, T, C)).shape == (B, T, C)


def test_hidden_layer_is_4x_wider(ffn: FeedForward) -> None:
    widths = [m.out_features for m in ffn.modules() if isinstance(m, torch.nn.Linear)]
    assert widths == [4 * C, C]


def test_each_token_is_processed_alone(ffn: FeedForward) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 2] = torch.randn(B, C)  # change token 2 only
    out, out2 = ffn(x), ffn(x2)
    others = [0, 1, 3, 4]
    assert torch.allclose(out[:, others], out2[:, others])  # past AND future untouched
    assert not torch.allclose(out[:, 2], out2[:, 2])
