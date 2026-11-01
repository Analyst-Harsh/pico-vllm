"""F5 step 1c: Block (pre-norm, residuals) in src/pico_vllm/model.py."""

import pytest
import torch

from pico_vllm.model import Block

B, T, C, NH, CTX = 2, 6, 16, 4, 8


@pytest.fixture
def block() -> Block:
    torch.manual_seed(1337)
    return Block(n_embd=C, n_head=NH, ctx_len=CTX).eval()


def test_output_shape(block: Block) -> None:
    assert block(torch.randn(B, T, C)).shape == (B, T, C)


def test_causal(block: Block) -> None:
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, C)
    assert torch.allclose(block(x)[:, :3], block(x2)[:, :3])


def test_residual_block_with_silent_sublayers_is_identity(block: Block) -> None:
    # zero the last layer of each sub-layer -> both add 0 -> the residual path returns x
    last_layers = [block.sa.proj, block.ffwd.net[2]]
    with torch.no_grad():
        for layer in last_layers:
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(B, T, C)
    assert torch.allclose(block(x), x)
