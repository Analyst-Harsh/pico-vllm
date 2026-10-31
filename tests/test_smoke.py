"""Keeps CI green before the first real test exists (pytest exits 5 when nothing runs)."""

import torch

import pico_vllm


def test_package_and_torch_import() -> None:
    assert pico_vllm.__name__ == "pico_vllm"
    assert torch.ones(2, 3).sum().item() == 6
