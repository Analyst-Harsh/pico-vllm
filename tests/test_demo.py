"""demo.py: both scenes run end to end on a tiny random model."""

import string

import torch

from pico_vllm.demo import race, serve
from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.serving.sequence import Status

CHARS = dict(enumerate("\n" + string.printable[:95]))


def tiny(ctx_len: int) -> ModernGPT:
    torch.manual_seed(0)
    return ModernGPT(len(CHARS), 16, 4, 2, ctx_len, rmsnorm=True, n_kv_head=2).eval()


def test_race_same_text_less_work() -> None:
    r = race(tiny(32), CHARS, "ROMEO:", slowmo=0.01, fps=1000, wait=False)
    assert r["identical"]
    assert r["work"][0] > r["work"][1] == 32 - 1  # prompt + new tokens, minus the last


def test_serve_finishes_every_request_and_reuses_the_shared_start() -> None:
    eng = serve(tiny(128), CHARS, delay=0, wait=False)
    assert len(eng.finished) == 8
    assert all(s.status is Status.FINISHED for s in eng.finished)
    assert eng.prefill_cached > 0
