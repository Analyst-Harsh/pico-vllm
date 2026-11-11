"""What this file teaches: how a saved model is loaded and used to write text, and what the
KV cache buys you.

Key idea: a checkpoint is weights + the config to rebuild the model + the character vocabulary.
Generation feeds the text back in one token at a time; with the KV cache each new token is the
only thing pushed through the model, without it the whole text so far is re-read every time.

Run:
    uv run python -m pico_vllm.generate --checkpoint gpt-l-gqa --prompt "ROMEO:"
    uv run python -m pico_vllm.generate --checkpoint gpt-l-gqa --prompt "ROMEO:" --compare

Read next: serving/kv_cache.py (what the cache stores), then serving/engine.py.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.train import CHECKPOINT_DIR, build


def load(path: Path, device: str) -> tuple[ModernGPT, dict[int, str]]:
    """Rebuild the model from its config, pour the saved weights in, return it with its vocab."""
    if not path.exists():
        raise SystemExit(
            f"no checkpoint at {path}. Train it first:\n"
            f"  uv run python -m pico_vllm.train --model {path.stem}"
        )
    ck = torch.load(path, map_location="cpu")
    model = build(path.stem, len(ck["chars"]), ck["config"])
    if not isinstance(model, ModernGPT):
        raise SystemExit(f"{path.stem} isn't a ModernGPT; try gpt-l-gqa or gpt-m-gqa2")
    model.load_state_dict(ck["model"])
    return model.eval().to(device), ck["chars"]


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()  # GPU work is queued; wait for it before reading the clock


def run(
    model: ModernGPT,
    idx: Tensor,
    n: int,
    cached: bool,
    temperature: float = 0.0,
    top_k: int | None = None,
    on_token: Callable[[int], None] | None = None,
) -> tuple[list[int], float]:
    """Stream n tokens after idx (1, T): on_token(id) runs the moment each one is picked.

    Returns the new ids and the seconds it took.
    """
    sync(str(idx.device))
    t0 = time.perf_counter()
    stream = model.stream_cached if cached else model.stream
    ids = []
    for nxt in stream(idx, n, temperature=temperature, top_k=top_k):
        ids.append(int(nxt))  # int() waits for this token, so streaming also keeps time honest
        if on_token:
            on_token(ids[-1])
    sync(str(idx.device))
    return ids, time.perf_counter() - t0


def positions_processed(prompt_len: int, n: int, ctx_len: int, cached: bool) -> int:
    """How many token positions go through the model to generate n new tokens."""
    if cached:  # prefill the prompt once, then one position per new token (the last isn't fed)
        return prompt_len + n - 1
    return sum(min(prompt_len + i, ctx_len) for i in range(n))  # re-read the window every step


def compare(
    model: ModernGPT,
    idx: Tensor,
    n: int,
    on_token: Callable[[int], None] | None = None,
    on_start: Callable[[str], None] | None = None,
) -> dict:
    """Greedy generation without, then with, the KV cache: same text, different cost."""
    run(model, idx, 4, cached=True)  # warm-up, so neither side pays first-call overhead
    run(model, idx, 4, cached=False)
    if on_start:
        on_start("no cache")
    slow, t_slow = run(model, idx, n, cached=False, on_token=on_token)
    if on_start:
        on_start("KV cache")
    fast, t_fast = run(model, idx, n, cached=True, on_token=on_token)
    p = idx.size(1)
    return {
        "ids": fast,
        "identical": slow == fast,
        "seconds": (t_slow, t_fast),
        "positions": tuple(positions_processed(p, n, model.ctx_len, c) for c in (False, True)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Write text with a trained pico-vllm model.")
    ap.add_argument("--checkpoint", default="gpt-l-gqa")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, help="new characters (default: fill the context)")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-cache", action="store_true", help="re-read the whole text every step")
    ap.add_argument("--compare", action="store_true", help="greedy, with vs without the cache")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    model, chars = load(CHECKPOINT_DIR / f"{args.checkpoint}.pt", args.device)
    stoi = {c: i for i, c in chars.items()}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise SystemExit(f"characters not in the vocabulary: {unknown}")
    idx = torch.tensor([[stoi[c] for c in args.prompt]], device=args.device)
    room = model.ctx_len - idx.size(1)  # learned positions: prompt + output must fit ctx_len
    n = args.tokens or room
    if n > room and not args.no_cache:
        raise SystemExit(f"prompt + {n} new > ctx_len={model.ctx_len}; use --no-cache to go on")

    def show(i: int) -> None:  # one character, on screen the moment it's picked
        print(chars[i], end="", flush=True)

    torch.manual_seed(args.seed)
    if not args.compare:
        print(args.prompt, end="", flush=True)
        _, secs = run(model, idx, n, not args.no_cache, args.temperature, args.top_k, show)
        cache = "off" if args.no_cache else "on"
        print(f"\n\n[{n} tokens in {secs:.2f} s on {args.device}, cache {cache}]")
        return

    def start(label: str) -> None:
        print(f"\n── {label} ──\n{args.prompt}", end="", flush=True)

    r = compare(model, idx, n, on_token=show, on_start=start)
    (t_slow, t_fast), (p_slow, p_fast) = r["seconds"], r["positions"]
    print(f"\n\nsame text with and without the cache: {'yes' if r['identical'] else 'NO'}")
    print(f"{'':22}{'no cache':>12}{'KV cache':>12}")
    print(f"{'time (s)':22}{t_slow:12.2f}{t_fast:12.2f}")
    print(f"{'ms per token':22}{t_slow / n * 1e3:12.1f}{t_fast / n * 1e3:12.1f}")
    print(f"{'positions processed':22}{p_slow:12,}{p_fast:12,}")
    print(f"{'speed-up':22}{'':12}{t_slow / t_fast:11.1f}x")
    print(f"[{n} tokens, greedy, {args.device}; wall-clock is indicative]")


if __name__ == "__main__":
    main()
