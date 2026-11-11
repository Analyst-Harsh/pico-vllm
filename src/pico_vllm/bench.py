"""v9 B3: static vs continuous batching on one seeded synthetic workload.

Run: uv run python -m pico_vllm.bench [--checkpoint gpt-m-gqa2] [--requests 64] [--rate 0.5]

Metrics are in engine steps (DESIGN "Metrics"); wall-clock is indicative only.
"""

import argparse
import random
import time
from pathlib import Path

import torch

from pico_vllm.engine import Engine
from pico_vllm.modern import ModernGPT
from pico_vllm.train import build


def workload(
    n: int, rate: float, vocab: int, ctx_len: int, seed: int = 0
) -> list[tuple[list[int], int, int]]:
    """(prompt, max_new_tokens, arrival_step). ~rate arrivals per step; 70% short, 30% long."""
    rng = random.Random(seed)  # noqa: S311  (a seeded workload, not security)
    reqs, t = [], 0.0
    for _ in range(n):
        t += rng.expovariate(rate)  # Poisson arrivals
        prompt_len = rng.randint(4, 16)
        short = rng.random() < 0.7
        new = rng.randint(4, 16) if short else rng.randint(32, ctx_len - prompt_len)
        reqs.append(([rng.randrange(vocab) for _ in range(prompt_len)], new, int(t)))
    return reqs


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run(model: ModernGPT, reqs: list, policy: str, num_blocks: int, max_num_seqs: int) -> dict:
    eng = Engine(
        model, num_blocks=num_blocks, page_size=16, max_num_seqs=max_num_seqs, policy=policy
    )
    seqs = [eng.add_request(p, n, a) for p, n, a in reqs]
    t0 = time.perf_counter()
    eng.run()
    wall = time.perf_counter() - t0
    busy = [st for st in eng.stats if st.tokens or st.held]
    ttft = [s.ttft for s in seqs if s.ttft is not None]
    tpot = [s.tpot for s in seqs if s.tpot is not None]
    slots = sum(st.tokens + st.held for st in busy)
    return {
        "policy": policy,
        "pages": num_blocks,
        "steps": eng.step_no,
        "tok/step": sum(st.tokens for st in busy) / len(busy),
        "util": sum(st.tokens for st in busy) / (len(busy) * max_num_seqs),
        "padding": sum(st.held for st in busy) / slots,
        "TTFT p50": pct(ttft, 0.5),
        "TTFT p99": pct(ttft, 0.99),
        "TPOT p50": pct(tpot, 0.5),
        "TPOT p99": pct(tpot, 0.99),
        "preempt": eng.preemptions,
        "peak pages": max(st.blocks_used for st in eng.stats),
        "wall s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="gpt-m-gqa2")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(Path("checkpoints") / f"{args.checkpoint}.pt")
    model = build(args.checkpoint, len(ck["chars"]), ck["config"])
    model.load_state_dict(ck["model"])
    if not isinstance(model, ModernGPT):
        raise SystemExit(
            "the engine needs a ModernGPT checkpoint (gpt-m-csa, gpt-m-rms, gpt-m-gqa*)"
        )
    reqs = workload(args.requests, args.rate, len(ck["chars"]), model.ctx_len)
    full = args.max_num_seqs * model.ctx_len // 16  # enough pages for every slot at full length
    runs = [
        run(model, reqs, "static", full, args.max_num_seqs),
        run(model, reqs, "continuous", full, args.max_num_seqs),
        run(model, reqs, "continuous", full // 3, args.max_num_seqs),  # memory-tight
    ]
    cols = list(runs[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values()]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
