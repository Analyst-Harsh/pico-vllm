"""What this file teaches: what the KV cache and the serving engine look like while they run.

Two terminal scenes (they're also the demo video):
  race   the same text written without and with the KV cache. Both runs are timed for real,
         then replayed side by side, slowed down by --slowmo so the eye can follow.
  serve  a live board of requests flowing through the Engine: a few run at once, finished
         ones leave and waiting ones take their slot, and a shared prompt start is reused.

Run:
    uv run python -m pico_vllm.demo race
    uv run python -m pico_vllm.demo serve

Read next: generate.py (the race without the show), serving/engine.py (what serve drives).
"""

import argparse
import shutil
import sys
import textwrap
import time

import torch

from pico_vllm.generate import load, positions_processed, run
from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.serving.engine import Engine
from pico_vllm.serving.sequence import Status
from pico_vllm.train import CHECKPOINT_DIR

BOLD, DIM, RED, GREEN, YELLOW, BLUE, CYAN, RESET = (
    f"\033[{c}m" for c in (1, 2, 31, 32, 33, 34, 36, 0)
)
HOME, CLEAR_DOWN, HIDE_CURSOR, SHOW_CURSOR = "\033[H", "\033[J", "\033[?25l", "\033[?25h"
X = "\u00d7"  # the multiplication sign


def draw(lines: list[str]) -> None:
    """Repaint the whole screen from the top: one write per frame, so there's no flicker."""
    sys.stdout.write(HOME + "\n".join(f"{line}\033[K" for line in lines) + CLEAR_DOWN)
    sys.stdout.flush()


def tail(text: str, width: int, height: int) -> list[str]:
    """The last `height` screen lines of text wrapped at `width` (padded, so panels keep size)."""
    rows = [r for line in text.split("\n") for r in (textwrap.wrap(line, width) or [""])]
    return (rows[-height:] + [""] * height)[:height]


def bar(frac: float, cells: int) -> str:
    full = round(min(1.0, frac) * cells)
    return (
        "\u2501" * full + f"{DIM}" + "\u2500" * (cells - full) + RESET
    )  # thin, so bars stay apart


# ---- scene 1: the KV cache race ------------------------------------------------------------


def timed(model: ModernGPT, idx: torch.Tensor, n: int, cached: bool, seed: int) -> tuple:
    """Generate n tokens; return the ids and the moment (s from start) each one appeared."""
    stamps: list[float] = []
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    ids, _ = run(model, idx, n, cached, 0.8, None, lambda _: stamps.append(time.perf_counter()))
    return ids, [t - t0 for t in stamps]


def race(
    model: ModernGPT, chars: dict[int, str], prompt: str, slowmo: float, fps: int, wait: bool
) -> dict:
    stoi = {c: i for i, c in chars.items()}
    idx = torch.tensor([[stoi[c] for c in prompt]], device=next(model.parameters()).device)
    p, n = idx.size(1), model.ctx_len - idx.size(1)
    timed(model, idx, 4, True, 0)  # warm-up, so neither side pays first-call overhead
    timed(model, idx, 4, False, 0)
    runs = [timed(model, idx, n, cached, seed=7) for cached in (False, True)]
    work = [positions_processed(p, n, model.ctx_len, c) for c in (False, True)]
    width, rows = shutil.get_terminal_size()
    panel = max(3, (rows - 11) // 2)  # caption 2 + 2x(3 header + 1 gap) + result 1
    sides = [("without KV cache", "re-reads everything", RED),
             ("with KV cache", "reads only the new letter", GREEN)]  # fmt: skip
    end = max(stamps[-1] for _, stamps in runs) * slowmo
    if wait:
        input()  # recording: start the replay on a key press, after loading and timing
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter() - t0
        lines = [
            f"{DIM}same model \u00b7 same text \u00b7 replayed {slowmo:g}{X} slower{RESET}",
            "",
        ]
        for (title, how, color), (ids, stamps), cached in zip(
            sides, runs, (False, True), strict=True
        ):
            k = sum(s * slowmo <= now for s in stamps)  # letters visible so far
            read = positions_processed(p, k, model.ctx_len, cached) if k else 0
            how = f"{RESET}{color}\u2713 done in {stamps[-1]:.2f} s" if k == n else how
            lines.append(f"{BOLD}{color}{title}{RESET} {DIM}\u00b7 {how}{RESET}")
            lines.append(f"{color}{bar(k / n, 16)} {BOLD}{k:>3}/{n}{RESET} letters written")
            lines.append(f"{color}{bar(read / work[0], 16)} {BOLD}{read:>6,}{RESET} letters read")
            text = prompt + "".join(chars[i] for i in ids[:k])
            lines += [*tail(text, width - 1, panel), ""]
        if now > end:
            speedup = runs[0][1][-1] / runs[1][1][-1]
            same = "same text \u2713" if runs[0][0] == runs[1][0] else f"{RED}texts differ"
            lines.append(f"{BOLD}{speedup:.1f}{X} faster{RESET} {DIM}\u00b7 {same}{RESET}")
        draw(lines)
        if now > end:
            break
        time.sleep(1 / fps)
    slow, fast = runs[0][1][-1], runs[1][1][-1]
    return {"identical": runs[0][0] == runs[1][0], "speedup": slow / fast, "work": work}


# ---- scene 2: the live serving board -------------------------------------------------------

SHARED = "A gentle scene, by moonlight; and they speak:\n"  # every request starts the same way
SPEAKERS = ["ROMEO", "JULIET", "HAMLET", "PORTIA", "OTHELLO", "VIOLA", "PROSPERO", "MIRANDA"]


def serve(
    model: ModernGPT, chars: dict[int, str], delay: float, wait: bool, seed: int = 3
) -> Engine:
    stoi = {c: i for i, c in chars.items()}
    torch.manual_seed(seed)
    eng = Engine(model, num_blocks=48, page_size=16, max_num_seqs=4, temperature=0.8,
                 prefix_caching=True)  # fmt: skip
    seqs = [
        eng.add_request([stoi[c] for c in f"{SHARED}{name}:\n"], 40 + 17 * (i % 3), 6 * i)
        for i, name in enumerate(SPEAKERS)
    ]
    width = shutil.get_terminal_size().columns - 1
    if wait:
        input()
    dot = {Status.PENDING: DIM, Status.WAITING: YELLOW, Status.RUNNING: CYAN,
           Status.FINISHED: GREEN}  # fmt: skip
    legend = "  ".join(f"{dot[s]}\u25cf{RESET} {DIM}{w}{RESET}" for s, w in
                       [(Status.WAITING, "waiting"), (Status.RUNNING, "writing"),
                        (Status.FINISHED, "done")])  # fmt: skip
    while eng.has_work:
        eng.step()
        alloc = eng.cache.allocator
        used = alloc.num_blocks - alloc.num_free
        pages = "\u25ae" * (used // 2) + f"{DIM}" + "\u25af" * ((alloc.num_blocks - used) // 2)
        lines = [
            f"{DIM}live engine \u00b7 8 requests \u00b7 4 at a time{RESET}",
            "",
            f"{BOLD}step {eng.step_no:>3}{RESET} \u00b7 writing {len(eng.running)}/4"
            f" \u00b7 waiting {len(eng.waiting)}",
            f"memory {CYAN}{pages}{RESET} {used:>2}/{alloc.num_blocks} pages",
            f"{BOLD}{BLUE}shared prompt reused: {eng.prefill_cached} letters{RESET}",
            "",
            legend,
            "",
        ]
        room = max(10, width - 12)
        for s, name in zip(seqs, SPEAKERS, strict=True):
            text = "".join(chars[i] for i in s.output).replace("\n", " / ")
            if len(text) > room:  # keep the end, cut at a word boundary
                text = text[-room:].split(" ", 1)[-1]
            lines.append(f"{dot[s.status]}\u25cf{RESET} {BOLD}{name:<9}{RESET}{text}")
        draw(lines)
        time.sleep(delay)
    return eng


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the KV cache and the engine at work.")
    ap.add_argument("scene", choices=["race", "serve"])
    ap.add_argument("--checkpoint", default="gpt-l-gqa")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--slowmo", type=float, default=8.0, help="race: replay this many x slower")
    ap.add_argument("--delay", type=float, default=0.06, help="serve: seconds per engine step")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wait", action="store_true", help="start on Enter (for recording)")
    args = ap.parse_args()

    model, chars = load(CHECKPOINT_DIR / f"{args.checkpoint}.pt", args.device)
    sys.stdout.write(HIDE_CURSOR)
    try:
        if args.scene == "race":
            race(model, chars, args.prompt, args.slowmo, 30, args.wait)
            if not args.wait:  # recording: a newline here could scroll the last frame
                print()
        else:
            serve(model, chars, args.delay, args.wait)
    finally:
        sys.stdout.write(SHOW_CURSOR)


if __name__ == "__main__":
    main()
