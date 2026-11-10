"""F4/F5: train a model on tiny-shakespeare.

Run: uv run python -m pico_vllm.train --model <name in CONFIGS>
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from pico_vllm.data import CharTokenizer, get_batch, train_val_split
from pico_vllm.model import GPT, BigramLanguageModel, SingleHeadLanguageModel
from pico_vllm.modern import ModernGPT

SEED = 1337
EVAL_ITERS = 200
CHECKPOINT_DIR = Path("checkpoints")

# Per model. T (batch window) is also the model's ctx_len.
CONFIGS = {
    "bigram": {"B": 32, "T": 8, "max_iters": 3000, "lr": 1e-2},
    "head": {"B": 32, "T": 8, "max_iters": 5000, "lr": 1e-3, "n_embd": 32},
    "gpt": {
        "B": 32,
        "T": 32,
        "max_iters": 5000,
        "lr": 1e-3,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.0,
    },
}
# F5 scaling, one change at a time (EXPERIMENTS 2026-11-02)
CONFIGS["gpt-drop"] = {**CONFIGS["gpt"], "dropout": 0.2}  # A: same size + dropout
CONFIGS["gpt-m"] = {**CONFIGS["gpt-drop"], "T": 64, "n_embd": 128}  # B: wider, longer ctx
CONFIGS["gpt-m-nodrop"] = {**CONFIGS["gpt-m"], "dropout": 0.0}  # B without dropout
# F5 step 5: head-count sweep (same params; only how attention is split changes)
CONFIGS["gpt-m-h1"] = {**CONFIGS["gpt-m-nodrop"], "n_head": 1}
CONFIGS["gpt-m-h8"] = {**CONFIGS["gpt-m-nodrop"], "n_head": 8}
# F6 step 1: batched heads (modern.py), same configs as the looped runs above
CONFIGS["gpt-m-csa"] = {**CONFIGS["gpt-m-nodrop"], "modern": True}
CONFIGS["gpt-m-csa-h8"] = {**CONFIGS["gpt-m-h8"], "modern": True}
# F6 step 2: RMSNorm in place of every LayerNorm
CONFIGS["gpt-m-rms"] = {**CONFIGS["gpt-m-csa"], "rmsnorm": True}


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(
    model: nn.Module, splits: dict[str, Tensor], B: int, T: int, device: str
) -> dict[str, float]:
    """Average loss over EVAL_ITERS batches for each split. Less noisy than one batch."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, B, T, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def build(
    name: str, vocab_size: int, cfg: dict
) -> BigramLanguageModel | SingleHeadLanguageModel | GPT:
    if name == "bigram":
        return BigramLanguageModel(vocab_size)
    if name == "head":
        return SingleHeadLanguageModel(vocab_size, n_embd=cfg["n_embd"], ctx_len=cfg["T"])
    sizes = {k: cfg[k] for k in ("n_embd", "n_head", "n_layer", "dropout")}
    if cfg.get("modern"):
        return ModernGPT(vocab_size, ctx_len=cfg["T"], rmsnorm=cfg.get("rmsnorm", False), **sizes)
    return GPT(vocab_size, ctx_len=cfg["T"], **sizes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIGS, default="gpt")
    name = parser.parse_args().model
    cfg = CONFIGS[name]
    B, T, max_iters = cfg["B"], cfg["T"], cfg["max_iters"]
    eval_interval = max_iters // 10

    torch.manual_seed(SEED)
    device = pick_device()
    text = Path("data/input.txt").read_text()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = train_val_split(data)
    splits = {"train": train, "val": val}

    model = build(name, tok.vocab_size, cfg).to(device)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} params on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, splits, B, T, device)
            print(f"step {step:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        x, y = get_batch(train, B, T, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if isinstance(model, GPT):  # v9 serves one of these
        path = CHECKPOINT_DIR / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg, "chars": tok.itos}, path)
        print(f"saved {path}")

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # id 0 = '\n'
    print(tok.decode(model.generate(prompt, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
