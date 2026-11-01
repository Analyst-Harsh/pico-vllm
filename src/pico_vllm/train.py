"""F4: train a model on tiny-shakespeare.

Run: uv run python -m pico_vllm.train --model {bigram,head}
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from pico_vllm.data import CharTokenizer, get_batch, train_val_split
from pico_vllm.model import BigramLanguageModel, SingleHeadLanguageModel

SEED = 1337
B, T = 32, 8  # T is also the model's ctx_len
EVAL_ITERS = 200
N_EMBD = 32
# per model: (max_iters, lr). Attention trains slower, so smaller lr and more steps.
CONFIGS = {"bigram": (3000, 1e-2), "head": (5000, 1e-3)}


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model: nn.Module, splits: dict[str, Tensor], device: str) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIGS, default="head")
    name = parser.parse_args().model
    max_iters, lr = CONFIGS[name]
    eval_interval = max_iters // 10

    torch.manual_seed(SEED)
    device = pick_device()
    text = Path("data/input.txt").read_text()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = train_val_split(data)
    splits = {"train": train, "val": val}

    if name == "bigram":
        model: nn.Module = BigramLanguageModel(tok.vocab_size)
    else:
        model = SingleHeadLanguageModel(tok.vocab_size, n_embd=N_EMBD, ctx_len=T)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, splits, device)
            print(f"step {step:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        x, y = get_batch(train, B, T, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # id 0 = '\n'
    print(tok.decode(model.generate(prompt, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
