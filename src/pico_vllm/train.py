"""F4: train a model on tiny-shakespeare.  Run: uv run python -m pico_vllm.train"""

from pathlib import Path

import torch
from torch import Tensor, nn

from pico_vllm.data import CharTokenizer, get_batch, train_val_split
from pico_vllm.model import BigramLanguageModel

SEED = 1337
B, T = 32, 8
MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS = 3000, 300, 200
LR = 1e-2


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
    torch.manual_seed(SEED)
    device = pick_device()
    text = Path("data/input.txt").read_text()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = train_val_split(data)
    splits = {"train": train, "val": val}

    model = BigramLanguageModel(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
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
