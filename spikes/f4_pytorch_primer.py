# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
"""F session 4 primer: just enough PyTorch to read Karpathy's GPT code.

Run: uv run spikes/f4_pytorch_primer.py   (throwaway learning code)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)  # same "random" numbers every run, so output is reproducible


def show(name, t):
    print(f"\n{name}  shape={tuple(t.shape)}\n{t}")


# --- 1. Tensors: numpy arrays that can live on a GPU and track gradients -------
print("=" * 70, "\n1. TENSORS AND (B, T) SHAPES")
B, T = 2, 4  # 2 snippets in a batch, each 4 tokens long
idx = torch.tensor([[5, 2, 7, 1],  # snippet 0: token ids
                    [3, 3, 0, 6]])  # snippet 1
show("idx (token ids)", idx)  # shape (B, T): this is what the data loader yields

# --- 2. nn.Embedding: a learnable lookup table, token id -> vector -------------
print("=" * 70, "\n2. nn.Embedding: id -> vector (adds the C dimension)")
vocab_size, C = 8, 3  # 8 possible tokens, each becomes a 3-number vector
emb = nn.Embedding(vocab_size, C)  # a (vocab_size, C) table of random numbers...
show("embedding table (learned during training)", emb.weight.data)
x = emb(idx)  # ...and calling it just looks up rows
show("x = emb(idx)   <- (B, T) became (B, T, C)", x.data)
print("\nrow for token 3 appears twice in snippet 1 (same id -> same vector):")
print(x[1, 0].data, x[1, 1].data)

# --- 3. nn.Linear: y = x @ W.T + b, applied to the last dim -------------------
print("=" * 70, "\n3. nn.Linear: reshapes the last dim with learned weights")
head_size = 2
proj = nn.Linear(C, head_size, bias=False)  # W is (head_size, C) = learned
y = proj(x)
show("y = proj(x)   <- (B, T, C) became (B, T, head_size)", y.data)
print("\nThis is exactly what Karpathy's key/query/value layers are:")
print("three nn.Linear(C, head_size) -> three different learned views of each token.")

# --- 4. The 'mathematical trick': average over the past with a matrix multiply --
print("=" * 70, "\n4. THE TRICK: each token = average of itself + earlier tokens")
xb = x[0].data  # take snippet 0 only, shape (T, C)

# Version 1: obvious but slow python loop
v1 = torch.stack([xb[: t + 1].mean(dim=0) for t in range(T)])

# Version 2: lower-triangular matrix of weights, one matrix multiply
tril = torch.tril(torch.ones(T, T))
show("tril (1 = 'may look at')", tril)
wei = tril / tril.sum(dim=1, keepdim=True)
show("wei = each row sums to 1 -> 'how much to take from each token'", wei)
v2 = wei @ xb  # (T, T) @ (T, C) -> (T, C)

# Version 3: same thing written with softmax -- this is the form attention uses
scores = torch.zeros(T, T)  # attention will put REAL q.k scores here instead of 0
scores = scores.masked_fill(tril == 0, float("-inf"))  # future = -inf
show("scores after mask (-inf = cannot see the future)", scores)
wei3 = F.softmax(scores, dim=-1)  # exp(-inf)=0, so the future gets weight 0
v3 = wei3 @ xb

print("\nall three versions equal?", torch.allclose(v1, v2), torch.allclose(v1, v3))
print("\nTHE KEY IDEA: in v3, scores are all zeros -> plain average.")
print("Self-attention replaces those zeros with q.k scores, so each token chooses")
print("WHICH past tokens to listen to. That swap is your Head.forward() in session 4.")
