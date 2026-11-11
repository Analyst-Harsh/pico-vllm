# pico-vllm roadmap

> **My personal learning plan, kept as a record.** The codes (F4–F6, v9, B1–B5) refer to my Fryday roadmap; file paths are as they were at the time (the code now lives in `model/` and `serving/`, see [DESIGN](DESIGN.md#engine-v9)). To study the repo, start from the [README](../README.md).

Every session, in order, as planned. A session is about 90 minutes. ✍️ flags the pieces that matter most.

**Where this sits in Fryday's plan:**

```
F4 → F5 → F6  (here)  →  v9 (here + Fryday)  →  v8 Triton (Fryday) → …     (v7a lite skipped, 2026-11-10)
```

v9 comes before v8 on purpose: Triton is easier to judge once you've built an engine yourself.

| Block | Sessions | Where |
|---|---|---|
| F4 tiny GPT 1/2 | 1 on the roadmap; plan for 2 | here |
| F5 tiny GPT 2/2 | 1 on the roadmap; plan for 2 | here |
| F6 modern architecture | 1 learn + 3 small builds (batched heads, RMSNorm, GQA) | here |
| v9 learn | 3 | here |
| v9 build | 7: B1–B5 here; vllm-metal timebox + OpenAI-contract swap in Fryday | both |
| v9 interview | 1 | Fryday |

---

## F4: tiny GPT 1/2, self-attention in code

**Learn:**
- ⭐ [Karpathy, *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY), from the start through the chapter "version 4: self-attention".
- Warm-up: open `spikes/f4_pytorch_primer.ipynb` (kernel: `.venv`) and run cell by cell (tensors, `(B, T, C)` shapes, `nn.Embedding`, `nn.Linear`, softmax).

**Build:**

| Step | ✍️ What you write | Test to aim at | Predict first |
|---|---|---|---|
| 1 | `data.py`: read `data/input.txt`, a `CharTokenizer` (encode/decode), a 90/10 train/val split, `get_batch(split, B, T, device)` | decode(encode(s)) == s; `y` is `x` shifted by one | the vocab size |
| 2 | `model.py`: `BigramLanguageModel` (forward with optional targets, returning logits and loss; `generate`) and `train.py` (AdamW, MPS then CPU, `estimate_loss` on train and val, seed 1337) | loss is finite; `generate` returns `(B, T + n)` | **the loss at step 0, and where it ends after training** |
| 3 | ✍️ `Head.forward` in `model.py`: one head of causal self-attention (Q, K, V, the mask, softmax, the weighted sum) | shape `(B, T, head_size)`; **causality** (changing future tokens doesn't change past outputs); matches a reference in `eval()`, with the reference itself barred from use inside `Head` | does attention beat the bigram? |
| 4 | Add a position embedding and a single head to the model, then retrain | val loss below the bigram's | by how much? |

> Hint to remember: scale by `head_size ** -0.5`, not by the embedding size.

**Exit criteria:**
- `make test` is green, including the new `Head` test.
- An EXPERIMENTS entry: step-0 loss predicted vs actual; val loss bigram vs one head; iterations; wall time; chip.
- A LEARNING entry in Fryday.

## F5: tiny GPT 2/2, train on MPS, generate, look inside

**Learn:** the rest of the [same video](https://www.youtube.com/watch?v=kCc8FmEb1nY), from "multi-headed self-attention" to the end. Optional: [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).

**Build:**

| Step | ✍️ What you write | Test to aim at | Predict first |
|---|---|---|---|
| 1 | `MultiHeadAttention`, `FeedForward`, `Block` (pre-norm, residuals, LayerNorm, dropout), `GPT` | shapes; parameter count matches your hand calculation | the parameter count |
| 2 | Train on MPS. Start smaller than Karpathy's 384/6/6/256 and scale up if time allows. Save `checkpoints/gpt.pt` (v9 needs it). | val loss falls | wall time and final val loss (Karpathy's ~1.48 was on an A100) |
| 3 | ✍️ `sample.py::sample_next(logits, temperature=1.0, top_k=None)`, the sampler, moved here from Fryday v0 | temperature → 0 is greedy; top-k never picks outside the k; rows are valid distributions | what temperature 0.5 vs 1.5 sounds like |
| 4 | Print or plot one head's attention weights (`uv run --with matplotlib`, no new dependency) | — | which tokens a head attends to |
| 5 | Head-count sweep: same width and steps, `n_head` = 1, 4, 8 (head_size 32, 8, 4). One EXPERIMENTS entry | — | which wins, and does a tiny head_size hurt? |

**Exit criteria:** a 500-character sample and the val loss in EXPERIMENTS; `make test` green; the checkpoint saved locally.

## F6: modern LLM architecture (learn, then three Llama-style changes)

**Learn:**
- ⭐ [Umar Jamil, *LLaMA explained*](https://www.youtube.com/watch?v=Mn_9W1nCFLo): KV cache, RoPE, RMSNorm, GQA, SwiGLU. Watch this first.
- [Karpathy, *Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI), in parts: pretraining → SFT → RLHF/DPO.

**Build** (one change at a time; each retrains `gpt-m-nodrop`'s config and gets an EXPERIMENTS entry):

| Step | ✍️ What you write | Test to aim at | Predict first |
|---|---|---|---|
| 1 | **Batched heads:** `CausalSelfAttention` replaces the `Head` loop. One `Linear(n_embd, 3 * n_embd)` for q, k, v, then `.view(B, T, n_head, head_size).transpose(1, 2)` so all heads run in one matmul. | same output as the looped `MultiHeadAttention` given the same weights; causal | wall time for 4 and 8 heads (looped: 110 s and 158 s); val loss unchanged? |
| 2 | **RMSNorm** in place of LayerNorm (Block norms and `ln_f`) | output has root-mean-square ≈ 1 per token; matches `torch.nn.RMSNorm` | val loss and wall time vs LayerNorm |
| 3 | **GQA:** `n_kv_head` (k and v heads) < `n_head` (query heads); each k/v head is shared by `n_head / n_kv_head` query heads | `n_kv_head == n_head` reproduces step 1 exactly; param count drops by the k/v weights saved | KV-cache bytes per token for n_kv_head = 4, 2, 1; how much val loss you lose |

The best result becomes the checkpoint v9 serves. RoPE and SwiGLU stay in the backlog: RoPE is a v9 stretch goal (long context), SwiGLU teaches nothing new about serving.

**Exit criteria:**
- `make test` green; EXPERIMENTS entries for steps 1–3.
- You can explain MHA → GQA and what it does to the KV-cache size (the formula is in v9 learn 1).
- Quiz; LEARNING entry in Fryday.

---

*v7a lite skipped by the owner (2026-11-10): v9 follows F6 directly.*

---

## v9 learn 1: why serving is a memory problem
- [Anyscale, *Fast LLM Serving with vLLM and PagedAttention*](https://www.youtube.com/watch?v=5ZlavKF_98U)
- [*The KV Cache: Memory Usage in Transformers*](https://www.youtube.com/watch?v=80bIUggRJf4)
- **Exit:** the KV-size formula on paper (2 × n_layer × n_kv_heads × head_dim × bytes × tokens), worked out for our GPT and written in DESIGN.md. Qwen3-4B's numbers go in Fryday.

## v9 learn 2: PagedAttention and continuous batching
- [vLLM paper](https://arxiv.org/abs/2309.06180), §3–4 · [vLLM launch blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Orca, OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu) (iteration-level scheduling) · [Anyscale, continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- **Exit:** you can draw static vs continuous batching on a timeline.

## v9 learn 3: serving metrics
- [DeepLearning.AI, *Efficiently Serving LLMs*](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
- [Lilian Weng, *Large Transformer Model Inference Optimization*](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- TTFT, TPOT, throughput vs latency, Little's law, goodput. Depth: [Umar Jamil, *Flash Attention*](https://www.youtube.com/watch?v=zy8ChVd_oTM).
- **Exit:** the metric definitions are filled in in DESIGN.md.

## v9 B1: KV cache
- **✍️ Write:** `forward(idx, kv_cache, start_pos)` and cached generation.
- **Test:** greedy output with the cache == greedy output without it (tiny random model, CPU).
- **Predict:** how per-token cost grows with context, with and without the cache.
- **Constraint:** prompt + output ≤ `ctx_len` (learned position embeddings).

## v9 B2: paged KV
- **✍️ Write:**
  - **`BlockAllocator`**: `allocate() -> int`, `free(block_id)`, `num_free`, and raise when out of blocks.
  - The KV pool per layer: `(num_blocks, page_size, n_kv_head, head_size)` (GQA from F6 shrinks it).
  - A block table per sequence.
  - Attention that gathers K and V through the block table.
- **Test:** paged output == contiguous output; the allocator never hands out a block twice.
- **Read:** [vLLM, PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention.html).

## v9 B3: continuous batching
- **✍️ Write:**
  - `Sequence` (WAITING → RUNNING → FINISHED) and `engine.py` (`add_request`, `step`, `run`).
  - A static-batching baseline.
  - `bench.py`: a seeded synthetic workload with random prompt and output lengths, and arrivals per step.
  - **`Scheduler.step()`**: admit waiting sequences FIFO while blocks and `max_num_seqs` allow, run decode, retire finished sequences and free their blocks.
- **Test:** continuous-batching outputs == sequential greedy outputs.
- **Measure:** tokens/step, batch utilisation, padding waste, TTFT and TPOT in steps, and wall-clock labelled indicative.
- **Exit:** a static vs continuous table in EXPERIMENTS and in the README.

## v9 B4: prefix caching
- **✍️ Write:** a hash per full block (the parent's hash plus the block's tokens), ref counts, and LRU reuse of freed cached blocks.
- **Workload:** a shared system prompt.
- **Measure:** hit rate and prefill tokens saved.
- **Read:** [vLLM, automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html).

## v9 B5: read the real thing
- [vLLM v1 scheduler](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched) and [kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py), next to [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
- **Exit:** a mapping table in DESIGN.md (their class or function → ours).
- **Skipped (2026-11-11, owner's call):** B1–B4 and the walkthroughs covered the ideas; reading vLLM's code can happen later if a production issue needs it. Two points stay unverified: how vLLM stores two blocks with the same hash, and whether it frees a sequence's blocks in reverse order.

**Stays in Fryday:** the vllm-metal vs `mlx_lm.server` comparison (1-session timebox), the OpenAI-contract swap, and the v9 interview session.

---

## Enhancements backlog (optional, after the core path)

F4–F5 built the simplest GPT that teaches the ideas. This is what a production model and training setup would add. Serving-side gaps (paging kernel, scheduler, preemption, chunked prefill, API) live in [DESIGN.md "Gaps vs vLLM"](DESIGN.md#gaps-vs-vllm). Pick items one at a time, each with a prediction and an EXPERIMENTS entry.

**Model architecture**

| Change | What it replaces | Why | When |
|---|---|---|---|
| Batched heads | Python loop over `Head` | one matmul for all heads; needed for KV-cache shapes | **F6 step 1** |
| Fused attention, `F.scaled_dot_product_attention(is_causal=True)` | hand-written mask + softmax | FlashAttention-style kernel, less memory | after F6 step 1 |
| RMSNorm | LayerNorm | cheaper, used by Llama | **F6 step 2** |
| RoPE | learned position table | no hard `ctx_len` limit; relative positions | v9 stretch (long context) |
| GQA (`n_kv_head < n_head`) | one k, v per head | smaller KV cache | **F6 step 3**; pays off in v9 |
| SwiGLU (or GELU) | ReLU FFN | better loss per parameter (Llama, GPT-2) | any time |
| Weight tying (`lm_head.weight = token_embedding.weight`) | separate lm_head | fewer params, often better | any time |
| Scaled init for residual projections (std 0.02/√(2·n_layer)) | plain std 0.02 | keeps deep stacks stable | with more layers |
| No biases in Linear / LayerNorm | biases everywhere | small speed and stability win (nanoGPT option) | any time |

**Training**

| Change | What it replaces | Why | When |
|---|---|---|---|
| LR warmup + cosine decay | constant lr 1e-3 | faster, more stable convergence | before scaling up |
| Gradient clipping (norm 1.0) | none | guards against loss spikes | before scaling up |
| AdamW weight decay on matrices only | default decay on everything | norms and biases shouldn't decay | before scaling up |
| Mixed precision (`torch.autocast`, bf16) | fp32 | ~2× less memory, faster on real GPUs | when scaling up |
| `torch.compile` | eager mode | fewer kernel launches (helps MPS overhead) | when timing matters |
| Gradient accumulation | B=32 per step | large effective batch on small memory | when scaling up |
| Early stopping on val + keep best checkpoint | save the last step | avoids overfit checkpoints | when val stalls (run B) |
| Resumable checkpoints (optimizer state, step, RNG) | weights + config only | restart long runs | long runs |
| Several seeds per experiment | one seed | the head-count sweep was within noise | any comparison closer than ~0.02 |

**Inference and sampling**

| Change | What it replaces | Why | When |
|---|---|---|---|
| KV cache | recompute the whole window | O(1) work per new token | **v9 B1** |
| Top-p (nucleus), min-p | top-k only | adapts the cut-off to the distribution | any time |
| Repetition / frequency penalty | none | greedy loops ("the state of the state…") | any time |
| Stop sequences, max length, EOS | fixed `max_new_tokens` | real requests end on their own | v9 B3 (`Sequence` finishes) |
| Per-request seeds | one global generator | reproducible requests in a batch | v9 B3 |

**Data and tokenizer**

| Change | What it replaces | Why | When |
|---|---|---|---|
| BPE (`tiktoken` or trained) | characters | ~4× shorter sequences, real-world KV maths | after v9, or a v9 stretch |
| Pre-tokenised `uint16` memmap file | encode the whole text into a Python list each run | fast startup, larger datasets | with a bigger dataset |
| Bigger corpus | tiny-shakespeare (1 MB) | room for bigger models without overfitting | when the model outgrows it |

**Engineering**

| Change | What it replaces | Why | When |
|---|---|---|---|
| `@dataclass GPTConfig` | plain dicts in `train.py` | typed, saved with the checkpoint, one source of truth | before v9 loads checkpoints |
| `safetensors` checkpoints + saved tokenizer | `torch.save` pickle | safe to load, portable | before sharing checkpoints |
| Experiment logging (CSV or TensorBoard) | printed lines | loss curves you can compare | when runs multiply |
| Expose attention weights (`return_attn=True`) | recompute in a script | inspection without duplicate code | if we inspect heads again |

---

## Reference shelf
- [nanoGPT](https://github.com/karpathy/nanoGPT) · [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) (the video's code)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · [paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)
- Look at these **after** writing your own version, not before.
