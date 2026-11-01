# pico-vllm roadmap

Every session, in order. A session is about 90 minutes; the protocol is in [CLAUDE.md](../CLAUDE.md). ✍️ marks code the owner writes. In this repo that's all of it, so ✍️ flags the pieces that matter most for interviews (from Fryday roadmap §5).

**Where this sits in Fryday's plan:**

```
F4 → F5 → F6  (here)  →  v7a lite (Fryday)  →  v9 (here + Fryday)  →  v8 Triton (Fryday) → …
```

v9 comes before v8 on purpose: Triton is easier to judge once you've built an engine yourself.

| Block | Sessions | Where |
|---|---|---|
| F4 tiny GPT 1/2 | 1 on the roadmap; plan for 2 | here |
| F5 tiny GPT 2/2 | 1 on the roadmap; plan for 2 | here |
| F6 modern architecture | 1, learn only | here (notes) |
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
- `make test` is green, and the learner marker on `Head` is removed.
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

**Exit criteria:** a 500-character sample and the val loss in EXPERIMENTS; `make test` green; the checkpoint saved locally.

## F6: modern LLM architecture (learn only)

**Learn:**
- ⭐ [Karpathy, *Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI), in parts: pretraining → SFT → RLHF/DPO.
- [Umar Jamil, *LLaMA explained*](https://www.youtube.com/watch?v=Mn_9W1nCFLo): KV cache, RoPE, RMSNorm, GQA.

**Optional ✍️ change:** swap LayerNorm for RMSNorm, or add `n_kv_head` (GQA). RoPE would also remove the context limit noted in DESIGN.

**Exit criteria:** quiz; you can explain MHA → GQA and what it does to KV-cache size.

---

*Fryday runs v7a (lite) here; this repo is idle.*

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
  - **`BlockAllocator`**: `allocate() -> int`, `free(block_id)`, `num_free`, and raise when out of blocks. This is the §5 interview piece.
  - The KV pool per layer: `(num_blocks, page_size, n_head, head_size)`.
  - A block table per sequence.
  - Attention that gathers K and V through the block table.
- **Test:** paged output == contiguous output; the allocator never hands out a block twice.
- **Read:** [vLLM, PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention.html).

## v9 B3: continuous batching
- **✍️ Write:**
  - `Sequence` (WAITING → RUNNING → FINISHED) and `engine.py` (`add_request`, `step`, `run`).
  - A static-batching baseline.
  - `bench.py`: a seeded synthetic workload with random prompt and output lengths, and arrivals per step.
  - **`Scheduler.step()`**: admit waiting sequences FIFO while blocks and `max_num_seqs` allow, run decode, retire finished sequences and free their blocks. This is the §5 interview piece.
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

**Stays in Fryday:** the vllm-metal vs `mlx_lm.server` comparison (1-session timebox), the OpenAI-contract swap, and the v9 interview session.

---

## Reference shelf
- [nanoGPT](https://github.com/karpathy/nanoGPT) · [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) (the video's code)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · [paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)
- Look at these **after** writing your own version, not before.
