# Experiments

Every result, including negative ones, newest first. Write down the prediction **before** the run.

```
## YYYY-MM-DD: <title> (F4 | F5 | v9-B1 …)
- **Question:**
- **Setup:** chip, torch version (from uv.lock), device, model config, seed, commit SHA, command.
- **Prediction:** the owner's guess, written before the run.
- **Result:** numbers first (with units and n), then observations.
- **Decision:**
```

## 2026-11-11: prefix caching on a shared system prompt (v9 B4)
- **Question:** when every request starts with the same system prompt, how much prefill and memory does prefix caching save, and does it change throughput?
- **Setup:** Apple M5, torch 2.15.0, CPU. `checkpoints/gpt-m-gqa2.pt`, greedy, continuous batching, `max_num_seqs` 8, `page_size` 16. `uv run python -m pico_vllm.bench` (B4 table): 64 requests at 0.5/step, each = one shared 32-token system prompt (2 pages) + a 4–12 token user part, 4–16 new tokens, seed 1. Pool 32 pages (plenty) or 10 (tight), prefix caching off vs on. Branch `v9` after 5a8ced2 (uncommitted).
- **Prediction:** not given.
- **Result:** outputs equal one-at-a-time greedy with caching on, including under eviction and preemption (tests).

  | pages | prefix cache | steps | tokens/step | TTFT p50 / p99 | TPOT p99 | preemptions | peak pages | prefill tokens | hit rate | wall (indicative) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 32 | off | 114 | 5.67 | 1 / 4 | 1.00 | 0 | 27 | 2,541 | 0% | 0.17 s |
  | 32 | **on** | 114 | 5.67 | 1 / 4 | 1.00 | 0 | **13** | **525** | **79%** | 0.17 s |
  | 10 | off | 205 | 3.15 | 55 / 94 | 1.40 | 13 | 10 | 3,115 | 0% | 0.22 s |
  | 10 | **on** | **118** | **5.47** | **1 / 9** | 1.67 | 17 | 10 | **670** | **79%** | 0.18 s |

- **Observations:** 79% of prompt tokens come from the cache (32 shared of ~40 per prompt, minus the first request), so prefill work drops 79% and, because the 2 system pages are stored once instead of per sequence, peak memory halves (27 → 13 pages). With plenty of memory, steps and TTFT don't move: our step count charges a prefill the same one step whatever its length, so the prefill saving is invisible in steps (on a real GPU it would cut TTFT directly). With tight memory the memory saving is what matters: more sequences fit, so steps drop 205 → 118 and median TTFT 55 → 1. Preemptions rise slightly (13 → 17) because more sequences run at once and compete for growth pages.
- **Decision:** turn prefix caching on whenever prompts share a start (Fryday: system prompt, tools, a user's history). Its win shows up as prefill tokens saved and memory, so measure both; TTFT in steps under-reports it. Metric gap noted: steps don't weigh prefill length. Next: B5, read vLLM's scheduler and KV-cache manager.

## 2026-11-11: static vs continuous batching (v9 B3)
- **Question:** on the same mixed workload, how much do continuous batching and paging beat static batching, and what does a memory-tight pool cost?
- **Setup:** Apple M5, torch 2.15.0, CPU. `checkpoints/gpt-m-gqa2.pt`, greedy. `uv run python -m pico_vllm.bench` (defaults): 64 requests, Poisson arrivals at 0.5/step, prompts 4–16 chars, outputs 70% short (4–16) and 30% long (32 to ctx_len − prompt), seed 0. `max_num_seqs` 8, `page_size` 16. Pool: 32 pages (8 sequences × 64 positions, full) or 10 (tight). Static reserves each request's full length; continuous takes pages on demand and preempts (recompute) the newest sequence when out. Branch `v9` after acdf988 (uncommitted).
- **Prediction:** not given.
- **Result:** outputs equal one-at-a-time greedy for both policies, with and without preemption (tests).

  | Policy | pages | steps | tokens/step | utilisation | padding waste | TTFT p50 / p99 (steps) | TPOT p50 / p99 (steps) | preemptions | peak pages | wall (indicative) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | static | 32 | 421 | 3.29 | 41% | 56% | 139 / 247 | 1.00 / 1.00 | 0 | 23 | 0.36 s |
  | **continuous** | 32 | **197** | **7.08** | **89%** | 0% | **8 / 22** | 1.00 / 1.00 | 0 | 23 | 0.27 s |
  | continuous, tight | 10 | 312 | 4.45 | 56% | 0% | 58 / 118 | 1.00 / 6.40 | 55 | 10 | 0.34 s |

- **Observations:** continuous finishes the same work in **2.1× fewer steps** (2.2× tokens/step), and median TTFT drops from 139 to 8 steps: in static, a request arriving mid-batch waits for the batch's longest member, and 56% of batch slots are held by finished sequences. TPOT is 1.0 for both when memory is plentiful: once running, every sequence gets a token every step. With a third of the pages, continuous still beats static on every number (312 steps, TTFT p50 58) but pays in preemptions (55): p99 TPOT rises to 6.4 because preempted sequences sit out and are recomputed. Wall-clock barely moves (0.27–0.36 s): on this tiny model a step's cost is mostly fixed overhead, so steps saved ≠ time saved here.
- **Decision:** continuous batching is the engine default. Memory sets the ceiling: below what the workload's concurrency needs, preemption keeps outputs right but shows up first in p99 TPOT, so watch preemptions and p99, not averages. Next: B4 prefix caching.

## 2026-11-11: paged KV memory held vs contiguous (v9 B2)
- **Question:** how many KV slots does the paged cache hold compared with B1's ctx_len-per-sequence box?
- **Setup:** `checkpoints/gpt-m-gqa2.pt`, CPU, batch 3, 7-token prompt, `PagedKVCache(page_size=16, num_blocks=32)`. Contiguous holds 3 × 64 = 192 slots whatever the length. Commit acdf988.
- **Prediction:** not given.
- **Result:** paged output equals contiguous (tests + checkpoint check).

  | new tokens | positions cached / seq | pages / seq | slots held (paged) | vs contiguous |
  |---|---|---|---|---|
  | 10 | 16 | 1 | 48 | 25% |
  | 20 | 26 | 2 | 96 | 50% |
  | 50 | 56 | 4 | 192 | 100% |

  Sampled run (temperature 0.8, top-k 20, seed 1337, 50 new): block tables `[31, 28, 25, 22]`, `[30, 27, 24, 21]`, `[29, 26, 23, 20]`; pool 20/32 free during, 32/32 after `free()`.
- **Observations:** the saving is exactly the unused tail: short outputs hold a quarter of the memory, and outputs near ctx_len save nothing (4 × 16 = 64 = ctx_len). Pages interleave across sequences because all three grow in lockstep, so each sequence's pages are scattered: the block table is doing real work. Waste per sequence is under one page.
- **Decision:** paging pays off when lengths vary and most stop well before ctx_len, which is B3's mixed workload. Measure KV block usage there.

## 2026-11-11: KV cache vs recompute (v9 B1)
- **Question:** how much faster is `generate_cached` than `generate`, and does per-token cost grow with position?
- **Setup:** Apple M5, torch 2.15.0. `checkpoints/gpt-m-gqa2.pt` (ModernGPT, RMSNorm, 4 query / 2 k/v heads, ctx_len 64), batch 1, prompt = 1 token, greedy. Median of 5 runs after a warm-up, MPS (with `torch.mps.synchronize`) and CPU. Branch `v9` after 914d11c (uncommitted). Script in the session scratchpad (`time_kv.py`).
- **Prediction:** not given.
- **Result:** outputs identical (also checked for `gpt-m-rms`). Token positions pushed through the model for 63 new tokens: **2,016 without the cache, 63 with it** (1 + 63 counting the prompt).

  | Device | new tokens | generate | cached | speed-up |
  |---|---|---|---|---|
  | MPS | 15 / 31 / 63 | 2.53 / 2.28 / 2.39 ms/tok | 2.00 / 2.00 / 2.22 ms/tok | 1.3× / 1.1× / 1.1× |
  | CPU | 15 / 31 / 63 | 0.38 / 0.40 / 0.46 ms/tok | 0.33 / 0.33 / 0.33 ms/tok | 1.2× / 1.2× / 1.4× |

- **Observations:** 32× less work, but only 1.1–1.4× faster. The shape is right on CPU: uncached ms/token grows with position (0.38 → 0.46), cached stays flat (0.33). The size isn't: for a 750K-param model, pushing 64 positions through instead of 1 is almost free, and each step's cost is the fixed ~50 kernel launches plus Python, which the cache doesn't remove. MPS is ~6× slower than CPU here (launch overhead again), and its per-step time is flat at ~2 ms whatever the work.
- **Decision:** B1 is correct and does what it should (work per token constant). Its speed win needs a model where compute or memory per step dominates. In the engine, measure tokens/step and work, not wall-clock. The cache's real role in v9 is enabling batching: B2/B3 build on it.

## 2026-11-10: GQA, n_kv_head 4 / 2 / 1 (F6 step 3)
- **Question:** how much val loss do we pay for sharing k/v heads (4 query heads throughout)?
- **Setup:** Apple M5, torch 2.15.0, MPS. `ModernGPT(rmsnorm=True, n_kv_head=…)`, otherwise `gpt-m-rms` (n_embd 128, n_head 4, n_layer 4, ctx_len 64, 5000 iters). Attention is now always `GroupedQueryAttention` (n_kv 4 = plain MHA via `repeat_interleave(1)`). Seed 1337. Branch `f6` after ce5ada2 (uncommitted). `--model gpt-m-rms`, `gpt-m-gqa2`, `gpt-m-gqa1`, run one after the other.
- **Prediction:** small loss for 2, bigger for 1.
- **Result:**

  | Run | k/v heads | params | KV cache / token (fp32, 4 layers) | final train | final val | wall time (indicative) |
  |---|---|---|---|---|---|---|
  | gpt-m-rms | 4 | 815,553 | 4 KB | 1.3389 | **1.5681** | 84 s |
  | gpt-m-gqa2 | 2 | 750,017 | 2 KB | 1.3401 | **1.5836** | 101 s |
  | gpt-m-gqa1 | 1 | 717,249 | 1 KB | 1.3435 | **1.5726** | 93 s |

- **Observations:** params drop by exactly the k/v weights (−65,536 and −98,304). **Train** loss rises in order (1.3389 → 1.3401 → 1.3435): less capacity, as expected. **Val** doesn't: n_kv 1 beat n_kv 2. The n_kv-4 config has now given val 1.5722, 1.5621 and 1.5681 across three runs (spread 0.010), so +0.016 (n_kv 2) and +0.005 (n_kv 1) are at or just past the noise. Prediction half right: the cost is small, but "bigger for 1" isn't visible at this size. GQA is **slower** to train here (+9 to +17 s): `repeat_interleave` adds copies and launches, and the k/v saving shows up only at decode (v9).
- **Follow-up, fold instead of copy:** `repeat_interleave` replaced by folding each group's query heads into `g*T` rows, so k/v are never copied (test matches the copy version via SDPA). Prediction: about the same time. Result `gpt-m-gqa2`: val 1.5854, wall time **85 s** (vs 101 s with the copy, 84 s for n_kv 4). Prediction wrong: the copies cost ~16 s, so they weren't free even in training. gqa2 has now given 1.5836 and 1.5854, both above all three n_kv-4 runs (1.562–1.572), so its ~0.015 cost looks real; gqa1's single 1.5726 still needs a repeat.
- **Decision:** at our size the quality cost of GQA is ≲ 0.015 val, within noise. Separating it would need several seeds per config. Default for v9: decide when the KV cache exists and the memory win can be measured (tokens/step, max batch). The `repeat_interleave` copy is logged in DESIGN "Gaps vs vLLM".

## 2026-11-10: RMSNorm vs LayerNorm (F6 step 2)
- **Question:** does RMSNorm in place of all 9 LayerNorms keep val loss and cut wall time?
- **Setup:** Apple M5, torch 2.15.0, MPS. `ModernGPT(rmsnorm=True)`, otherwise `gpt-m-csa` (batched heads, n_embd 128, n_layer 4, ctx_len 64, 5000 iters). 815,553 params (−1,152: 9 biases × 128). Our hand-written `RMSNorm` (pow, mean, rsqrt, 2 muls). Seed 1337. Branch `f6` after 4586773 (uncommitted). `--model gpt-m-rms`, then `--model gpt-m-csa` rerun right after as the same-conditions baseline.
- **Prediction:** same loss, slightly faster.
- **Result:**

  | Run | norm | final train | final val | wall time (indicative) |
  |---|---|---|---|---|
  | gpt-m-rms | RMSNorm | 1.3386 | **1.5722** | 89 s |
  | gpt-m-csa (rerun) | LayerNorm | 1.3362 | **1.5719** | 81 s |

- **Observations:** loss prediction right (Δ 0.0003). Speed prediction wrong: **~10% slower**. The `gpt-m-csa` rerun gave val 1.5719 vs 1.5776 this morning with the same seed, so MPS runs aren't bit-reproducible and val noise is ~0.006; any Δ below that means nothing. Likely cause of the slowdown: `nn.LayerNorm` is one fused kernel, while our RMSNorm is ~5 separate ops, and on MPS each launch costs more than the arithmetic RMSNorm saves. Not verified.
- **Decision:** keep RMSNorm (Llama-standard, same loss). The "faster" claim only holds for a fused kernel; check `F.rms_norm` before claiming a speed win. Next: GQA (step 3).
- **Follow-up, `F.rms_norm`:** forward swapped to `F.rms_norm(x, (C,), weight, eps)` (tests still match `nn.RMSNorm`). Prediction: back to ~81 s. Result: val 1.5621, wall time **88 s**: no faster than the hand-written 89 s. Val is 0.010 below the other runs, a bit beyond the ~0.006 noise seen so far, so treat it as noise until it repeats. So either `F.rms_norm` isn't one fused kernel on MPS (forward + backward) in torch 2.15, or the norm wasn't where the 8 s went. Not separated yet; a forward + backward microbenchmark of the two norms would settle it. Kept `F.rms_norm` (shorter, same maths).

## 2026-11-10: batched heads vs looped heads (F6 step 1)
- **Question:** does running all heads in one matmul (`CausalSelfAttention`) cut wall time without changing val loss?
- **Setup:** Apple M5, torch 2.15.0, MPS. `ModernGPT` (modern.py) = `gpt-m-nodrop` config (n_embd 128, n_layer 4, ctx_len 64, B 32, 5000 iters, AdamW 1e-3, dropout 0), 816,705 params (same as looped). Seed 1337. Branch `f6` after de0ffe2 (uncommitted). `--model gpt-m-csa` and `gpt-m-csa-h8`, run one after the other. Wall time = `/usr/bin/time` real, includes evals and the 500-char sample.
- **Prediction:** not given (owner asked a GQA question instead).
- **Result:**

  | Run | heads | final train | final val | wall time (indicative) | looped wall time |
  |---|---|---|---|---|---|
  | gpt-m-csa | 4 | 1.3377 | **1.5776** | 82 s | 110 s |
  | gpt-m-csa-h8 | 8 | 1.3380 | **1.5766** | 87 s | 158 s |

- **Observations:** 4 heads run 1.34× faster, 8 heads 1.82× faster. The cost of more heads nearly vanishes (+5 s, vs +48 s looped): the extra cost was kernel launches, not maths. Val is 0.008 lower than the looped runs. The maths is identical (the test matches to 1e-6), so this is init noise: `ModernGPT` draws its random weights in a different order. Head count is still a null result for loss.
- **Decision:** batched heads become the default for F6 and v9. `(B, nh, T, hs)` k/v is the KV-cache shape. Next: RMSNorm (step 2).

## 2026-11-02: head-count sweep, n_head 1 / 4 / 8 (F5 step 5), null result
- **Question:** at a fixed width (n_embd 128), does splitting attention into more, smaller heads change val loss?
- **Setup:** as run B (`gpt-m-nodrop`): 816,705 params in every run (n_head doesn't change the count), ctx 64, B=32, 5000 iters, lr 1e-3, no dropout, seed 1337, Apple M5, torch 2.15.0, MPS. `--model gpt-m-h1` / `gpt-m-nodrop` / `gpt-m-h8`. Branch `f5` after 6058af0 (config lines uncommitted at run time).
- **Prediction:** 4 heads best, then 8, then 1.
- **Result:**

  | n_head | head_size | final train | final val | best val (step) | wall time (indicative) |
  |---|---|---|---|---|---|
  | 1 | 128 | 1.3657 | 1.5858 | 1.5858 (4999) | 76 s |
  | 4 | 32 | 1.3436 | 1.5851 | 1.5851 (4999) | 110 s |
  | 8 | 16 | 1.3372 | 1.5846 | 1.5784 (4500) | 158 s |

- **Observations:** final val is identical to within **0.001**, far below the ~0.01 step-to-step noise of `estimate_loss`. So no measurable difference: the ranking can't be judged from one seed. Small, consistent hint: more heads give slightly lower *train* loss (1.366 → 1.337) and the best *val* point (8 heads, 1.578). Wall time nearly doubles per 2× heads: our `MultiHeadAttention` runs heads in a **Python loop**, so MPS pays launch overhead per head, the same cost a batched implementation (one matmul for all heads) avoids.
- **Decision:** keep 4 heads (`gpt-m-nodrop.pt` stays the checkpoint). At this size head count doesn't matter; it would need several seeds and a bigger model to separate them. Added a "Gaps vs vLLM" note: per-head Python loop instead of batched heads.

## 2026-11-02: what the heads look at (F5 step 4)
- **Question:** which earlier characters do the trained heads attend to?
- **Setup:** `checkpoints/gpt-m-nodrop.pt` on CPU. `wei` recomputed from each head's own key/query on `ln1(x)` of its Block. Stats: 200 random val windows of T=64, positions t ≥ 8, mean attention mass per category. Heatmap: Block 3 on `"ROMEO:\nWhat is the matter, man?"`. `uv run --with matplotlib`, script and PNG in the session scratchpad (not committed). Seed 1337. Branch `f5` after ce838c5.
- **Prediction:** heads look at the start of the current word.
- **Result (attention mass; rows sum to 1, categories overlap):**

  | Head | self | prev char | word's 1st char | space/newline | > 8 back |
  |---|---|---|---|---|---|
  | L0 H0 | 0.17 | **0.55** | 0.18 | 0.21 | 0.07 |
  | L1 H3 | 0.15 | 0.19 | 0.04 | **0.80** | 0.11 |
  | L2 H3 | 0.06 | 0.05 | 0.04 | 0.49 | **0.56** |
  | L3 H0–H3 | 0.11–0.19 | 0.09–0.19 | 0.05–0.12 | 0.11–0.27 | **0.26–0.40** |

  | **uniform baseline** (equal weight on every visible char) | 0.04 | 0.04 | 0.04 | 0.19 | 0.67 |

  No head puts more than 0.19 on the word's first character.
- **Observations (corrected after adding the uniform baseline):**
  - **Previous-char head:** L0 H0 = 0.55, 15× uniform. Strong.
  - **Boundary head:** L1 H3 = 0.80 on spaces/newlines, 4× uniform: it finds the **boundary before** the word.
  - **Word start:** 0.15–0.19 in L0 and L1 H2, about 4–5× uniform. So the prediction is weakly supported: some heads lean towards the word's first letter, but none specialises in it.
  - **No head is "long-range":** every head is *below* the uniform 0.67 for "> 8 back". Block 3 is just more spread out (diffuse), not reaching further. An earlier draft of this entry said "later Blocks look further back"; that was wrong without the baseline.
- **Decision:** none needed; this is understanding. Lesson: always compare attention stats with a uniform baseline.

## 2026-11-02: temperature and top-k on the best checkpoint (F5)
- **Question:** what do temperature 0 / 0.5 / 1.0 / 1.5 and top-k 5 sound like?
- **Setup:** `checkpoints/gpt-m-nodrop.pt` (817K, val 1.585) on CPU, prompt `"ROMEO:\n"`, 200 new chars, `torch.Generator().manual_seed(1337)` per setting. Branch `f5` after 7530fd7 (uncommitted). Script in the session scratchpad (not committed).
- **Prediction:** asked in ROADMAP; none given before the run.
- **Result (qualitative, n=1 sample each):**
  - **0.0 (greedy):** loops immediately: "What is the state of the state of the state?" repeated with "Second Servingman:".
  - **0.5:** fluent, safe, generic ("the prince of the s…"); few invented words.
  - **1.0:** varied, real names (HENRY VI, POMPHEY), some invented words.
  - **1.5:** breaks down into non-words ("Cpruto-mocket", "lrzes", "Eveg-gpoevery").
  - **1.0 + top-k 5:** the best mix: varied but rarely broken; FRIAR LAURENCE, dialogue layout.
- **Decision:** greedy for engine tests (deterministic, v9 compares outputs exactly); temperature ~0.8–1.0 with top-k for demos.

## 2026-11-02: run B, wider GPT + ctx 64, dropout 0.2 vs 0 (F5)
- **Question:** does 4× capacity (n_embd 128) and 2× context (ctx_len 64) beat val 1.73, and does dropout help at this size?
- **Setup:** Apple M5, torch 2.15.0, MPS. `GPT(65, n_embd=128, n_head=4, n_layer=4, ctx_len=64)`, **816,705** params. B=32, T=64, AdamW lr=1e-3, 5000 iters, eval 200 batches per split. Seed 1337. Branch `f5` after 7bdda62 (uncommitted). `--model gpt-m` (dropout 0.2) and `--model gpt-m-nodrop` (0.0). Saved `checkpoints/gpt-m.pt`, `checkpoints/gpt-m-nodrop.pt`.
- **Prediction:** no number given; no dropout wins; longer context helps because each token sees more.
- **Result:**

  | Run | dropout | final train | final val | gap | wall time (indicative) |
  |---|---|---|---|---|---|
  | gpt-m | 0.2 | 1.4325 | 1.6338 | 0.20 | 116 s |
  | **gpt-m-nodrop** | 0.0 | 1.3436 | **1.5851** | 0.24 | 110 s |

- **Observations:** prediction right: no dropout wins by 0.05, though its gap is larger (0.24) and val flattened from step 3500 (1.595 → 1.585), the first sign of memorising. Both beat the 210K GPT (1.73) by 0.10–0.15. Sample has longer coherent phrases ("for the hollow of York", "I beseech you").
- **Decision:** `gpt-m-nodrop` is the best checkpoint so far; v9 uses it. Dropout becomes worth it once val stalls for longer, i.e. bigger model or more steps.

## 2026-11-02: run A, same GPT + dropout 0.2 (F5), negative result
- **Question:** does dropout 0.2 close the 0.19 train/val gap of the first GPT and lower val?
- **Setup:** as "first GPT on MPS" below, only `dropout=0.2`. `--model gpt-drop`. Seed 1337. Branch `f5` after 7bdda62 (uncommitted). Saved `checkpoints/gpt-drop.pt`.
- **Prediction:** final val ≈ 1.65; gap shrinks.
- **Result:** final train 1.6482 / val **1.8255** (vs 1.5415 / 1.7299 without dropout). Gap **0.18**, about the same. Wall time not captured.
- **Observations:** worse by 0.10, and the gap didn't move. So the gap was mostly **not** overfitting: a 210K-param model on 1M chars is capacity-limited, and val is the last 10% of the file (different scenes and speakers), so some gap is distribution shift. Dropout just slowed learning of a model that wasn't memorising yet.
- **Decision:** dropout off at this size. Run B (wider + ctx 64) keeps dropout 0.2 as planned, because a 4× larger model is where memorising becomes likely; compare against a B without dropout if B's gap grows.

## 2026-11-02: first GPT on MPS (F5)
- **Question:** how far below one head (val 2.41) does a small full GPT go?
- **Setup:** Apple M5, torch 2.15.0, MPS. `GPT(65, n_embd=64, n_head=4, n_layer=4, ctx_len=32, dropout=0.0)`, 209,729 params, init N(0, 0.02). B=32, T=32, AdamW lr=1e-3, 5000 iters, loss = mean of 200 batches per split. Seed 1337. Code on branch `f5` after 7bdda62 (uncommitted at run time). `uv run python -m pico_vllm.train --model gpt`. Saved `checkpoints/gpt.pt` (not committed).
- **Prediction:** final val ≈ 2.0. Wall time: not predicted.
- **Result:** step 0: train 4.1963 / val 4.1956 (≈ ln 65 = 4.17, as the init intends). Final (step 4999): train 1.5415 / val **1.7299**. Wall time 78.8 s (indicative). Sample: real words and names ("Edward", "Romeo", "QUEEN:"), dialogue layout, broken grammar.
- **Observations:** beat the prediction by 0.27. Train–val gap grew from ~0.04 (step 500) to **0.19**: the model is starting to overfit, and dropout is still 0. Val was still falling at the end.
- **Decision:** keep this checkpoint for now. Next levers when scaling: dropout 0.2, longer training, then width/depth toward Karpathy's 384/6/6/256 (his val ~1.48 on an A100).

## 2026-11-01: one attention head vs bigram (F4)
- **Question:** does one causal self-attention head (with position embeddings) beat the bigram's val 2.49?
- **Setup:** Apple M5, torch 2.15.0, MPS. `SingleHeadLanguageModel(65, n_embd=32, ctx_len=8)`, head_size 32, B=32, T=8, AdamW lr=1e-3, 5000 iters, loss = mean of 200 batches per split. Seed 1337. Code on branch `f4` after af69327 (uncommitted at run time). `uv run python -m pico_vllm.train --model head`.
- **Prediction:** final val ≈ 2.4.
- **Result:** step 0: train 4.2000 / val 4.2047. Final (step 4999): train 2.3951 / val **2.4126** (best 2.3992 at step 4000). Wall time 11.82 s (indicative). Sample is still gibberish, with slightly more word-like pieces ("that", "thy", "you", "my").
- **Observations:** step 0 ≈ ln(65) = 4.17 this time: a small-init `lm_head` gives near-uniform logits, unlike the bigram's raw embedding table. Only −0.08 vs the bigram: context is just 8 characters, one head, no feed-forward; the head mostly gathers information but there's little capacity to use it.
- **Decision:** beats the baseline (exit criterion met). The big drop should come in F5 (multi-head + FFN + blocks + longer ctx_len).

## 2026-11-01: bigram baseline on tiny-shakespeare (F4)
- **Question:** what are the step-0 and final losses of a character bigram? This is the baseline attention must beat.
- **Setup:** Apple M5, torch 2.15.0, MPS. `BigramLanguageModel(65)`, B=32, T=8, AdamW lr=1e-2, 3000 iters, loss = mean of 200 batches per split. Seed 1337. Code on branch `f4` after d8303d6 (uncommitted at run time). `uv run python -m pico_vllm.train`.
- **Prediction:** step 0 ≈ 4; final ≈ 2.5.
- **Result:** step 0: train 4.7305 / val 4.7241. Final (step 2999): train 2.4611 / val **2.4903**. Plateau from ~step 1500. Wall time 6.45 s (indicative only: MPS on a tiny model is mostly launch overhead). Sample: word-shaped gibberish with real line breaks and `NAME:` speaker lines.
- **Observations:** step 0 sits above uniform-guess ln(65) ≈ 4.17 because random init isn't uniform: some wrong tokens get confident scores. Train ≈ val, so no overfitting; the model is too small to memorise.
- **Decision:** baseline val = 2.49. F4 step 4 (position embedding + one head) must go below it.
