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
