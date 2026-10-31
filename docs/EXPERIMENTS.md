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

## 2026-11-01: bigram baseline on tiny-shakespeare (F4)
- **Question:** what are the step-0 and final losses of a character bigram? This is the baseline attention must beat.
- **Setup:** Apple M5, torch 2.15.0, MPS. `BigramLanguageModel(65)`, B=32, T=8, AdamW lr=1e-2, 3000 iters, loss = mean of 200 batches per split. Seed 1337. Code on branch `f4` after d8303d6 (uncommitted at run time). `uv run python -m pico_vllm.train`.
- **Prediction:** step 0 ≈ 4; final ≈ 2.5.
- **Result:** step 0: train 4.7305 / val 4.7241. Final (step 2999): train 2.4611 / val **2.4903**. Plateau from ~step 1500. Wall time 6.45 s (indicative only: MPS on a tiny model is mostly launch overhead). Sample: word-shaped gibberish with real line breaks and `NAME:` speaker lines.
- **Observations:** step 0 sits above uniform-guess ln(65) ≈ 4.17 because random init isn't uniform: some wrong tokens get confident scores. Train ≈ val, so no overfitting; the model is too small to memorise.
- **Decision:** baseline val = 2.49. F4 step 4 (position embedding + one head) must go below it.
