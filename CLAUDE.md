# pico-vllm: notes for Claude

- **Plan:** [docs/ROADMAP.md](docs/ROADMAP.md), the session-by-session learning plan the repo was built from.
- **Design:** [docs/DESIGN.md](docs/DESIGN.md), what we built, plus the "Gaps vs vLLM" register.
- **Numbers:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

A tiny GPT built from scratch, then a mini-vLLM that serves it. It's a sibling of [Fryday](https://github.com/Analyst-Harsh/Fryday), a local Hindi voice assistant; what's learned here feeds how Fryday is served.

## How we work
The repo is a learning project, built with Claude Code as tutor and pair programmer.
- **Claude explains** the concept the next piece needs: summary → diagram → parts → dry run. Short.
- **Claude writes code on request**, then walks the owner through it step by step until the owner can explain it back.
- **The owner predicts** the output, loss or latency before each non-trivial run, compares afterwards, and asks the "why" questions.
- **Every piece is checked against a simple reference** (looped vs batched attention, cached vs recomputed, paged vs contiguous, engine vs one-at-a-time greedy).
- **Ask before acting:** before runs, commits and the next step. Keep answers short: key points plus a table.
- **No unrequested scope.**

## Rules
- **When we cut a corner production wouldn't**, say so in a `# Simplification:` comment and add a row to DESIGN.md "Gaps vs vLLM".
- **Every measurement goes in EXPERIMENTS.md, including negative results.** Each entry records the commit SHA, chip, torch version (from `uv.lock`) and seed. **No number goes in the README unless it links to an EXPERIMENTS entry.**
- **On MPS, wall-clock time for a tiny model is mostly kernel launch and Python overhead.** Report scheduler metrics first: tokens/step, utilisation, TTFT and TPOT in steps. Label wall-clock as indicative.
- **Naming:** `ctx_len` is the model's context length (Karpathy's `block_size`). `page_size` is the number of tokens per KV page. Never write a bare `block_size`.
- **Verify a link resolves before recommending it.**
- **Never commit `data/` or `checkpoints/`.**
- **One branch and PR per milestone**, squash-merged on the owner's yes.

## Commands
`make sync` (deps + git hooks) · `make data` (tiny-shakespeare) · `make test` (must stay green) · `make lint`
