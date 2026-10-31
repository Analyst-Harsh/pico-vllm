<div align="center">

# pico-vllm

**A GPT I'm building from scratch, then a mini-vLLM to serve it.**<br/>
KV cache → paged KV → continuous batching → prefix caching, in plain PyTorch on a Mac.

[![CI](https://github.com/Analyst-Harsh/pico-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/Analyst-Harsh/pico-vllm/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

</div>

> 🚧 **Learning in public. This repo starts empty on purpose.** I write every line of the model and the engine myself, session by session, following [the roadmap](docs/ROADMAP.md). The status table below shows how far I've got.

## Why

Tools like vLLM make LLM serving look like one command. I want to know what's underneath: why serving is a memory problem, how PagedAttention stores the KV cache in blocks, and how a scheduler keeps the GPU busy with requests that start and finish at different times. The best way I know to learn that is to build a small version on a model I also built.

It's a sibling of [**Fryday**](https://github.com/Analyst-Harsh/Fryday), my local Hindi voice assistant. What I learn here feeds how Fryday is served.

## Status

| Milestone | What it adds | Status |
|---|---|---|
| F4 | Character tokenizer, bigram baseline, one self-attention head | ⏳ next |
| F5 | Full tiny GPT trained on MPS, sampler, attention-weight inspection | ⏳ |
| v9.1 | KV cache | ⏳ |
| v9.2 | Paged KV: block allocator, block tables | ⏳ |
| v9.3 | Continuous batching: scheduler, engine loop, static-batching baseline | ⏳ |
| v9.4 | Prefix caching: block hashing, ref counts | ⏳ |

## Architecture (the target)

```mermaid
flowchart LR
  REQ["add_request"] --> WQ["waiting queue"]
  WQ --> SCH["Scheduler.step"]
  SCH <--> BM["BlockManager<br/>allocator · block tables · prefix hashes"]
  SCH --> ENG["Engine.step"]
  ENG --> GPT["tiny GPT forward<br/>(paged K/V)"]
  GPT <--> KV[("KV pool")]
  GPT --> SMP["sampler"] --> SCH
  ENG -.-> MET["metrics"]
```

- **Tiny GPT:** a character-level transformer trained on tiny-shakespeare, following Karpathy's *Let's build GPT*.
- **KV cache and paged KV:** keys and values live in fixed-size blocks drawn from a pool, so memory isn't reserved for the longest possible reply.
- **Scheduler:** each step decides which requests run, admitting new ones as soon as blocks free up (continuous batching).
- **Prefix caching:** requests that share a prompt prefix reuse the same KV blocks.

More in [docs/DESIGN.md](docs/DESIGN.md).

## Results

Filled in as each milestone lands. Every number links to its entry in [EXPERIMENTS](docs/EXPERIMENTS.md).

| Measurement | Result |
|---|---|
| Val loss: bigram → one head → full GPT | TBD |
| Static vs continuous batching: tokens/step, utilisation, TTFT, TPOT | TBD |
| Prefix caching: hit rate, prefill tokens saved | TBD |

## Quickstart

macOS on Apple silicon (for MPS) and [uv](https://docs.astral.sh/uv/). Linux works on CPU.

```sh
make sync    # torch + dev tools + git hooks
make data    # tiny-shakespeare, checksum-verified
make test
```

## How it's built

- **I write the model and engine code myself.** Claude Code is my tutor: it explains each concept, sets a failing test for me to aim at, gives hints, and reviews my code.
- **The repo tooling was set up with Claude Code:** uv, ruff, pytest, lefthook and CI.
- **The pieces I'd be asked about in an interview** are written without help: self-attention, the sampler, the block allocator and the scheduler step.

## What this isn't

Not fast and not production. It has no CUDA kernels, uses a character tokenizer and a short context, and runs as a single process. The full list, with what vLLM does instead, is in [Gaps vs vLLM](docs/DESIGN.md#gaps-vs-vllm).

## Credits

- [Andrej Karpathy, *Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
- [Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180)
- [Yu et al., *Orca*](https://www.usenix.org/conference/osdi22/presentation/yu) (iteration-level scheduling)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

## License

[MIT](LICENSE)
