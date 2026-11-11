"""What this file teaches: continuous batching, the scheduler loop of a serving engine.

Key idea: schedule per step, not per batch. A finished request leaves at once and a waiting
one takes its slot in the same step, so the batch stays full. One step:
  1. decode one token for every running sequence (one batched forward),
  2. retire finished sequences and free their pages at once,
  3. admit waiting sequences (FIFO) while pages and max_num_seqs allow; each is prefilled
     and gets its first token in this same step.
Out of pages mid-decode: preempt the most recently admitted sequence (free its pages,
recompute it later), as vLLM does.

prefix_caching=True: a newcomer reuses the cached pages of its longest shared prompt start
and prefills only the rest.

policy="static" is the baseline: a batch reserves pages for its full length, runs until its
longest member is done (finished members keep their slot), and only then is the next admitted.

Read next: bench.py (static vs continuous, prefix caching off vs on).
"""

from collections import deque

import torch

from pico_vllm.model.llama_style import ModernGPT
from pico_vllm.sampler import sample_next
from pico_vllm.serving.block_allocator import OutOfBlocksError
from pico_vllm.serving.paged_kv_cache import PagedKVCache
from pico_vllm.serving.sequence import Sequence, Status, StepStats


class Engine:
    def __init__(
        self,
        model: ModernGPT,
        num_blocks: int,
        page_size: int = 16,
        max_num_seqs: int = 8,
        policy: str = "continuous",
        temperature: float = 0.0,
        top_k: int | None = None,
        prefix_caching: bool = False,
    ) -> None:
        if policy not in ("continuous", "static"):
            raise ValueError(f"policy must be 'continuous' or 'static', got {policy!r}")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        sa = model.blocks[0].sa
        self.cache = PagedKVCache(
            len(model.blocks),
            num_blocks,
            page_size,
            sa.n_kv_head,
            sa.head_size,
            device=self.device,
            prefix_caching=prefix_caching,
        )
        self.max_num_seqs, self.policy = max_num_seqs, policy
        self.temperature, self.top_k = temperature, top_k
        self.pending: list[Sequence] = []
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # admission order (static: plus finished-but-held)
        self.finished: list[Sequence] = []
        self.stats: list[StepStats] = []
        self.step_no = 0
        self.preemptions = 0
        self.prefill_computed = 0  # prompt tokens pushed through the model at prefill
        self.prefill_cached = 0  # prompt tokens skipped thanks to prefix caching
        self._next_id = 0

    # ---- requests -------------------------------------------------------------------------

    def add_request(
        self, prompt: list[int], max_new_tokens: int, arrival_step: int = 0
    ) -> Sequence:
        total = len(prompt) + max_new_tokens
        if not prompt or max_new_tokens < 1:
            raise ValueError("need a non-empty prompt and max_new_tokens >= 1")
        if total > self.model.ctx_len:
            raise ValueError(f"prompt + new = {total} > ctx_len={self.model.ctx_len}")
        if self.cache.blocks_needed(total) > self.cache.allocator.num_blocks:
            raise ValueError(f"needs {self.cache.blocks_needed(total)} pages; the pool has fewer")
        seq = Sequence(self._next_id, list(prompt), max_new_tokens, arrival_step)
        self._next_id += 1
        self.pending.append(seq)
        return seq

    @property
    def has_work(self) -> bool:
        return bool(self.pending or self.waiting or self.running)

    def run(self) -> list[Sequence]:
        while self.has_work:
            self.step()
        return self.finished

    # ---- one engine step --------------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> int:
        arrived = sorted(
            (s for s in self.pending if s.arrival_step <= self.step_no),
            key=lambda s: s.arrival_step,
        )
        for s in arrived:
            s.status = Status.WAITING
            self.waiting.append(s)
            self.pending.remove(s)

        tokens = self._decode()  # 1. everyone already running gets one token
        self._retire()  # 2. finished leave; continuous frees their pages right now
        tokens += self._admit()  # 3. newcomers: prefill + first token
        self._retire()  # (a request asking for a single token is done after prefill)

        held = sum(s.status is Status.FINISHED for s in self.running)
        used = self.cache.allocator.num_blocks - self.cache.allocator.num_free
        self.stats.append(StepStats(self.step_no, tokens, len(self.running), held, used))
        self.step_no += 1
        return tokens

    def _decode(self) -> int:
        seqs = [s for s in self.running if s.status is Status.RUNNING]
        if self.policy == "continuous":
            seqs = self._make_room(seqs)
        if not seqs:
            return 0
        self.cache.bind([s.seq_id for s in seqs])
        newest = torch.tensor([[s.tokens[-1]] for s in seqs], device=self.device)  # (B, 1)
        cached = torch.tensor([len(s.tokens) - 1 for s in seqs], device=self.device)  # (B,)
        logits, _ = self.model(newest, cache=self.cache, start_pos=cached)  # one batched step
        self._emit(seqs, logits[:, -1, :])
        for s in seqs:
            self.cache.commit(s.seq_id, s.tokens[:-1])  # name any page that just filled up
        return len(seqs)

    def _make_room(self, seqs: list[Sequence]) -> list[Sequence]:
        """Preempt the newest sequences until every remaining one can get its next page."""

        def needs_page(s: Sequence) -> bool:
            return self.cache.blocks_needed(len(s.tokens)) > len(self.cache.block_tables[s.seq_id])

        while sum(map(needs_page, seqs)) > self.cache.allocator.num_free:
            if len(seqs) == 1:  # can't happen: add_request checks a request fits the pool
                raise OutOfBlocksError("one sequence needs more pages than the pool has")
            self._preempt(seqs.pop())  # most recently admitted, like vLLM's default
        return seqs

    def _preempt(self, s: Sequence) -> None:
        # Simplification: recompute only (re-prefill prompt + output later); vLLM v0 could also swap
        self.cache.free_seq(s.seq_id)
        self.running.remove(s)
        s.status = Status.WAITING
        s.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(s)  # back to the front: it was admitted before anyone waiting

    def _retire(self) -> None:
        for s in [s for s in self.running if s.status is Status.RUNNING and s.done]:
            s.status, s.finish_step = Status.FINISHED, self.step_no
            self.finished.append(s)
            if self.policy == "continuous":
                self.running.remove(s)
                self.cache.free_seq(s.seq_id)
        if self.running and all(s.status is Status.FINISHED for s in self.running):
            for s in self.running:  # static: the whole batch leaves together
                self.cache.free_seq(s.seq_id)
            self.running.clear()

    def _admit(self) -> int:
        if self.policy == "static" and self.running:
            return 0  # static: wait for the whole batch
        reserved = 0  # static: pages promised to admitted sequences but not allocated yet
        admitted = 0
        while self.waiting and len(self.running) < self.max_num_seqs:
            s = self.waiting[0]
            length = len(s.tokens)
            if self.policy == "static":
                length += s.max_new_tokens - len(s.output)  # reserve the full length
            need = self.cache.admission_cost(s.tokens, length)  # minus pages already shared
            if need > self.cache.allocator.num_free - reserved:
                break  # FIFO: later, smaller requests don't jump the queue
            self.waiting.popleft()
            s.status = Status.RUNNING
            self.running.append(s)
            before = len(s.tokens)
            self._prefill(s)
            reserved += self.cache.blocks_needed(length) - self.cache.blocks_needed(before)
            admitted += 1
        return admitted

    def _prefill(self, s: Sequence) -> None:
        # Simplification: one forward per newcomer; vLLM batches prefills and chunks long ones
        skip = self.cache.attach_prefix(s.seq_id, s.tokens)  # reuse cached prefix pages
        self.cache.bind([s.seq_id])
        rest = torch.tensor([s.tokens[skip:]], device=self.device)
        logits, _ = self.model(rest, cache=self.cache, start_pos=skip)  # only the uncached part
        self.prefill_cached += skip
        self.prefill_computed += len(s.tokens) - skip
        self._emit([s], logits[:, -1, :])
        self.cache.commit(s.seq_id, s.tokens[:-1])

    def _emit(self, seqs: list[Sequence], logits: torch.Tensor) -> None:
        next_ids = sample_next(logits, self.temperature, self.top_k)[:, 0].tolist()
        for s, t in zip(seqs, next_ids, strict=True):
            s.output.append(t)
            if s.first_token_step is None:
                s.first_token_step = self.step_no
