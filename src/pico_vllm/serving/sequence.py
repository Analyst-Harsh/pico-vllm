"""v9 B3: a request's life cycle and the per-step stats the engine records."""

from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    PENDING = "pending"  # not arrived yet (benchmark workloads arrive at a given step)
    WAITING = "waiting"  # arrived, or preempted, and queued
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """One request and everything the scheduler and the metrics need to know about it."""

    seq_id: int
    prompt: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    status: Status = Status.PENDING
    output: list[int] = field(default_factory=list)
    first_token_step: int | None = None
    finish_step: int | None = None
    preemptions: int = 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens

    @property
    def ttft(self) -> int | None:
        """Steps from arrival to the first token; 1 = first token in the arrival step."""
        if self.first_token_step is None:
            return None
        return self.first_token_step - self.arrival_step + 1

    @property
    def tpot(self) -> float | None:
        """Steps per output token after the first; 1.0 = never paused or preempted."""
        if self.first_token_step is None or self.finish_step is None or len(self.output) < 2:
            return None
        return (self.finish_step - self.first_token_step) / (len(self.output) - 1)


@dataclass
class StepStats:
    step: int
    tokens: int  # output tokens produced (= sequences that ran this step)
    running: int  # batch size after admission (static: includes finished-but-held)
    held: int  # static only: finished sequences still holding a batch slot
    blocks_used: int
