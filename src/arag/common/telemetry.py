"""Per-stage latency + cost/token accounting.

`Trace` is threaded through a single query so the API can return where the time
and money went, and the eval harness can report p50/p95 latency and cost/query.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager

from arag.common.schemas import StageTiming

# Rough public prices (USD) per 1M tokens. Kept here so cost reporting has *a*
# number; update to taste. Only used when provider usage is available.
PRICE_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "claude-3-5-sonnet-latest": {"in": 3.00, "out": 15.00},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}


def estimate_cost(model: str, in_tokens: int, out_tokens: int = 0) -> float:
    price = PRICE_PER_M_TOKENS.get(model)
    if not price:
        return 0.0
    return (in_tokens * price["in"] + out_tokens * price["out"]) / 1_000_000


class Trace:
    def __init__(self, on_stage: Callable[[StageTiming], None] | None = None) -> None:
        self.stages: list[StageTiming] = []
        self.cost_usd: float = 0.0
        self.tokens: dict[str, int] = {"in": 0, "out": 0}
        # Fired the moment a stage closes, so a caller can surface progress while
        # the query is still running. Without it the only way to "stream" is to
        # finish the whole pipeline and replay the recorded trace afterwards,
        # which shows nothing until the user has already waited for everything.
        self._on_stage = on_stage

    @contextmanager
    def stage(self, name: str, **meta):
        start = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - start) * 1000.0
            timing = StageTiming(stage=name, ms=round(ms, 2), meta=meta)
            self.stages.append(timing)
            if self._on_stage is not None:
                # A broken listener must never take down the query it is watching.
                try:
                    self._on_stage(timing)
                except Exception:
                    pass

    def add_usage(self, model: str, in_tokens: int, out_tokens: int = 0) -> None:
        self.tokens["in"] += in_tokens
        self.tokens["out"] += out_tokens
        self.cost_usd += estimate_cost(model, in_tokens, out_tokens)

    def total_ms(self) -> float:
        return round(sum(s.ms for s in self.stages), 2)
