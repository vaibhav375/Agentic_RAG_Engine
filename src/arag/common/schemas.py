"""Core data contracts used across ingestion, retrieval, generation, the agent
loop, and evaluation. Pydantic models so everything serializes cleanly to JSON
for traces, the API, and the eval harness."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of text plus provenance for citations."""

    chunk_id: str
    doc_id: str
    text: str
    # Text actually embedded (may include contextual-enrichment prefix).
    embed_text: str | None = None
    section: str | None = None
    source_path: str | None = None
    ordinal: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = 0.0
    # Where the candidate came from, for debugging hybrid fusion.
    source: str = "dense"  # dense | sparse | fused | rerank
    ranks: dict[str, int] = Field(default_factory=dict)  # e.g. {"dense": 3, "sparse": 7}


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    section: str | None = None
    quote: str | None = None


class StageTiming(BaseModel):
    stage: str
    ms: float
    meta: dict[str, Any] = Field(default_factory=dict)


class ClaimJudgement(BaseModel):
    claim: str
    supported: bool
    reason: str | None = None
    method: str = "llm"  # llm | nli
    score: float | None = None
    # Severity. "Unsupported" covers two different failures: the sources say
    # otherwise (contradicted -> fabrication) versus the sources are silent
    # (an aside beyond the context, which may still be true). Only available
    # from the NLI signal, which reports contradiction probability directly.
    contradiction: float | None = None
    contradicted: bool = False


class CritiqueResult(BaseModel):
    supported: bool                     # overall verdict (all/enough claims supported)
    support_fraction: float
    claims: list[ClaimJudgement] = Field(default_factory=list)
    missing_info: str | None = None     # what to go find on the next iteration
    contradicted_fraction: float = 0.0  # share of claims the sources contradict
    should_abstain: bool = False


class Answer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    # Full ranked candidate list from the final retrieval pass (pre top-n cut).
    # Used to score retrieval quality (recall@k / precision@k / MRR) separately
    # from generation quality — a wrong answer from good retrieval is a
    # generation bug; a missed passage is a retrieval bug.
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    abstained: bool = False
    iterations: int = 1
    from_cache: bool = False
    route: str = "full"                 # full | direct (router decision)
    retrieval_grade: str | None = None  # CRAG: correct | ambiguous | incorrect
    retrieval_grade_score: float | None = None
    input_flags: list[str] = Field(default_factory=list)  # guardrail: injection patterns hit
    critique: CritiqueResult | None = None
    trace: list[StageTiming] = Field(default_factory=list)
    cost_usd: float = 0.0
    tokens: dict[str, int] = Field(default_factory=dict)

    def context_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.contexts]


class Difficulty(str, Enum):
    easy = "easy"
    multi_hop = "multi_hop"
    unanswerable = "unanswerable"
    adversarial = "adversarial"   # prompt-injection / false-premise traps


class GoldQA(BaseModel):
    """One row of the gold evaluation set."""

    id: str
    question: str
    answer: str | None = None                    # None for unanswerable
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_doc_ids: list[str] = Field(default_factory=list)
    # A short verbatim span from the corpus that supports the answer. Used to
    # locate gold chunks robustly across different chunking configs (which change
    # chunk ids), so context precision/recall stay meaningful in the ablation.
    supporting_quote: str | None = None
    difficulty: Difficulty = Difficulty.easy
    # For adversarial rows that plant a FALSE PREMISE: the false claim itself,
    # stated plainly. A safe answer must either abstain or actively contradict
    # it. Without this, a grounded answer that quietly leaves the premise
    # standing scores as a pass — measured on x17, where the model gave accurate
    # BackgroundTasks usage while never correcting "tasks run before the
    # response is sent". See eval/metrics.py.
    must_refute: str | None = None

    @property
    def is_answerable(self) -> bool:
        return self.difficulty in (Difficulty.easy, Difficulty.multi_hop)


class EvalRecord(BaseModel):
    """Per-question evaluation outcome for one pipeline config."""

    id: str
    question: str
    predicted: str
    gold_answer: str | None
    difficulty: Difficulty
    retrieved_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
