"""Reranker backends: mock (lexical) and a real cross-encoder."""

from __future__ import annotations

from arag.providers.base import MockReranker, Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def make_reranker(cfg) -> Reranker:
    provider = cfg.get("mode", "mock")
    if provider in {"local", "api"}:
        return CrossEncoderReranker(cfg.get("retrieval.rerank_model", "BAAI/bge-reranker-base"))
    return MockReranker()
