"""Cross-encoder reranking over fused candidates. Big precision win, cheap."""

from __future__ import annotations

from arag.common.schemas import RetrievedChunk
from arag.providers.base import Reranker


def rerank(
    reranker: Reranker,
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Return candidates re-sorted by the cross-encoder. If `top_n` is None the
    full ranked list is returned (so retrieval metrics see the whole ranking);
    the caller slices the top-n it feeds to the generator."""
    if not candidates:
        return []
    passages = [rc.chunk.text for rc in candidates]
    scores = reranker.score(query, passages)
    # Explicit descending-score / ascending-index key rather than reverse=True:
    # ties then resolve on candidate position on every platform, and rounding
    # keeps float noise from reordering scores that are equal in any report.
    order = sorted(range(len(candidates)), key=lambda i: (-round(float(scores[i]), 6), i))
    if top_n is not None:
        order = order[:top_n]
    out: list[RetrievedChunk] = []
    for rank, i in enumerate(order):
        rc = candidates[i]
        out.append(
            RetrievedChunk(
                chunk=rc.chunk,
                score=round(float(scores[i]), 6),
                source="rerank",
                ranks={**rc.ranks, "rerank": rank},
            )
        )
    return out
