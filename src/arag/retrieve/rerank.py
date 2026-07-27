"""Cross-encoder reranking over fused candidates.

Two ways to apply the reranker, config-gated by `retrieval.rerank_fusion`:

* `replace` — the textbook version: the cross-encoder's order *is* the new order.
  Right when the reranker is strictly better informed than the retriever.
* `rrf` — fuse the reranker's order with the retrieval order it was given
  (Reciprocal Rank Fusion, same primitive used for dense+sparse). The retrieval
  prior is evidence too, and throwing it away means a reranker that is merely
  noisy on some queries can drag good hits down. Measured on this corpus in mock
  mode: `replace` costs 8.4 points of recall@1 vs. no reranking at all, while
  `rrf` keeps the hybrid gain — see RESULTS.md.
"""

from __future__ import annotations

from arag.common.schemas import RetrievedChunk
from arag.providers.base import Reranker


def _order_by_score(scores: list[float]) -> list[int]:
    """Indices sorted by descending score, ties by ascending position.

    Explicit key rather than `reverse=True` so ties resolve identically on every
    platform, and rounded so float noise can't reorder equal scores.
    """
    return sorted(range(len(scores)), key=lambda i: (-round(float(scores[i]), 6), i))


def rerank(
    reranker: Reranker,
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int | None = None,
    fusion: str = "replace",
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Return candidates re-sorted by the cross-encoder. If `top_n` is None the
    full ranked list is returned (so retrieval metrics see the whole ranking);
    the caller slices the top-n it feeds to the generator."""
    if not candidates:
        return []
    passages = [rc.chunk.text for rc in candidates]
    scores = reranker.score(query, passages)

    rerank_order = _order_by_score(scores)
    if fusion == "rrf":
        # `candidates` arrives in retrieval order, so position == prior rank.
        rerank_rank = {idx: rank for rank, idx in enumerate(rerank_order)}
        fused = [
            1.0 / (rrf_k + prior_rank + 1) + 1.0 / (rrf_k + rerank_rank[i] + 1)
            for i, prior_rank in enumerate(range(len(candidates)))
        ]
        order = _order_by_score(fused)
        final_scores = fused
    else:
        order = rerank_order
        final_scores = [float(s) for s in scores]

    if top_n is not None:
        order = order[:top_n]
    out: list[RetrievedChunk] = []
    for rank, i in enumerate(order):
        rc = candidates[i]
        out.append(
            RetrievedChunk(
                chunk=rc.chunk,
                score=round(float(final_scores[i]), 6),
                source="rerank",
                ranks={**rc.ranks, "rerank": rank},
            )
        )
    return out
