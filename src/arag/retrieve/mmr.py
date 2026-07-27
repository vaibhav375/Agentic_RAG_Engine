"""Maximal Marginal Relevance (MMR) selection.

Picks the final context set to balance relevance to the query against redundancy
between chunks, so the generator sees diverse evidence instead of five near-copies
of the same passage. Standard MMR:

    score(d) = λ · sim(d, query) − (1 − λ) · max_{s∈selected} sim(d, s)
"""

from __future__ import annotations

import numpy as np

from arag.common.schemas import RetrievedChunk


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    store,
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int,
    lambda_: float = 0.7,
) -> list[RetrievedChunk]:
    if len(candidates) <= top_n:
        return candidates
    qv = store.embedder.encode_one(query)
    vecs = {rc.chunk.chunk_id: store.vector_for(rc.chunk.chunk_id) for rc in candidates}
    rel = {rc.chunk.chunk_id: _cos(qv, vecs[rc.chunk.chunk_id])
           for rc in candidates if vecs[rc.chunk.chunk_id] is not None}

    selected: list[RetrievedChunk] = []
    remaining = [rc for rc in candidates if rc.chunk.chunk_id in rel]
    while remaining and len(selected) < top_n:
        best, best_score = None, -1e9
        for rc in remaining:
            cid = rc.chunk.chunk_id
            redundancy = max(
                (_cos(vecs[cid], vecs[s.chunk.chunk_id]) for s in selected), default=0.0
            )
            score = lambda_ * rel[cid] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = rc, score
        selected.append(best)
        remaining.remove(best)
    return selected
