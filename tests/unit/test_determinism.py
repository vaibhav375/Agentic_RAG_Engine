"""Ranking must be a pure function of the index — identical on every machine.

Mock mode's whole value is that CI reproduces the published numbers exactly, and
ties are the norm here (lexical embeddings, BM25 zeros). numpy's `argpartition`
and default `argsort` are unstable, so tie order used to vary with numpy build
and SIMD width — enough to move retrieval metrics between macOS and the Linux
runner. These tests pin the two defenses: a stable index tie-break, and rounding
so sub-precision float noise can't reorder equal scores.
"""

import numpy as np

from arag.common.schemas import Chunk, RetrievedChunk
from arag.ingest.index import Store
from arag.providers.base import MockEmbedder
from arag.retrieve.hybrid import reciprocal_rank_fusion
from arag.retrieve.rerank import rerank


class _ConstantReranker:
    """Scores every passage identically — worst case for tie handling."""

    def __init__(self, scores):
        self._scores = scores

    def score(self, query, passages):
        return self._scores[: len(passages)]


def _store(n: int = 6) -> Store:
    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text="alpha beta") for i in range(n)]
    embedder = MockEmbedder(dim=32)
    embeddings = embedder.encode([c.text for c in chunks])
    return Store(chunks, embeddings, bm25=None, embedder=embedder, meta={})


def test_all_tied_dense_scores_rank_by_index():
    """Every chunk has identical text -> identical similarity. Order must be
    index order, not whatever the partition happened to emit."""
    store = _store(6)
    hits = store.dense_search("alpha beta", k=4)
    assert [c.chunk_id for c, _ in hits] == ["c0", "c1", "c2", "c3"]


def test_dense_ranking_ignores_subprecision_noise():
    """Two runs whose scores differ by ~1e-9 (BLAS accumulation order) must
    produce the same ranking, not a flipped one."""
    store = _store(4)
    scores = np.array([0.5, 0.5, 0.25, 0.25])
    nudged = scores + np.array([0.0, 1e-9, 1e-9, 0.0])
    assert list(store._rank_topk(scores, 4)) == list(store._rank_topk(nudged, 4))


def test_dense_ranking_respects_real_score_gaps():
    """Determinism must not flatten genuine differences."""
    store = _store(4)
    order = store._rank_topk(np.array([0.1, 0.9, 0.3, 0.7]), 4)
    assert list(order) == [1, 3, 2, 0]


def test_rerank_ties_preserve_candidate_order():
    cands = [
        RetrievedChunk(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text="t"), score=0.0, ranks={})
        for i in range(4)
    ]
    out = rerank(_ConstantReranker([0.7, 0.7, 0.7, 0.7]), "q", cands)
    assert [rc.chunk.chunk_id for rc in out] == ["c0", "c1", "c2", "c3"]
    # And a sub-precision nudge doesn't reshuffle them.
    nudged = rerank(_ConstantReranker([0.7, 0.7 + 1e-9, 0.7, 0.7 - 1e-9]), "q", cands)
    assert [rc.chunk.chunk_id for rc in nudged] == ["c0", "c1", "c2", "c3"]


def test_rrf_ties_put_dense_hits_first():
    """Equal fused scores keep insertion order — a dense hit outranks a
    sparse-only one at the same RRF score, which is the intended tie-break."""
    def rc(cid):
        return RetrievedChunk(chunk=Chunk(chunk_id=cid, doc_id="d", text=cid), score=0.0, ranks={})

    fused = reciprocal_rank_fusion([rc("dense_hit")], [rc("sparse_hit")], rrf_k=60)
    assert [x.chunk.chunk_id for x in fused] == ["dense_hit", "sparse_hit"]


def test_repeated_searches_are_identical():
    store = _store(8)
    first = [c.chunk_id for c, _ in store.dense_search("alpha", k=5)]
    for _ in range(5):
        assert [c.chunk_id for c, _ in store.dense_search("alpha", k=5)] == first
