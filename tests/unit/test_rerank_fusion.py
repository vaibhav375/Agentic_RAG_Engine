"""Reranking must not be able to throw away a good retrieval ranking.

The mock reranker scores token-F1 overlap, which carries *less* information than
the dense+sparse RRF fusion feeding it — so `replace` mode measurably degraded
recall@1 (0.917 -> 0.833 on the gold set). `rrf` mode fuses the two rankings so a
weak or noisy reranker can only nudge, never overturn.
"""

from arag.common.schemas import Chunk, RetrievedChunk
from arag.retrieve.rerank import rerank


class _Reranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, query, passages):
        return self._scores[: len(passages)]


def _cands(n):
    return [
        RetrievedChunk(chunk=Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text="t"), score=0.0, ranks={})
        for i in range(n)
    ]


def _ids(out):
    return [rc.chunk.chunk_id for rc in out]


def test_replace_mode_follows_the_reranker_completely():
    # Reranker inverts the retrieval order; `replace` obeys it.
    out = rerank(_Reranker([0.1, 0.2, 0.3, 0.9]), "q", _cands(4), fusion="replace")
    assert _ids(out) == ["c3", "c2", "c1", "c0"]


def test_rrf_mode_resists_a_single_bad_reranker_call():
    """The retriever's top hit was ranked last by the reranker. Under `replace`
    it falls to the bottom; under `rrf` the retrieval prior keeps it near the top."""
    scores = [0.0, 0.9, 0.8, 0.7]  # c0 (retrieval rank 0) scored worst
    replaced = _ids(rerank(_Reranker(scores), "q", _cands(4), fusion="replace"))
    fused = _ids(rerank(_Reranker(scores), "q", _cands(4), fusion="rrf"))
    assert replaced[-1] == "c0"
    assert fused.index("c0") < replaced.index("c0")


def test_rrf_still_promotes_what_the_reranker_likes():
    # Retrieval's last-place item is the reranker's favourite -> it should climb.
    scores = [0.1, 0.1, 0.1, 0.9]
    fused = _ids(rerank(_Reranker(scores), "q", _cands(4), fusion="rrf"))
    assert fused.index("c3") < 3


def test_rrf_agreement_keeps_the_order():
    # Both rankings agree -> fusion is a no-op.
    out = rerank(_Reranker([0.9, 0.8, 0.7, 0.6]), "q", _cands(4), fusion="rrf")
    assert _ids(out) == ["c0", "c1", "c2", "c3"]


def test_top_n_slices_after_fusion():
    out = rerank(_Reranker([0.1, 0.9, 0.2, 0.3]), "q", _cands(4), fusion="rrf", top_n=2)
    assert len(out) == 2
    assert [rc.ranks["rerank"] for rc in out] == [0, 1]


def test_default_mode_is_backwards_compatible():
    # Callers that don't pass `fusion` get the textbook behavior.
    out = rerank(_Reranker([0.1, 0.9]), "q", _cands(2))
    assert _ids(out) == ["c1", "c0"]


def test_empty_candidates():
    assert rerank(_Reranker([]), "q", [], fusion="rrf") == []
