from arag.common.schemas import Chunk, RetrievedChunk
from arag.retrieve.hybrid import reciprocal_rank_fusion


def _rc(cid, rank):
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, doc_id="d", text=cid), score=0.0, ranks={})


def test_rrf_rewards_agreement():
    dense = [_rc("a", 0), _rc("b", 1), _rc("c", 2)]
    sparse = [_rc("b", 0), _rc("a", 1), _rc("d", 2)]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    ids = [rc.chunk.chunk_id for rc in fused]
    # a and b appear in both lists near the top -> should rank above c and d.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_rrf_scores_descending():
    dense = [_rc("a", 0), _rc("b", 1)]
    sparse = [_rc("a", 0), _rc("b", 1)]
    fused = reciprocal_rank_fusion(dense, sparse)
    scores = [rc.score for rc in fused]
    assert scores == sorted(scores, reverse=True)
    # 'a' at rank 0 in both -> highest fused score.
    assert fused[0].chunk.chunk_id == "a"


def test_rrf_dedupes():
    dense = [_rc("a", 0), _rc("a", 0)]
    sparse = []
    fused = reciprocal_rank_fusion(dense, sparse)
    assert len([rc.chunk.chunk_id for rc in fused]) == 1
