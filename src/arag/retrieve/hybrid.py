"""Retrieval: dense-only (baseline) and hybrid dense+sparse with Reciprocal Rank
Fusion (RRF)."""

from __future__ import annotations

from arag.common.schemas import RetrievedChunk
from arag.ingest.index import Store


def dense_retrieve(store: Store, query: str, k: int, dense_text: str | None = None) -> list[RetrievedChunk]:
    # `dense_text` lets HyDE embed a hypothetical answer instead of the raw query.
    hits = store.dense_search(dense_text or query, k)
    return [
        RetrievedChunk(chunk=c, score=s, source="dense", ranks={"dense": i})
        for i, (c, s) in enumerate(hits)
    ]


def sparse_retrieve(store: Store, query: str, k: int) -> list[RetrievedChunk]:
    hits = store.sparse_search(query, k)
    return [
        RetrievedChunk(chunk=c, score=s, source="sparse", ranks={"sparse": i})
        for i, (c, s) in enumerate(hits)
    ]


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists with RRF: score(d) = sum_r 1 / (rrf_k + rank_r(d)).

    Ranks are 0-based here, so we use (rrf_k + rank + 1) to avoid rank-0 dominating.
    """
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for lst, name in ((dense, "dense"), (sparse, "sparse")):
        for rank, rc in enumerate(lst):
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            ranks.setdefault(cid, {})[name] = rank
            if cid not in fused:
                fused[cid] = rc

    # Python's sort is stable, so RRF ties keep insertion order — dense hits
    # before sparse-only ones, which is the tie-break we want. Rounding the key
    # keeps float-sum noise from disturbing that.
    out: list[RetrievedChunk] = []
    for cid, score in sorted(scores.items(), key=lambda x: -round(x[1], 9)):
        rc = fused[cid]
        out.append(
            RetrievedChunk(chunk=rc.chunk, score=round(score, 6), source="fused", ranks=ranks[cid])
        )
    return out


def fuse_many(lists: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    """N-way Reciprocal Rank Fusion (used to merge per-sub-question pools)."""
    scores: dict[str, float] = {}
    keep: dict[str, RetrievedChunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    for li, lst in enumerate(lists):
        for rank, rc in enumerate(lst):
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            ranks.setdefault(cid, {})[f"list{li}"] = rank
            keep.setdefault(cid, rc)
    out = []
    for cid, sc in sorted(scores.items(), key=lambda x: -round(x[1], 9)):
        out.append(RetrievedChunk(chunk=keep[cid].chunk, score=round(sc, 6), source="fused", ranks=ranks[cid]))
    return out


def hybrid_retrieve(store: Store, query: str, cfg, dense_text: str | None = None) -> list[RetrievedChunk]:
    r = cfg.retrieval
    k_dense = int(r.get("k_dense", 10))
    k_sparse = int(r.get("k_sparse", 10))
    rrf_k = int(r.get("rrf_k", 60))
    dense = dense_retrieve(store, query, k_dense, dense_text=dense_text)
    sparse = sparse_retrieve(store, query, k_sparse)
    return reciprocal_rank_fusion(dense, sparse, rrf_k)


def retrieve(store: Store, query: str, cfg, dense_text: str | None = None) -> list[RetrievedChunk]:
    """Entry point honoring config: hybrid if enabled, else dense-only baseline.
    `dense_text` (optional) is the HyDE hypothetical passage to embed for the
    dense half; the sparse half always uses the literal query."""
    r = cfg.retrieval
    if bool(r.get("use_hybrid", False)):
        return hybrid_retrieve(store, query, cfg, dense_text=dense_text)
    return dense_retrieve(
        store, query, int(r.get("fetch_k", r.get("k_dense", 10))), dense_text=dense_text
    )
