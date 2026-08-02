"""Index building + the persistent Store that retrieval reads from.

Builds a dense vector index and a sparse BM25 index over the same chunks and
persists both to `vector_store.persist_dir`.

The dense engine is a normalized brute-force cosine index: zero heavy
dependencies, exact rather than approximate, and fast enough at this corpus
scale (47 chunks — retrieval is ~4% of query latency, see
docs/local-mode-eval.md).

**There is no FAISS or Qdrant backend.** An earlier version of this docstring
claimed both; neither was ever implemented, and `vector_store.backend` was never
read. An ANN index only earns its complexity when exact search stops being
affordable, which has not happened here. If that changes, wire it in `Store`
and make the config key real at the same time.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from arag.common.schemas import Chunk
from arag.ingest.chunk import chunk_corpus
from arag.ingest.load import load_corpus
from arag.providers.base import tokenize
from arag.providers.embeddings import make_embedder

# Scores are rounded to this many decimals before ranking so that float noise
# below the precision anyone reports can't reorder results. See `_rank_topk`.
_RANK_DECIMALS = 6

CHUNKS_FILE = "chunks.jsonl"
EMB_FILE = "embeddings.npy"
BM25_FILE = "bm25.pkl"
META_FILE = "meta.json"


class Store:
    """Loaded, queryable index (dense + sparse) over the chunk set."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, bm25, embedder, meta: dict,
                 tokenized: list[list[str]] | None = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = bm25
        self.embedder = embedder
        self.meta = meta
        self._by_id = {c.chunk_id: c for c in chunks}
        # Document frequency per token, for IDF-weighted relevance scoring
        # (used by the CRAG answerability gate so ubiquitous tokens don't inflate).
        self._n = max(1, len(chunks))
        self._df: dict[str, int] = {}
        for toks in (tokenized or []):
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1

        self._row = {c.chunk_id: i for i, c in enumerate(chunks)}

    def idf(self, token: str) -> float:
        import math

        df = self._df.get(token, 0)
        # Smoothed IDF; unseen tokens get the max weight (they're maximally rare).
        return math.log((self._n + 1) / (df + 1)) + 1.0

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        i = self._row.get(chunk_id)
        return None if i is None else self.embeddings[i]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def _rank_topk(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Top-k indices by score, ranked reproducibly on any machine.

        Ties are the norm here (lexical mock embeddings, BM25 zeros), and both
        `argpartition` and the default `argsort` are *unstable* — their tie order
        varies with numpy build and SIMD width, which silently moved retrieval
        metrics between macOS and the Linux CI runner. Two defenses:
          * round to `_RANK_DECIMALS` so float32/BLAS accumulation noise (~1e-7)
            can't turn a mathematical tie into a strict ordering;
          * break remaining ties on chunk index via a stable lexsort, so the
            ranking is a pure function of the index, not of memory layout.
        """
        k = min(k, len(self.chunks))
        rounded = np.round(np.asarray(scores, dtype=np.float64), _RANK_DECIMALS)
        # lexsort's LAST key is primary: score desc, then index asc.
        return np.lexsort((np.arange(len(rounded)), -rounded))[:k]

    def dense_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        qv = self.embedder.encode_one(query)
        # embeddings are L2-normalized -> dot product = cosine similarity
        sims = self.embeddings @ qv
        return [(self.chunks[i], float(sims[i])) for i in self._rank_topk(sims, k)]

    def sparse_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        toks = tokenize(query)
        scores = np.asarray(self.bm25.get_scores(toks))
        return [(self.chunks[i], float(scores[i])) for i in self._rank_topk(scores, k)]


def build_index(cfg) -> Store:
    persist_dir = Path(cfg.get("vector_store.persist_dir", ".arag_index"))
    persist_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = cfg.get("corpus_dir", "data/corpus")
    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, cfg)

    embedder = make_embedder(cfg)
    embed_texts = [c.embed_text or c.text for c in chunks]
    embeddings = embedder.encode(embed_texts).astype(np.float32)

    # BM25 over raw chunk text (sparse half of hybrid).
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Persist
    with open(persist_dir / CHUNKS_FILE, "w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    np.save(persist_dir / EMB_FILE, embeddings)
    with open(persist_dir / BM25_FILE, "wb") as f:
        pickle.dump({"tokenized": tokenized}, f)
    (persist_dir / META_FILE).write_text(
        json.dumps(
            {
                "n_chunks": len(chunks),
                "n_docs": len(docs),
                "dim": int(embeddings.shape[1]) if embeddings.size else 0,
                "embeddings_provider": cfg.get("embeddings.provider"),
                "chunking": cfg.chunking.as_dict(),
            },
            indent=2,
        )
    )
    return Store(
        chunks, embeddings, bm25, embedder,
        json.loads((persist_dir / META_FILE).read_text()), tokenized=tokenized,
    )


def load_store(cfg) -> Store:
    persist_dir = Path(cfg.get("vector_store.persist_dir", ".arag_index"))
    if not (persist_dir / CHUNKS_FILE).exists():
        raise FileNotFoundError(
            f"No index at {persist_dir}. Run `make ingest` (or `arag ingest`) first."
        )
    chunks: list[Chunk] = []
    with open(persist_dir / CHUNKS_FILE) as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    embeddings = np.load(persist_dir / EMB_FILE)
    with open(persist_dir / BM25_FILE, "rb") as f:
        payload = pickle.load(f)
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(payload["tokenized"])
    embedder = make_embedder(cfg)
    meta = json.loads((persist_dir / META_FILE).read_text())
    return Store(chunks, embeddings, bm25, embedder, meta, tokenized=payload["tokenized"])
