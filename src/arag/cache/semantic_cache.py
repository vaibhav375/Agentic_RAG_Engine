"""Semantic (near-duplicate) query cache.

Embeds the query and returns a cached answer when a previous query is within a
cosine threshold. Semantic caching is high-leverage for latency/cost but carries
a real risk: a *false hit* (two similar-looking queries with different answers)
silently returns a wrong answer. Two guards mitigate this:

1. A conservative default threshold (0.95) — tune, don't guess.
2. `false_hit_rate()` measurement in the eval harness, so the quality cost of the
   cache is quantified rather than assumed away.

Backends: in-process (default) and Redis (persistent across restarts). Both do
the vector comparison in-process; Redis is used for durable storage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from arag.common.schemas import Answer


@dataclass
class _Entry:
    vec: np.ndarray
    query: str
    answer_json: str
    ts: float


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.95, ttl: int = 3600, max_entries: int = 1000):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: list[_Entry] = []
        self.hits = 0
        self.misses = 0

    # -- persistence hooks (overridden by Redis backend) -------------------- #
    def _persist(self, entry: _Entry) -> None:
        pass

    def _now(self) -> float:
        return time.time()

    def _evict_expired(self) -> None:
        if self.ttl <= 0:
            return
        now = self._now()
        self._entries = [e for e in self._entries if now - e.ts <= self.ttl]

    def get(self, query: str) -> Answer | None:
        self._evict_expired()
        if not self._entries:
            self.misses += 1
            return None
        qv = self.embedder.encode_one(query).astype(np.float32)
        mat = np.vstack([e.vec for e in self._entries])
        sims = mat @ qv
        i = int(np.argmax(sims))
        if float(sims[i]) >= self.threshold:
            self.hits += 1
            ans = Answer.model_validate_json(self._entries[i].answer_json)
            ans.from_cache = True
            return ans
        self.misses += 1
        return None

    def put(self, query: str, answer: Answer) -> None:
        qv = self.embedder.encode_one(query).astype(np.float32)
        entry = _Entry(vec=qv, query=query, answer_json=answer.model_dump_json(), ts=self._now())
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        self._persist(entry)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "threshold": self.threshold,
        }


class RedisSemanticCache(SemanticCache):
    def __init__(self, embedder, url: str, **kw):
        super().__init__(embedder, **kw)
        import redis  # optional dependency

        self._r = redis.from_url(url)
        self._key = "arag:semcache"
        self._load()

    def _load(self) -> None:
        import json

        raw = self._r.lrange(self._key, 0, -1)
        for item in raw:
            d = json.loads(item)
            self._entries.append(
                _Entry(
                    vec=np.asarray(d["vec"], dtype=np.float32),
                    query=d["query"],
                    answer_json=d["answer_json"],
                    ts=d["ts"],
                )
            )

    def _persist(self, entry: _Entry) -> None:
        import json

        self._r.rpush(
            self._key,
            json.dumps(
                {
                    "vec": entry.vec.tolist(),
                    "query": entry.query,
                    "answer_json": entry.answer_json,
                    "ts": entry.ts,
                }
            ),
        )
        self._r.ltrim(self._key, -self.max_entries, -1)


def build_cache(cfg, embedder) -> SemanticCache:
    c = cfg.cache
    kw = dict(
        threshold=float(c.get("similarity_threshold", 0.95)),
        ttl=int(c.get("ttl_seconds", 3600)),
        max_entries=int(c.get("max_entries", 1000)),
    )
    if c.get("backend", "memory") == "redis":
        try:
            return RedisSemanticCache(embedder, c.get("redis_url", "redis://localhost:6379/0"), **kw)
        except Exception:
            # Redis unavailable -> fall back to in-process cache rather than crash.
            return SemanticCache(embedder, **kw)
    return SemanticCache(embedder, **kw)
