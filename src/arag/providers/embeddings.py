"""Embedding backends: mock (default), sentence-transformers, OpenAI."""

from __future__ import annotations

import os

import numpy as np

from arag.providers.base import Embedder, MockEmbedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, dim: int, normalize: bool = True, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension() or dim
        self.normalize = normalize
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, dim: int, normalize: bool = True, batch_size: int = 64):
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self.dim = dim
        self.normalize = normalize
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend([d.embedding for d in resp.data])
        arr = np.array(out, dtype=np.float32)
        self.dim = arr.shape[1]
        if self.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


def make_embedder(cfg) -> Embedder:
    e = cfg.embeddings
    provider = e.provider
    dim = int(e.get("dim", 384))
    normalize = bool(e.get("normalize", True))
    batch_size = int(e.get("batch_size", 64))
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(e.model, dim, normalize, batch_size)
    if provider == "openai":
        model = e.get("openai_model", "text-embedding-3-small")
        return OpenAIEmbedder(model, int(e.get("dim", 1536)), normalize, batch_size)
    return MockEmbedder(dim=dim, normalize=normalize)
