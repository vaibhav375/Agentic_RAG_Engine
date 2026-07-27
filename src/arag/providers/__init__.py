"""Backend abstraction. Every capability (embeddings, LLM tasks, reranking, NLI)
has a `mock` implementation so the full pipeline runs deterministically with no
network or API keys, plus real implementations selected by config."""

from arag.providers.embeddings import make_embedder
from arag.providers.llm import make_llm
from arag.providers.nli import make_nli
from arag.providers.rerank import make_reranker

__all__ = ["make_embedder", "make_llm", "make_reranker", "make_nli"]
