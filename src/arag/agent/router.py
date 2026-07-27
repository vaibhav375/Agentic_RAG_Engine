"""Query router / complexity classifier.

Cost-aware control: trivially simple lookups can skip the expensive
self-correction loop (single critic+regen pass or none), while complex/multi-hop
queries go through the full loop. Heuristic by default; optional LLM classifier.
"""

from __future__ import annotations

from arag.providers.base import LanguageModel, content_tokens


def classify_query(llm: LanguageModel, query: str, cfg) -> str:
    """Return 'simple' or 'complex'."""
    router = cfg.agent.get("router", {})
    if hasattr(router, "as_dict"):
        router = router.as_dict()
    mode = router.get("mode", "heuristic")
    if mode == "llm":
        try:
            return llm.classify_complexity(query)
        except Exception:
            pass  # fall back to heuristic on any provider error
    return _heuristic(query, int(router.get("complex_min_tokens", 12)))


def _heuristic(query: str, complex_min_tokens: int) -> str:
    q = query.lower()
    signals = (" and ", " vs ", "compare", "difference", "versus", "both", "how many", "why")
    if any(s in q for s in signals):
        return "complex"
    if len(content_tokens(query)) >= complex_min_tokens:
        return "complex"
    return "simple"
