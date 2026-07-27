"""Query reformulation / widening for the self-correction loop."""

from __future__ import annotations

from arag.providers.base import LanguageModel


def reformulate_query(
    llm: LanguageModel,
    original_query: str,
    missing_info: str | None,
    prior_queries: list[str],
) -> str:
    new_q = llm.reformulate(original_query, missing_info, prior_queries)
    new_q = (new_q or "").strip()
    if not new_q or new_q in prior_queries:
        # Fallback widening: fold the missing-info hint into the query.
        hint = (missing_info or "").strip()
        new_q = f"{original_query} {hint}".strip()
    return new_q
