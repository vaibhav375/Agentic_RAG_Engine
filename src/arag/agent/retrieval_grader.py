"""Corrective-RAG (CRAG) style retrieval evaluator / answerability gate.

Before generating, grade whether the retrieved context can actually answer the
query. This is the standard, high-leverage fix for out-of-scope hallucination:
if the corpus doesn't contain the answer, the system should decline rather than
let the generator improvise from loosely-related passages.

Signal (heuristic mode): **IDF-weighted query coverage** — the fraction of the
query's information content (weighting rare, answer-bearing terms over ubiquitous
ones like the product name) that is actually present in the retrieved context.
Off-topic questions score low because their distinctive terms (e.g. "websocket",
"oauth", "cron") never appear in the corpus.

Grades follow the CRAG paper's three-way decision:
    correct    -> use the context and answer
    ambiguous  -> answer but keep the strict critic / trigger a corrective retry
    incorrect  -> discard; abstain (a web-search fallback would slot in here)
"""

from __future__ import annotations

from dataclasses import dataclass

from arag.common.schemas import RetrievedChunk
from arag.providers.base import content_tokens


@dataclass
class RetrievalGrade:
    grade: str          # correct | ambiguous | incorrect
    score: float        # IDF-weighted coverage in [0, 1]
    method: str = "idf"


def idf_coverage(store, query: str, contexts: list[RetrievedChunk]) -> float:
    q_tokens = content_tokens(query)
    if not q_tokens:
        return 0.0
    weights = {t: store.idf(t) for t in q_tokens}
    total = sum(weights.values()) or 1.0
    present = set()
    for rc in contexts:
        present |= content_tokens(rc.chunk.text)
    covered = sum(w for t, w in weights.items() if t in present)
    return covered / total


def grade_retrieval(store, query: str, contexts: list[RetrievedChunk], cfg) -> RetrievalGrade:
    crag = cfg.agent.get("crag", {})
    if hasattr(crag, "as_dict"):
        crag = crag.as_dict()
    correct_t = float(crag.get("correct_threshold", 0.55))
    incorrect_t = float(crag.get("incorrect_threshold", 0.40))

    score = round(idf_coverage(store, query, contexts), 4)
    if score >= correct_t:
        grade = "correct"
    elif score < incorrect_t:
        grade = "incorrect"
    else:
        grade = "ambiguous"
    return RetrievalGrade(grade=grade, score=score)
