"""Bounded self-correction state graph.

State machine (bounded to `max_iter` retrieval retries to cap latency/cost):

    generate ─▶ critique ─┬─ supported ─▶ ACCEPT
                          └─ unsupported ─▶ reformulate ─▶ retrieve+generate ─▶ critique …
    (after max_iter unsupported) ─▶ ABSTAIN

Abstaining on unsupported answers — instead of returning a fabricated one — is
the mechanism that drives the hallucination rate down and makes the system
correctly decline on unanswerable questions.
"""

from __future__ import annotations

from arag.agent.critic import critique_answer
from arag.agent.reformulate import reformulate_query
from arag.agent.retrieval_grader import grade_retrieval
from arag.common.schemas import Answer, CritiqueResult
from arag.common.telemetry import Trace


def run_self_correction(comp, query: str, trace: Trace | None = None, max_iter: int | None = None) -> Answer:
    from arag.engine import generate_from, retrieve_contexts  # lazy import avoids cycle

    cfg = comp.cfg
    trace = trace or Trace()
    if max_iter is None:
        max_iter = int(cfg.get("agent.max_iterations", 2))
    crag_on = bool(cfg.get("agent.crag.enabled", False))
    abstain_message = cfg.get(
        "agent.abstain_message",
        "I don't have enough information in the provided sources to answer this confidently.",
    )

    def _grade(q: str):
        ranked, contexts = retrieve_contexts(comp, q, trace=trace)
        if crag_on:
            with trace.stage("crag_grade"):
                g = grade_retrieval(comp.store, q, contexts, cfg)
            return ranked, contexts, g
        return ranked, contexts, None

    def _abstain(ranked, grade) -> Answer:
        ans = Answer(query=query, answer=abstain_message, abstained=True, retrieved=ranked or [])
        if grade is not None:
            ans.retrieval_grade = grade.grade
            ans.retrieval_grade_score = grade.score
        ans.critique = CritiqueResult(supported=False, support_fraction=0.0, should_abstain=True)
        ans.trace = trace.stages
        return ans

    ranked, contexts, grade = _grade(query)
    # CRAG: out-of-scope context -> decline before generating (kills hallucination
    # on unanswerable questions). A web-search fallback would slot in here.
    if grade is not None and grade.grade == "incorrect":
        return _abstain(ranked, grade)

    ans = generate_from(comp, query, ranked, contexts, trace=trace)
    if grade is not None:
        ans.retrieval_grade, ans.retrieval_grade_score = grade.grade, grade.score
    with trace.stage("critique", iteration=0):
        critique = critique_answer(comp.judge or comp.llm, ans.answer, ans.contexts, cfg, nli=comp.nli)
    iterations = 1
    prior_queries = [query]

    # Corrective retries fire on unsupported claims. An "ambiguous" CRAG grade is
    # recorded for telemetry but does not force a retry — the claim-level critic is
    # the authority on whether the produced answer is actually grounded, so we
    # don't degrade already-good retrieval on borderline queries.
    first_grade = grade
    while (not critique.supported) and iterations <= max_iter:
        with trace.stage("reformulate", iteration=iterations):
            new_q = reformulate_query(comp.llm, query, critique.missing_info, prior_queries)
        prior_queries.append(new_q)
        # Retries use intentionally-broadened internal queries; the answerability
        # gate is applied only to the user's original query, so a widened retry
        # can't trigger a false abstention on an in-scope question.
        ranked, contexts = retrieve_contexts(comp, new_q, trace=trace)
        ans = generate_from(comp, query, ranked, contexts, trace=trace)
        if first_grade is not None:
            ans.retrieval_grade, ans.retrieval_grade_score = first_grade.grade, first_grade.score
        with trace.stage("critique", iteration=iterations):
            critique = critique_answer(comp.judge or comp.llm, ans.answer, ans.contexts, cfg, nli=comp.nli)
        iterations += 1

    ans.iterations = iterations
    ans.critique = critique

    if not critique.supported:
        ans.answer = abstain_message
        ans.abstained = True
        ans.citations = []
        critique.should_abstain = True

    return ans
