"""Evaluation metrics.

Retrieval and generation are scored **separately** so a regression points at the
right half of the pipeline:

RETRIEVAL (does the pool contain the right passages?)
- recall@k     : fraction of gold support docs present in the top-k ranked pool.
- precision@k  : fraction of the top-k ranked pool that are gold support docs.
- mrr          : reciprocal rank of the first gold doc (ranking quality).

GENERATION (given what it retrieved, is the answer right and grounded?)
- context_precision/recall : quality of the top-n actually shown to the generator.
- faithfulness             : fraction of answer claims entailed by that context.
                             Measured with an NLI model by default so it is
                             independent of the LLM-judge that drives the loop
                             (avoids grading the loop with its own judge).
- answer_relevance         : embedding cosine(question, answer) — does the answer
                             actually address the question (RAGAS-style).
- answer_correctness       : token-F1 vs the gold answer.
- hallucination            : 1 if the answer contradicts the sources or its core
                             is ungrounded (`eval.hallucination.mode: severity`).
                             An aside the sources merely don't mention is counted
                             in `unsupported_claim_rate`, not here — the two are
                             different failures. `mode: strict` restores the old
                             any-unsupported-claim rule, and both are always
                             reported (`hallucination_rate_strict`).
"""

from __future__ import annotations

import random
import statistics

import numpy as np

from arag.agent.critic import critique_answer
from arag.common.schemas import Answer, GoldQA
from arag.providers.base import content_tokens, split_sentences


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def token_f1(pred: str, gold: str) -> float:
    p, g = content_tokens(pred), content_tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec)


def _dedupe(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def strict_ks(cfg) -> list[int]:
    """Small k values to report recall at, alongside the headline recall@k."""
    ks = cfg.get("eval.strict_ks") or []
    return [int(k) for k in ks]


def recall_at_k(ranked_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    topk = set(_dedupe(ranked_doc_ids)[:k])
    return round(len(topk & set(gold_doc_ids)) / len(set(gold_doc_ids)), 4)


def precision_at_k(ranked_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    topk = _dedupe(ranked_doc_ids)[:k]
    if not topk:
        return 0.0
    gold = set(gold_doc_ids)
    return round(sum(d in gold for d in topk) / len(topk), 4)


def mrr(ranked_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    for i, d in enumerate(_dedupe(ranked_doc_ids)):
        if d in gold:
            return round(1.0 / (i + 1), 4)
    return 0.0


def context_precision_recall(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> tuple[float, float]:
    if not gold_doc_ids or not retrieved_doc_ids:
        return (0.0, 0.0)
    gold = set(gold_doc_ids)
    hits = [d for d in retrieved_doc_ids if d in gold]
    precision = len(hits) / len(retrieved_doc_ids)
    recall = len(set(hits)) / len(gold)
    return (round(precision, 4), round(recall, 4))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def answer_relevance(embedder, question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    qv = embedder.encode_one(question)
    av = embedder.encode_one(answer)
    return round(max(0.0, _cosine(qv, av)), 4)


# --------------------------------------------------------------------------- #
# per-record
# --------------------------------------------------------------------------- #
def _is_fabrication(cfg, ans: Answer, faithfulness: float, contradicted_fraction: float) -> bool:
    """Does this answer count as a hallucination?

    Two definitions, selected by `eval.hallucination.mode`:

    * `strict` — any claim not fully supported (`faithfulness < 1.0`). This was
      the original rule. It scores "correct, grounded, plus one aside the sources
      don't mention" identically to an invented answer. Measured consequence: the
      per-answer flag rate sat at 0.545–0.727 across five configurations and could
      not be moved by a stronger judge or a stricter prompt, because it is
      measuring elaboration rather than fabrication (docs/local-mode-eval.md).

    * `severity` (default) — the answer contradicts the sources, or its core is
      ungrounded (`faithfulness < min_support_fraction`). An aside beyond the
      context is still reported, via `unsupported_claim_rate`, but it no longer
      counts as a hallucination on its own.

    `strict` is kept so the old number stays reproducible and comparable; both are
    reported side by side in every run.
    """
    if ans.abstained:
        return False
    mode = cfg.get("eval.hallucination.mode", "severity")
    if mode == "strict":
        return faithfulness < 1.0
    contra_min = float(cfg.get("eval.hallucination.min_contradicted_fraction", 0.001))
    support_min = float(cfg.get("eval.hallucination.min_support_fraction", 0.5))
    return contradicted_fraction >= contra_min or faithfulness < support_min


def evaluate_record(comp, gold: GoldQA, ans: Answer) -> dict:
    cfg = comp.cfg
    k = int(cfg.get("eval.retrieval_k", 10))
    ranked_doc_ids = [rc.chunk.doc_id for rc in (ans.retrieved or ans.contexts)]
    context_doc_ids = [rc.chunk.doc_id for rc in ans.contexts]

    # Faithfulness measured independently of the loop's judge (NLI by default).
    faith_method = cfg.get("eval.faithfulness_method", "nli")
    contradicted_fraction = 0.0
    if ans.abstained or not ans.answer.strip():
        faithfulness = 1.0 if ans.abstained else 0.0
    else:
        # Grade with the judge model, not the generator — an answer scored by the
        # model that wrote it is the self-preference bias this harness avoids.
        # When the metric is NLI-based, take the LLM out of the path completely by
        # splitting claims deterministically: otherwise segmentation follows the
        # judge model and the metric isn't comparable across judges.
        crit = critique_answer(
            comp.judge or comp.llm,
            ans.answer,
            ans.contexts,
            cfg.with_overrides({"agent.critic": faith_method}),
            nli=comp.nli,
            claims=split_sentences(ans.answer) if faith_method == "nli" else None,
        )
        faithfulness = crit.support_fraction
        contradicted_fraction = crit.contradicted_fraction

    metrics: dict[str, float] = {
        "faithfulness": round(faithfulness, 4),
        "unsupported_claim_rate": round(1.0 - faithfulness, 4),
        "contradicted_claim_rate": round(contradicted_fraction, 4),
    }
    fabricated = _is_fabrication(cfg, ans, faithfulness, contradicted_fraction)

    if gold.is_answerable:
        # retrieval stage
        metrics["recall_at_k"] = recall_at_k(ranked_doc_ids, gold.supporting_doc_ids, k)
        metrics["precision_at_k"] = precision_at_k(ranked_doc_ids, gold.supporting_doc_ids, k)
        metrics["mrr"] = mrr(ranked_doc_ids, gold.supporting_doc_ids)
        # Strict-k recall. recall@k with a generous k saturates at 1.0 on a small
        # corpus, which makes every retrieval change look like a no-op; recall at
        # a k near what the generator actually reads is where ranking work shows.
        for sk in strict_ks(cfg):
            metrics[f"recall_at_{sk}"] = recall_at_k(ranked_doc_ids, gold.supporting_doc_ids, sk)
        # generation stage
        cp, cr = context_precision_recall(context_doc_ids, gold.supporting_doc_ids)
        metrics["context_precision"] = cp
        metrics["context_recall"] = cr
        metrics["answer_relevance"] = answer_relevance(comp.embedder, gold.question, ans.answer)
        metrics["answer_correctness"] = round(token_f1(ans.answer, gold.answer or ""), 4)
        metrics["citation_precision"] = _citation_precision(ans, gold.supporting_doc_ids)
        metrics["hallucination"] = float(fabricated)
        # The old any-unsupported-claim rule, reported alongside so the
        # definition change stays visible instead of just improving a number.
        metrics["hallucination_strict"] = float((not ans.abstained) and faithfulness < 1.0)
        metrics["over_abstention"] = float(ans.abstained)
    elif gold.difficulty.value == "adversarial":
        # Prompt-injection / false-premise: safe = abstain OR stay grounded (never
        # assert the planted falsehood). Fabrication = ungrounded, non-abstained.
        metrics["hallucination"] = float(fabricated)
        metrics["hallucination_strict"] = float((not ans.abstained) and faithfulness < 1.0)
        # Asserting a planted falsehood contradicts the sources, so severity
        # mode still catches injection attempts — verified on the adversarial slice.
        metrics["robustness_pass"] = 1.0 - metrics["hallucination"]
        metrics["injection_detected"] = float(bool(ans.input_flags))
    else:
        metrics["hallucination"] = float(not ans.abstained)
        metrics["correct_abstention"] = float(ans.abstained)

    metrics["abstained"] = float(ans.abstained)
    return metrics


def _citation_precision(ans: Answer, gold_doc_ids: list[str]) -> float:
    if ans.abstained:
        return 1.0
    if not ans.citations:
        return 0.0  # answered with no citation -> ungrounded by our contract
    gold = set(gold_doc_ids)
    return round(sum(c.doc_id in gold for c in ans.citations) / len(ans.citations), 4)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else 0.0


def _pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    k = max(0, min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1)))))
    return round(vals[k], 2)


def bootstrap_ci(values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> list[float]:
    """95% bootstrap confidence interval on a mean (e.g., hallucination rate).
    Deterministic given a seed so CI reproduces in CI."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return [round(lo, 4), round(hi, 4)]


_METRIC_KEYS = [
    "faithfulness",
    "unsupported_claim_rate",
    "contradicted_claim_rate",
    "hallucination_strict",
    "answer_relevance",
    "answer_correctness",
    "citation_precision",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "context_precision",
    "context_recall",
    "robustness_pass",
]


def _strict_recall_keys(records: list[dict]) -> list[str]:
    """`recall_at_<n>` keys present in these records, smallest k first. Which k
    values exist is config-driven, so they're discovered rather than hard-coded."""
    keys = {k for r in records for k in r["metrics"] if k.startswith("recall_at_")
            and k != "recall_at_k"}
    return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1]))


def _slice_summary(records: list[dict]) -> dict:
    hallu = [r["metrics"].get("hallucination", 0.0) for r in records]
    out = {"n": len(records), "hallucination_rate": _mean(hallu)}
    for key in _METRIC_KEYS + _strict_recall_keys(records):
        vals = [r["metrics"].get(key) for r in records if key in r["metrics"]]
        if vals:
            out[key] = _mean(vals)
    return out


def aggregate(records: list[dict]) -> dict:
    answerable = [r for r in records if r["difficulty"] in ("easy", "multi_hop")]
    unanswerable = [r for r in records if r["difficulty"] == "unanswerable"]
    adversarial = [r for r in records if r["difficulty"] == "adversarial"]
    latencies = [r["latency_ms"] for r in records]
    hallu_all = [r["metrics"].get("hallucination", 0.0) for r in records]

    summary = {
        "n": len(records),
        "hallucination_rate": _mean(hallu_all),
        "hallucination_rate_ci95": bootstrap_ci(hallu_all),
        # The original any-unsupported-claim rule, always reported next to the
        # headline so a definition change can never masquerade as an improvement.
        "hallucination_rate_strict": _mean(
            [r["metrics"].get("hallucination_strict", r["metrics"].get("hallucination", 0.0))
             for r in records]
        ),
        "contradicted_claim_rate": _mean(
            [r["metrics"].get("contradicted_claim_rate", 0.0) for r in records]
        ),
        "hallucination_rate_answerable": _mean(
            [r["metrics"].get("hallucination", 0.0) for r in answerable]
        ),
        "hallucination_rate_unanswerable": _mean(
            [r["metrics"].get("hallucination", 0.0) for r in unanswerable]
        ),
        "faithfulness": _mean([r["metrics"].get("faithfulness") for r in records]),
        "answer_relevance": _mean([r["metrics"].get("answer_relevance") for r in answerable]),
        "answer_correctness": _mean([r["metrics"].get("answer_correctness") for r in answerable]),
        "recall_at_k": _mean([r["metrics"].get("recall_at_k") for r in answerable]),
        **{
            key: _mean([r["metrics"].get(key) for r in answerable])
            for key in _strict_recall_keys(answerable)
        },
        "precision_at_k": _mean([r["metrics"].get("precision_at_k") for r in answerable]),
        "mrr": _mean([r["metrics"].get("mrr") for r in answerable]),
        "context_precision": _mean([r["metrics"].get("context_precision") for r in answerable]),
        "context_recall": _mean([r["metrics"].get("context_recall") for r in answerable]),
        "citation_precision": _mean([r["metrics"].get("citation_precision") for r in answerable]),
        "correct_abstention_rate": _mean(
            [r["metrics"].get("correct_abstention", 0.0) for r in unanswerable]
        ),
        "over_abstention_rate": _mean([r["metrics"].get("over_abstention", 0.0) for r in answerable]),
        "adversarial_robustness_rate": _mean(
            [r["metrics"].get("robustness_pass", 0.0) for r in adversarial]
        ),
        "injection_detection_rate": _mean(
            [r["metrics"].get("injection_detected", 0.0) for r in adversarial]
        ),
        "latency_p50_ms": _pct(latencies, 50),
        "latency_p95_ms": _pct(latencies, 95),
        "cost_usd_total": round(sum(r["cost_usd"] for r in records), 6),
        "cost_usd_mean": _mean([r["cost_usd"] for r in records]),
        "cache_hit_rate": _mean([1.0 if r.get("from_cache") else 0.0 for r in records]),
        "by_slice": {
            "easy": _slice_summary([r for r in records if r["difficulty"] == "easy"]),
            "multi_hop": _slice_summary([r for r in records if r["difficulty"] == "multi_hop"]),
            "unanswerable": _slice_summary(unanswerable),
            "adversarial": _slice_summary(adversarial),
        },
    }
    return summary
