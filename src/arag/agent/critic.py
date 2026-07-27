"""Claim-level support critic.

Decomposes the answer into atomic claims and judges whether each is supported by
the retrieved context. Two independent signals are supported:

- `llm`  — LLM-as-judge (the generator's model or a separate judge model).
- `nli`  — a natural-language-inference entailment check (premise=context,
           hypothesis=claim). This is a non-LLM cross-check that guards against
           LLM-judge self-preference bias, a well-documented failure mode.
- `both` — a claim counts as supported only if BOTH agree (conservative), which
           is what drives the hallucination-rate reduction.

Output feeds the self-correction loop: if too few claims are supported, the loop
reformulates and retries; if it still can't ground the answer, it abstains.
"""

from __future__ import annotations

from arag.common.schemas import ClaimJudgement, CritiqueResult, RetrievedChunk
from arag.providers.base import LanguageModel, NLIModel


def critique_answer(
    llm: LanguageModel,
    answer_text: str,
    contexts: list[RetrievedChunk],
    cfg,
    nli: NLIModel | None = None,
) -> CritiqueResult:
    mode = cfg.get("agent.critic", "llm")
    support_threshold = float(cfg.get("agent.support_threshold", 0.5))
    nli_thresh = float(cfg.get("agent.nli_entail_threshold", 0.5))

    context_text = "\n".join(rc.chunk.text for rc in contexts)
    claims = llm.extract_claims(answer_text)
    if not claims:
        return CritiqueResult(supported=False, support_fraction=0.0, missing_info=None)

    judgements: list[ClaimJudgement] = []
    for claim in claims:
        llm_ok = nli_ok = None
        score = None
        if mode in ("llm", "both"):
            s, reason, conf = llm.judge_claim(claim, context_text)
            llm_ok = s
            score = conf
        if mode in ("nli", "both") and nli is not None:
            res = _max_entailment(nli, contexts, claim)
            nli_ok = res >= nli_thresh
            score = res if score is None else score

        if mode == "llm":
            supported = bool(llm_ok)
            method = "llm"
        elif mode == "nli":
            supported = bool(nli_ok)
            method = "nli"
        else:  # both -> conservative AND
            supported = bool(llm_ok) and bool(nli_ok)
            method = "both"

        judgements.append(
            ClaimJudgement(
                claim=claim,
                supported=supported,
                reason=None if supported else "unsupported by retrieved context",
                method=method,
                score=score,
            )
        )

    n_supported = sum(j.supported for j in judgements)
    fraction = n_supported / len(judgements)
    unsupported = [j.claim for j in judgements if not j.supported]
    missing = "; ".join(unsupported) if unsupported else None

    return CritiqueResult(
        supported=fraction >= support_threshold,
        support_fraction=round(fraction, 4),
        claims=judgements,
        missing_info=missing,
    )


def _max_entailment(nli: NLIModel, contexts: list[RetrievedChunk], claim: str) -> float:
    """A claim is supported if ANY retrieved passage entails it."""
    best = 0.0
    for rc in contexts:
        res = nli.entail(rc.chunk.text, claim)
        best = max(best, res.entailment)
    return best
