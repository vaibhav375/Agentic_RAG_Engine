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

import re

from arag.common.schemas import ClaimJudgement, CritiqueResult, RetrievedChunk
from arag.providers.base import LanguageModel, NLIModel, split_sentences


def critique_answer(
    llm: LanguageModel,
    answer_text: str,
    contexts: list[RetrievedChunk],
    cfg,
    nli: NLIModel | None = None,
    claims: list[str] | None = None,
) -> CritiqueResult:
    """Judge claim-level support. `claims` overrides LLM claim extraction.

    The eval harness passes a deterministic split when its faithfulness metric is
    NLI-based: otherwise claim *segmentation* follows whichever model is judging,
    so the same answer decomposes differently under a different judge and
    `faithfulness`/`hallucination_rate` stop being comparable across runs. That
    silently confounded the first local-vs-mock comparison
    (see docs/local-mode-eval.md).
    """
    mode = cfg.get("agent.critic", "llm")
    support_threshold = float(cfg.get("agent.support_threshold", 0.5))
    nli_thresh = float(cfg.get("agent.nli_entail_threshold", 0.5))

    context_text = "\n".join(rc.chunk.text for rc in contexts)
    if claims is None:
        claims = llm.extract_claims(answer_text)
    if not claims:
        return CritiqueResult(supported=False, support_fraction=0.0, missing_info=None)

    contra_thresh = float(cfg.get("agent.nli_contradiction_threshold", 0.5))
    # All claims x all premise units in one batched NLI call (see _score_claims).
    nli_scores = (
        _score_claims(nli, contexts, claims, cfg.get("agent.nli_premise", "paragraph"))
        if mode in ("nli", "both") and nli is not None
        else [(None, None)] * len(claims)
    )
    judgements: list[ClaimJudgement] = []
    for claim, (nli_entail, nli_contra) in zip(claims, nli_scores, strict=True):
        llm_ok = nli_ok = None
        score = None
        contradiction = None
        if mode in ("llm", "both"):
            s, reason, conf = llm.judge_claim(claim, context_text)
            llm_ok = s
            score = conf
        if nli_entail is not None:
            nli_ok = nli_entail >= nli_thresh
            score = nli_entail if score is None else score
            # Only meaningful when the claim isn't entailed; see the helper.
            contradiction = 0.0 if nli_ok else nli_contra

        if mode == "llm":
            supported = bool(llm_ok)
            method = "llm"
        elif mode == "nli":
            supported = bool(nli_ok)
            method = "nli"
        else:  # both -> conservative AND
            supported = bool(llm_ok) and bool(nli_ok)
            method = "both"

        contradicted = contradiction is not None and contradiction >= contra_thresh
        judgements.append(
            ClaimJudgement(
                claim=claim,
                supported=supported,
                reason=None if supported else (
                    "contradicted by retrieved context" if contradicted
                    else "not stated in retrieved context"
                ),
                method=method,
                score=score,
                contradiction=contradiction,
                contradicted=contradicted,
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
        contradicted_fraction=round(
            sum(j.contradicted for j in judgements) / len(judgements), 4
        ),
    )


def _max_entailment(nli: NLIModel, contexts: list[RetrievedChunk], claim: str) -> float:
    """A claim is supported if ANY retrieved passage entails it."""
    return _entailment_and_contradiction(nli, contexts, claim)[0]


def _premise_units(text: str, granularity: str = "paragraph") -> list[str]:
    """Split a retrieved chunk into premises an NLI model can actually judge.

    NLI cross-encoders are trained on sentence-pair-scale inputs. Handing one a
    multi-paragraph chunk pushes it out of distribution and it drifts toward
    "contradiction" — measured on this corpus with `nli-deberta-v3-base` and a
    claim quoted almost verbatim from the source:

        whole 1200-char chunk : entailment 0.138, contradiction 0.499  (wrong)
        the paragraph in it   : entailment 0.993, contradiction 0.001  (right)

    So the claim is scored against each paragraph and the best match wins.
    `granularity: chunk` restores the old whole-chunk behavior for comparison.
    """
    if granularity == "chunk":
        return [text]
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    # Blank lines don't survive chunking, so split on sentences and slide a small
    # window over them: a claim usually needs one or two adjacent sentences as its
    # premise, and a window keeps the pair together without dragging in the whole
    # chunk.
    sentences = [s for s in split_sentences(flat) if s.strip()]
    if len(sentences) <= 1:
        return [flat]
    units = list(sentences)
    units += [f"{a} {b}" for a, b in zip(sentences, sentences[1:], strict=False)]
    # Drop headings and stubs — too short to entail anything on their own.
    return [u for u in units if len(u.split()) >= 4] or [flat]


def _score_claims(
    nli: NLIModel,
    contexts: list[RetrievedChunk],
    claims: list[str],
    granularity: str = "paragraph",
) -> list[tuple[float, float]]:
    """(entailment, contradiction) per claim, scored in ONE batched NLI call.

    Every claim is checked against every premise unit of every retrieved chunk —
    ~94 forward passes per query on this corpus. Issued one at a time that is
    ~2.9x slower than a single batch, for identical scores.
    """
    premises = [p for rc in contexts for p in _premise_units(rc.chunk.text, granularity)]
    if not premises or not claims:
        return [(0.0, 0.0) for _ in claims]

    pairs = [(p, c) for c in claims for p in premises]
    # `entail` is the only method the NLIModel contract requires, so a backend
    # that implements just that must keep working — batching is an optimization,
    # not a new requirement.
    batch = getattr(nli, "entail_batch", None)
    results = batch(pairs) if callable(batch) else [nli.entail(p, h) for p, h in pairs]

    out: list[tuple[float, float]] = []
    width = len(premises)
    for i in range(len(claims)):
        best_entail = best_contra = 0.0
        for res in results[i * width:(i + 1) * width]:
            # Same selection as the unbatched path: contradiction is read off the
            # best-matching premise, and only stands in when nothing entails.
            if res.entailment > best_entail:
                best_entail, best_contra = res.entailment, res.contradiction
            elif best_entail == 0.0:
                best_contra = max(best_contra, res.contradiction)
        out.append((best_entail, best_contra))
    return out


def _entailment_and_contradiction(
    nli: NLIModel, contexts: list[RetrievedChunk], claim: str, granularity: str = "paragraph"
) -> tuple[float, float]:
    """Best entailment and best contradiction across the retrieved passages.

    Keeping contradiction is what lets a caller separate two very different
    failures that "unsupported" alone conflates:
      * *contradicted* — the sources say otherwise. That is fabrication.
      * *neutral* — the sources simply don't cover it. An aside beyond the
        context, which may well be true.
    Contradiction is read off the *best-matching* premise, not maxed across all
    of them: "the passage most relevant to this claim says otherwise" is evidence
    of fabrication, whereas an unrelated passage disagreeing is noise.
    """
    best_entail = 0.0
    best_contra = 0.0
    for rc in contexts:
        for premise in _premise_units(rc.chunk.text, granularity):
            res = nli.entail(premise, claim)
            if res.entailment > best_entail:
                best_entail = res.entailment
                best_contra = res.contradiction
            elif best_entail == 0.0:
                # Nothing entails it yet — keep the strongest disagreement seen.
                best_contra = max(best_contra, res.contradiction)
    return best_entail, best_contra
