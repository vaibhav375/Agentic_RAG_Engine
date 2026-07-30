"""What counts as a hallucination.

The original rule flagged a record if *any* claim wasn't fully entailed, so
"correct, grounded answer plus one aside the sources don't mention" scored
identically to an invented answer. Five measured configurations could not move
the resulting flag rate out of 0.545-0.727, because it was measuring elaboration
rather than fabrication (docs/local-mode-eval.md).

Severity mode separates the two failures: the sources *contradict* the claim
(fabrication) versus the sources are merely silent (an aside). These tests pin
that distinction, and pin that the permissiveness is bounded — a mostly-ungrounded
answer must still be flagged.
"""

from arag.common.schemas import Answer
from eval.metrics import _is_fabrication


def _cfg(mock_cfg, **over):
    return mock_cfg.with_overrides(over)


def _ans(abstained=False, text="an answer"):
    return Answer(query="q", answer=text, abstained=abstained)


# ------------------------------------------------------------ severity mode


def test_correct_answer_with_one_unsupported_aside_is_not_a_hallucination(mock_cfg):
    """The case that motivated the change: core grounded, one aside beyond it."""
    assert _is_fabrication(mock_cfg, _ans(), faithfulness=0.75, contradicted_fraction=0.0) is False


def test_contradicted_claim_is_a_hallucination(mock_cfg):
    """The sources say otherwise -> fabrication, however grounded the rest is."""
    assert _is_fabrication(mock_cfg, _ans(), faithfulness=0.9, contradicted_fraction=0.25) is True


def test_mostly_ungrounded_answer_is_a_hallucination(mock_cfg):
    """Severity must not become 'anything goes' — a wrong core still counts."""
    assert _is_fabrication(mock_cfg, _ans(), faithfulness=0.25, contradicted_fraction=0.0) is True


def test_support_threshold_is_the_boundary(mock_cfg):
    cfg = _cfg(mock_cfg, **{"eval.hallucination.min_support_fraction": 0.5})
    assert _is_fabrication(cfg, _ans(), faithfulness=0.5, contradicted_fraction=0.0) is False
    assert _is_fabrication(cfg, _ans(), faithfulness=0.49, contradicted_fraction=0.0) is True


def test_fully_grounded_answer_is_never_a_hallucination(mock_cfg):
    assert _is_fabrication(mock_cfg, _ans(), faithfulness=1.0, contradicted_fraction=0.0) is False


def test_abstention_is_never_a_hallucination(mock_cfg):
    assert _is_fabrication(mock_cfg, _ans(abstained=True), 0.0, 1.0) is False


# -------------------------------------------------------------- strict mode


def test_strict_mode_still_flags_any_unsupported_claim(mock_cfg):
    """The old definition stays reproducible for comparison."""
    cfg = _cfg(mock_cfg, **{"eval.hallucination.mode": "strict"})
    assert _is_fabrication(cfg, _ans(), faithfulness=0.75, contradicted_fraction=0.0) is True
    assert _is_fabrication(cfg, _ans(), faithfulness=1.0, contradicted_fraction=0.0) is False


def test_the_two_modes_disagree_exactly_on_asides(mock_cfg):
    strict = _cfg(mock_cfg, **{"eval.hallucination.mode": "strict"})
    aside = dict(faithfulness=0.8, contradicted_fraction=0.0)
    fabrication = dict(faithfulness=0.8, contradicted_fraction=0.2)
    assert _is_fabrication(strict, _ans(), **aside) is True
    assert _is_fabrication(mock_cfg, _ans(), **aside) is False
    # They agree that a contradiction is a hallucination.
    assert _is_fabrication(strict, _ans(), **fabrication) is True
    assert _is_fabrication(mock_cfg, _ans(), **fabrication) is True


# --------------------------------------------------- the contradiction signal


def test_critic_separates_contradiction_from_silence():
    """The NLI signal must distinguish 'sources say otherwise' from 'sources are
    silent' — the whole basis of severity mode. Uses a stub NLI so the test is
    deterministic and offline; the real deberta model scores a direct
    contradiction at 1.000 and an unrelated aside at 0.001."""
    from arag.common.schemas import Chunk, RetrievedChunk
    from arag.providers.base import NLIResult
    from arag.providers.llm import MockLLM

    class _StubNLI:
        def entail(self, premise, hypothesis):
            if "red" in hypothesis:      # contradicts the premise
                return NLIResult(entailment=0.0, neutral=0.0, contradiction=1.0)
            if "blue" in hypothesis:     # entailed
                return NLIResult(entailment=1.0, neutral=0.0, contradiction=0.0)
            return NLIResult(entailment=0.0, neutral=0.99, contradiction=0.01)  # aside

    from arag.agent.critic import critique_answer
    from arag.common.config import load_config

    cfg = load_config("config/config.yaml").with_overrides({"agent.critic": "nli"})
    ctx = [RetrievedChunk(chunk=Chunk(chunk_id="c0", doc_id="d",
                                     text="The default color is blue."))]

    contra = critique_answer(MockLLM(), "x", ctx, cfg, nli=_StubNLI(),
                             claims=["The default color is red."])
    aside = critique_answer(MockLLM(), "x", ctx, cfg, nli=_StubNLI(),
                            claims=["Colors are configurable per widget."])
    assert contra.contradicted_fraction == 1.0
    assert aside.contradicted_fraction == 0.0      # unsupported, but not contradicted
    assert aside.support_fraction == 0.0           # still reported as unsupported


# --------------------------------------------------------- premise granularity


def test_premise_units_split_chunks_without_blank_lines():
    """Chunk text arrives with line wrapping but no blank lines, so paragraph
    splitting alone is inert — premises come from sentences and adjacent pairs."""
    from arag.agent.critic import _premise_units

    text = ("If the incoming body is missing a required field, Breeze returns a 422\n"
            "response whose body lists each invalid field. Validation runs before your\n"
            "handler is called, so the handler only ever sees valid data.")
    units = _premise_units(text)
    assert len(units) > 1
    # Individual sentences and the adjacent-pair window are all available.
    assert any("Validation runs before" in u and "422" not in u for u in units)
    assert any("422" in u and "Validation runs before" in u for u in units)
    # Line wrapping is normalized away.
    assert all("\n" not in u for u in units)


def test_premise_units_chunk_granularity_is_verbatim():
    from arag.agent.critic import _premise_units

    text = "One sentence. Another sentence."
    assert _premise_units(text, "chunk") == [text]


def test_premise_units_handles_degenerate_input():
    from arag.agent.critic import _premise_units

    assert _premise_units("") == []
    assert _premise_units("Short.") == ["Short."]


# ------------------------------------------------------ clause decomposition


def test_leading_conditional_is_stripped_so_the_assertion_is_judgeable():
    """Entailment models score compound conditionals far worse than the clause
    alone — measured 0.000 vs 0.992 against a premise that supports it."""
    from arag.providers.base import atomic_claims

    claims = atomic_claims(
        "If a JSON body is missing a required field, Breeze returns a 422 response."
    )
    assert claims == ["Breeze returns a 422 response."]


def test_trailing_clause_becomes_its_own_claim():
    from arag.providers.base import atomic_claims

    claims = atomic_claims("A query parameter without a default is required, "
                           "so omitting it returns a 422.")
    assert len(claims) == 2
    assert claims[0].startswith("A query parameter")
    assert "422" in claims[1]


def test_plain_sentences_are_untouched():
    from arag.providers.base import atomic_claims

    text = "Use app.url_for to build a URL for a named route."
    assert atomic_claims(text) == [text]


def test_short_main_clause_keeps_the_whole_sentence():
    """Stripping is only worth it when a real clause survives — otherwise the
    condition carries the meaning and dropping it would lose information."""
    from arag.providers.base import atomic_claims

    assert atomic_claims("When the cache is warm, lookups are fast.") == [
        "When the cache is warm, lookups are fast."
    ]


def test_decomposition_is_deterministic_and_offline():
    from arag.providers.base import atomic_claims

    text = "If X happens, the system returns a 422 response, so the handler is skipped."
    assert atomic_claims(text) == atomic_claims(text)


def test_metric_claims_honours_the_config(mock_cfg):
    from eval.metrics import _metric_claims

    text = "If a body is missing a field, Breeze returns a 422 response."
    assert _metric_claims(mock_cfg, text) == ["Breeze returns a 422 response."]
    sentence_mode = mock_cfg.with_overrides({"eval.claim_decomposition": "sentence"})
    assert _metric_claims(sentence_mode, text) == [text]
