from arag.agent.critic import critique_answer
from arag.common.schemas import Chunk, RetrievedChunk
from arag.providers.base import MockNLI
from arag.providers.llm import MockLLM


def _ctx(text, cid="c0"):
    return [RetrievedChunk(chunk=Chunk(chunk_id=cid, doc_id="d", text=text))]


def test_grounded_answer_is_supported(mock_cfg):
    ctx = _ctx("The default widget color is blue.")
    crit = critique_answer(
        MockLLM(), "The default widget color is blue.", ctx, mock_cfg, nli=MockNLI()
    )
    assert crit.supported is True
    assert crit.support_fraction >= 0.5


def test_fabricated_answer_is_unsupported(mock_cfg):
    ctx = _ctx("The default widget color is blue.")
    crit = critique_answer(
        MockLLM(),
        "Widgets support real-time websocket streaming enabled by default.",
        ctx,
        mock_cfg,
        nli=MockNLI(),
    )
    assert crit.supported is False
    assert crit.missing_info is not None


def test_both_mode_is_conservative(mock_cfg):
    cfg = mock_cfg.with_overrides({"agent.critic": "both"})
    ctx = _ctx("The default widget color is blue.")
    crit = critique_answer(MockLLM(), "The maximum battery capacity is 5000 mAh.", ctx, cfg, nli=MockNLI())
    # Claim not in context -> not supported under conservative AND of LLM+NLI.
    assert crit.supported is False


class _CountingLLM(MockLLM):
    """Records whether claim extraction was delegated to the model."""

    def __init__(self):
        self.extract_calls = 0

    def extract_claims(self, answer):
        self.extract_calls += 1
        return super().extract_claims(answer)


def test_supplied_claims_bypass_llm_extraction(mock_cfg):
    """The eval metric passes a deterministic split so claim segmentation doesn't
    follow whichever model is judging — otherwise faithfulness isn't comparable
    across judges (docs/local-mode-eval.md)."""
    llm = _CountingLLM()
    ctx = _ctx("The default widget color is blue.")
    crit = critique_answer(
        llm, "The default widget color is blue.", ctx, mock_cfg,
        nli=MockNLI(), claims=["The default widget color is blue."],
    )
    assert llm.extract_calls == 0
    assert crit.support_fraction == 1.0


def test_llm_extraction_used_when_explicitly_configured(mock_cfg):
    """`claim_extraction: llm` asks the model. Set explicitly rather than relying
    on the default, which is now `clause`."""
    llm = _CountingLLM()
    cfg = mock_cfg.with_overrides({"agent.claim_extraction": "llm"})
    critique_answer(llm, "The default widget color is blue.", _ctx("x"), cfg, nli=MockNLI())
    assert llm.extract_calls == 1


def test_clause_extraction_makes_no_llm_call(mock_cfg):
    """The default. It removes an LLM generation call from every loop iteration —
    measured 65.4s -> 42.6s p50 on qwen2.5:7b, while also improving faithfulness."""
    llm = _CountingLLM()
    cfg = mock_cfg.with_overrides({"agent.claim_extraction": "clause"})
    crit = critique_answer(llm, "The default widget color is blue.", _ctx("x"),
                           cfg, nli=MockNLI())
    assert llm.extract_calls == 0
    assert crit.claims  # it still produced claims, just without the model


def test_claim_segmentation_drives_support_fraction(mock_cfg):
    """Two claims, one grounded: the fraction reflects the supplied segmentation,
    which is exactly why it must not vary by judge model."""
    ctx = _ctx("The default widget color is blue.")
    crit = critique_answer(
        MockLLM(), "irrelevant", ctx, mock_cfg.with_overrides({"agent.critic": "nli"}),
        nli=MockNLI(),
        claims=["The default widget color is blue.", "Widgets stream over websockets."],
    )
    assert 0.0 < crit.support_fraction < 1.0
