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
