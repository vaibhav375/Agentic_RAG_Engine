"""End-to-end query on the tiny fixture corpus, and a mini-ablation sanity check
that self-correction abstains on an unanswerable question."""

from arag.engine import answer_query, build_components
from arag.ingest.index import build_index


def _components(cfg):
    store = build_index(cfg)
    return build_components(cfg, store=store)


def test_answerable_query_returns_grounded_answer(mock_cfg):
    comp = _components(mock_cfg)
    ans = answer_query(comp, "What is the default widget color?")
    assert not ans.abstained
    assert "blue" in ans.answer.lower()
    assert any(rc.chunk.doc_id == "widgets" for rc in ans.contexts)


def test_agent_abstains_on_unanswerable(mock_cfg):
    cfg = mock_cfg.with_overrides(
        {
            "agent.enabled": True,
            "agent.critic": "both",
            "retrieval.use_hybrid": True,
            "retrieval.use_rerank": True,
        }
    )
    comp = _components(cfg)
    ans = answer_query(comp, "How do I configure OAuth2 device flow for widgets?")
    assert ans.abstained is True


def test_baseline_does_not_abstain_structurally(mock_cfg):
    # With the agent disabled, the baseline attempts an answer (no critic gate).
    comp = _components(mock_cfg)
    ans = answer_query(comp, "How do I configure OAuth2 device flow for widgets?")
    assert ans.abstained is False


def test_cache_returns_same_answer_faster(mock_cfg):
    cfg = mock_cfg.with_overrides({"cache.enabled": True})
    comp = _components(cfg)
    a1 = answer_query(comp, "What is the default battery capacity?")
    a2 = answer_query(comp, "What is the default battery capacity?")
    assert a2.from_cache is True
    assert a1.answer == a2.answer
