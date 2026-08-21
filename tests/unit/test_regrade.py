"""Semantic grading runs after the eval, not inside it.

The judge that scored 9/9 on the hand-verified labels is larger than the
generator, and loading both in one process makes Ollama swap models per record —
the thrashing that once produced a phantom "7b costs 11x more" result. Grading a
finished results file avoids the conflict entirely, and leaves the run's latency
numbers untouched because they were measured before this pass existed.
"""

from __future__ import annotations

import json

import pytest

from eval.regrade import regrade


def _results(tmp_path, records):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"summary": {"answer_correctness": 0.3}, "detail": records}))
    return str(path)


def _rec(rid, difficulty="easy", abstained=False, **kw):
    return {
        "id": rid, "difficulty": difficulty, "abstained": abstained,
        "question": "q?", "gold_answer": "blue", "predicted": "blue",
        "metrics": {"answer_correctness": 0.5}, "latency_ms": 1234.5, **kw,
    }


def test_it_grades_answered_questions_and_writes_the_file(mock_cfg, tmp_path):
    path = _results(tmp_path, [_rec("e1"), _rec("m1", "multi_hop")])
    out = regrade(path, mock_cfg)

    assert out["graded"] == 2
    saved = json.loads(open(path).read())
    for rec in saved["detail"]:
        assert "answer_equivalence" in rec["metrics"]
    assert "answer_equivalence" in saved["summary"]
    assert saved["summary"]["answer_equivalence_n"] == 2


def test_abstentions_and_unanswerable_are_not_scored(mock_cfg, tmp_path):
    """Neither has a factual claim to compare, so a score would be invented."""
    path = _results(tmp_path, [
        _rec("e1"),
        _rec("e2", abstained=True),
        _rec("u1", "unanswerable"),
        _rec("x1", "adversarial"),
    ])
    out = regrade(path, mock_cfg)

    assert out["graded"] == 1
    saved = json.loads(open(path).read())
    scored = {r["id"] for r in saved["detail"] if "answer_equivalence" in r["metrics"]}
    assert scored == {"e1"}
    # The denominator is recorded, because a mean whose base moved would read as
    # a quality change.
    assert saved["summary"]["answer_equivalence_n"] == 1


class _FakeJudge:
    def judge_equivalence(self, question, gold, candidate):
        return True


def test_the_judge_used_is_recorded(mock_cfg, tmp_path, monkeypatch):
    """A score is not comparable across judges, so the file has to say which one."""
    import eval.regrade as mod

    monkeypatch.setattr(mod, "make_llm", lambda cfg, role=None: _FakeJudge())
    path = _results(tmp_path, [_rec("e1")])
    regrade(path, mock_cfg, judge_model="some-big-model")
    assert json.loads(open(path).read())["summary"]["answer_equivalence_judge"] == "some-big-model"


def test_asking_for_a_real_judge_and_getting_the_mock_one_raises(mock_cfg, tmp_path):
    """This shipped a confident fake number once and must never do it silently.

    config.yaml carries `mode: mock`, so `make_llm(role="judge")` returned MockLLM
    and every score came from the lexical fallback's 0.6 cutoff — while the summary
    recorded "qwen2.5:7b". The tell was e02: "The default type of a path parameter
    is a string" against gold "A string.", scored not-equivalent.
    """
    path = _results(tmp_path, [_rec("e1")])
    with pytest.raises(ValueError, match="mock"):
        regrade(path, mock_cfg, judge_model="qwen2.5:7b")


def test_no_judge_requested_still_works_offline(mock_cfg, tmp_path):
    """Without an explicit judge the mock fallback is legitimate — it is a
    deterministic stand-in, not a claim about quality."""
    path = _results(tmp_path, [_rec("e1")])
    assert regrade(path, mock_cfg)["graded"] == 1


def test_latency_is_untouched(mock_cfg, tmp_path):
    """Grading afterwards must not contaminate what the run measured."""
    path = _results(tmp_path, [_rec("e1")])
    regrade(path, mock_cfg)
    assert json.loads(open(path).read())["detail"][0]["latency_ms"] == 1234.5


def test_regrading_twice_is_idempotent(mock_cfg, tmp_path):
    path = _results(tmp_path, [_rec("e1"), _rec("e2", abstained=True)])
    first = regrade(path, mock_cfg)
    second = regrade(path, mock_cfg)
    assert first == second


def test_a_file_with_no_detail_fails_loudly(mock_cfg, tmp_path):
    """Silently writing a summary over nothing would look like a result."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"summary": {}, "detail": []}))
    with pytest.raises(ValueError, match="no detail"):
        regrade(str(path), mock_cfg)
