"""Semantic answer comparison — the job token-F1 measurably cannot do.

`answer_correctness` is token-F1, and on 9 hand-verified labels it ranked them
close to backwards: the one wrong answer scored highest (0.632 — it matches gold's
wording and swaps only the entity that decides it) while a fully correct answer
scored 0.000. NLI bidirectional entailment scored ~0.00 for everything and
embedding cosine put the wrong answer above four correct ones.

An LLM judge does it: qwen2.5:3b agrees 7/9 and qwen2.5:7b agrees 9/9, neither
ever accepting a wrong answer. See eval/experiments/judge_validation.py.
"""

from __future__ import annotations

from arag.providers.llm import MockLLM, PromptLLM


class _Canned(PromptLLM):
    """PromptLLM with the network replaced, to test parsing rather than a model."""

    def __init__(self, reply: str):
        self._reply = reply

    def _complete(self, system: str, user: str) -> str:  # type: ignore[override]
        self.last_user = user
        return self._reply


def test_a_json_verdict_is_parsed_both_ways():
    assert _Canned('{"equivalent": true, "reason": "same fact"}').judge_equivalence(
        "q", "True.", "The field defaults to True."
    )
    assert not _Canned('{"equivalent": false, "reason": "different entity"}').judge_equivalence(
        "q", "GZipMiddleware wraps most tightly.", "CORSMiddleware wraps most tightly."
    )


def test_prose_around_the_json_is_tolerated():
    """Small models wrap JSON in commentary; that must not read as a failure."""
    reply = 'Sure! Here is my verdict:\n{"equivalent": true, "reason": "ok"}\nHope that helps.'
    assert _Canned(reply).judge_equivalence("q", "True.", "Defaults to True.")


def test_an_unparseable_reply_counts_as_not_equivalent():
    """This feeds a quality metric, so a judge failure must read as a miss rather
    than silently inflate the score."""
    assert not _Canned("I think they're basically the same").judge_equivalence("q", "a", "a")
    assert not _Canned("").judge_equivalence("q", "a", "a")


def test_the_prompt_carries_question_gold_and_candidate():
    llm = _Canned('{"equivalent": true}')
    llm.judge_equivalence("Are routes authenticated?", "No; every route is public.", "Not by default.")
    for part in ("Are routes authenticated?", "every route is public", "Not by default."):
        assert part in llm.last_user


def test_the_mock_fallback_is_deterministic_and_offline():
    """Every backend keeps a mock fallback so the harness runs with no network."""
    m = MockLLM()
    assert m.judge_equivalence("q", "the default widget color is blue",
                               "the default widget color is blue")
    assert not m.judge_equivalence("q", "the default widget color is blue",
                                   "kubernetes autoscaling requires a metrics server")


def test_the_mock_fallback_handles_empty_answers():
    m = MockLLM()
    assert m.judge_equivalence("q", "", "")
    assert not m.judge_equivalence("q", "something", "")
