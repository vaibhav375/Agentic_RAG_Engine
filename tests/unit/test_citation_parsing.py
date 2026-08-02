"""Citation parsing for real-model answers.

Chunk ids contain `::`. The original pattern (`[A-Za-z0-9_\\-]+`) could not match
them, so with real models no citation was ever extracted, citation_precision
scored 0.0 on every answered record, and the raw markers stayed in the answer
text where they polluted the claims fed to the NLI metric. Mock mode hid it
entirely because MockLLM constructs citations instead of parsing them.
"""

from arag.providers.llm import parse_citations

IDS = ["01_routing::2", "02_query_params::1", "03_request_body::4"]


def test_chunk_ids_with_colons_are_parsed():
    got = parse_citations("Declare it in braces. [01_routing::2]", IDS)
    assert got.cited_chunk_ids == ["01_routing::2"]
    assert "[" not in got.text


def test_marker_is_removed_from_the_answer_text():
    """Left in place, the marker becomes a 'claim' the NLI metric must judge."""
    got = parse_citations("A query parameter is required. [02_query_params::1]", IDS)
    assert got.text == "A query parameter is required."


def test_stray_c_prefix_is_tolerated():
    """Models imitate the prompt's `[c3]` example and emit `[c01_routing::2]`."""
    got = parse_citations("Routes match in order. [c01_routing::2]", IDS)
    assert got.cited_chunk_ids == ["01_routing::2"]
    assert "c01_routing" not in got.text


def test_legitimate_bracketed_content_survives():
    """`list[str]` appears in this corpus — a permissive pattern would eat it."""
    got = parse_citations("Annotate the argument as `tags: list[str]`. [02_query_params::1]", IDS)
    assert "list[str]" in got.text
    assert got.cited_chunk_ids == ["02_query_params::1"]


def test_unknown_ids_are_left_alone():
    got = parse_citations("See [something_else::9] for details.", IDS)
    assert got.cited_chunk_ids == []
    assert "[something_else::9]" in got.text


def test_multiple_and_duplicate_citations():
    got = parse_citations(
        "First. [01_routing::2] Second. [02_query_params::1][01_routing::2]", IDS)
    assert got.cited_chunk_ids == ["01_routing::2", "02_query_params::1"]
    assert "[" not in got.text


def test_punctuation_is_tidied_after_stripping():
    got = parse_citations("Validation runs first [03_request_body::4] .", IDS)
    assert got.text == "Validation runs first."


def test_no_citations_at_all():
    got = parse_citations("A bare answer with no citation.", IDS)
    assert got.cited_chunk_ids == []
    assert got.text == "A bare answer with no citation."


# ------------------------------------------------- batched NLI equivalence


def test_batched_and_looped_nli_agree(mock_cfg):
    """Batching is an optimization: it must not change a single score.

    Verified numerically against the real cross-encoder too (max abs diff
    1.4e-5); this pins the wiring so a future refactor can't silently reorder
    claims against premises.
    """
    from arag.agent.critic import _score_claims
    from arag.common.schemas import Chunk, RetrievedChunk
    from arag.providers.base import MockNLI, NLIModel

    class _LoopOnly(NLIModel):
        """Implements only the required method — no entail_batch."""

        def __init__(self):
            self._inner = MockNLI()

        def entail(self, premise, hypothesis):
            return self._inner.entail(premise, hypothesis)

    ctx = [
        RetrievedChunk(chunk=Chunk(chunk_id="c0", doc_id="d", text=(
            "The default color is blue. Widgets ship with a warranty. "
            "Batteries last a thousand hours."))),
        RetrievedChunk(chunk=Chunk(chunk_id="c1", doc_id="d", text=(
            "GZip compresses responses over 500 bytes."))),
    ]
    claims = ["The default color is blue.", "Batteries last a thousand hours.",
              "Responses are compressed when large."]

    batched = _score_claims(MockNLI(), ctx, claims)
    looped = _score_claims(_LoopOnly(), ctx, claims)
    assert batched == looped
    assert len(batched) == len(claims)


def test_score_claims_handles_empty_input(mock_cfg):
    from arag.agent.critic import _score_claims
    from arag.providers.base import MockNLI

    assert _score_claims(MockNLI(), [], ["a claim here"]) == [(0.0, 0.0)]
    assert _score_claims(MockNLI(), [], []) == []
