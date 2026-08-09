"""Stage callbacks fire as the pipeline runs, not after it finishes.

The streaming endpoint used to run the whole query and then replay the recorded
trace with sleeps. That looks live but shows nothing until the user has already
waited for the complete answer — on a local model that is ~14 seconds of blank
panel. `Trace(on_stage=...)` is what makes it real.
"""

from arag.common.telemetry import Trace


def test_callback_fires_as_each_stage_closes():
    seen = []
    trace = Trace(on_stage=lambda t: seen.append((t.stage, len(trace.stages))))
    with trace.stage("retrieve", k=5):
        assert seen == []          # nothing emitted until the stage completes
    assert seen == [("retrieve", 1)]
    with trace.stage("generate"):
        pass
    assert [s for s, _ in seen] == ["retrieve", "generate"]


def test_callback_receives_timing_and_meta():
    seen = []
    trace = Trace(on_stage=seen.append)
    with trace.stage("rerank", candidates=12):
        pass
    assert seen[0].stage == "rerank"
    assert seen[0].meta == {"candidates": 12}
    assert seen[0].ms >= 0


def test_a_broken_listener_cannot_kill_the_query():
    """A failing SSE consumer must not take down the pipeline it is watching."""
    def explode(_):
        raise RuntimeError("listener died")

    trace = Trace(on_stage=explode)
    with trace.stage("retrieve"):
        pass
    assert len(trace.stages) == 1     # stage still recorded


def test_no_callback_is_the_default_and_still_records():
    trace = Trace()
    with trace.stage("retrieve"):
        pass
    assert [s.stage for s in trace.stages] == ["retrieve"]


def test_answer_query_accepts_a_stage_callback(mock_cfg, tmp_path):
    from arag.engine import answer_query, build_components
    from arag.ingest.index import build_index

    cfg = mock_cfg.with_overrides({
        "corpus_dir": "tests/fixtures/corpus",
        "vector_store.persist_dir": str(tmp_path / "idx"),
        "agent.enabled": True,
    })
    comp = build_components(cfg, store=build_index(cfg))
    seen = []
    ans = answer_query(comp, "What colour are widgets?", on_stage=seen.append)
    assert seen, "no stages streamed"
    # Every streamed stage is also in the final trace, in the same order.
    assert [s.stage for s in seen] == [s.stage for s in ans.trace]


# --------------------------------------------------- held-out eval split


def _gold():
    from eval.build_gold_set import load_gold
    return load_gold("data/eval/gold_qa.jsonl")


def test_dev_and_holdout_partition_the_set():
    """Every question lands in exactly one half — no leakage, nothing dropped."""
    from eval.run_eval import split_gold

    g = _gold()
    dev = {x.id for x in split_gold(g, "dev", 0.25, 20260804)}
    hold = {x.id for x in split_gold(g, "holdout", 0.25, 20260804)}
    assert not (dev & hold)
    assert dev | hold == {x.id for x in g}


def test_split_is_stratified_across_difficulties():
    """A holdout missing a slice couldn't measure that slice's overfitting."""
    from eval.run_eval import split_gold

    g = _gold()
    slices = {x.difficulty.value for x in g}
    for part in ("dev", "holdout"):
        got = {x.difficulty.value for x in split_gold(g, part, 0.25, 20260804)}
        assert got == slices, f"{part} missing {slices - got}"


def test_split_is_deterministic():
    """A partition that moved between runs would make every comparison
    meaningless — dev/holdout numbers would drift for no reason."""
    from eval.run_eval import split_gold

    g = _gold()
    a = [x.id for x in split_gold(g, "holdout", 0.25, 20260804)]
    assert a == [x.id for x in split_gold(g, "holdout", 0.25, 20260804)]


def test_a_different_seed_gives_a_different_partition():
    from eval.run_eval import split_gold

    g = _gold()
    a = {x.id for x in split_gold(g, "holdout", 0.25, 20260804)}
    b = {x.id for x in split_gold(g, "holdout", 0.25, 999)}
    assert a != b


def test_all_is_the_default_and_returns_everything():
    from eval.run_eval import split_gold

    g = _gold()
    assert len(split_gold(g, "all", 0.25, 20260804)) == len(g)
