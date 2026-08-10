"""Stage callbacks fire as the pipeline runs, not after it finishes.

The streaming endpoint used to run the whole query and then replay the recorded
trace with sleeps. That looks live but shows nothing until the user has already
waited for the complete answer — on a local model that is ~14 seconds of blank
panel. `Trace(on_stage=...)` is what makes it real.
"""

import json

import pytest

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


def test_split_report_covers_the_whole_set_exactly_once(mock_cfg, tmp_path):
    """The paired run must cost no more than one full run, and must not double-
    count: dev + holdout together ask every gold question exactly once, so the
    pooled summary has to match a plain `split: all` run's question count."""
    from eval.run_eval import run_eval_split_report

    cfg = mock_cfg.with_overrides(
        {"eval.results_dir": str(tmp_path), "vector_store.persist_dir": str(tmp_path / "idx")}
    )
    out = run_eval_split_report(cfg, tag="t")

    assert out["summary"]["n"] == len(_gold())
    assert set(out["splits"]) == {"dev", "holdout"}
    assert out["splits"]["dev"]["n"] + out["splits"]["holdout"]["n"] == out["summary"]["n"]
    ids = [d["id"] for d in out["detail"]]
    assert len(ids) == len(set(ids))
    # The pooled summary is what the CI gate and PR report read.
    assert json.loads((tmp_path / "t.json").read_text())["summary"]["n"] == out["summary"]["n"]


def test_only_ids_evaluates_exactly_the_named_cases(mock_cfg, tmp_path):
    """Re-running the handful a run got wrong shouldn't cost the whole gold set."""
    from eval.run_eval import run_eval

    want = ["e26", "m04"]
    cfg = mock_cfg.with_overrides(
        {
            "eval.only_ids": want,
            "eval.results_dir": str(tmp_path),
            "vector_store.persist_dir": str(tmp_path / "idx"),
        }
    )
    out = run_eval(cfg, tag="t")
    assert [d["id"] for d in out["detail"]] == want


def test_a_misspelled_only_id_fails_loudly(mock_cfg, tmp_path):
    """Quietly evaluating fewer cases than asked for would read as a result."""
    from eval.run_eval import run_eval

    cfg = mock_cfg.with_overrides(
        {
            "eval.only_ids": ["e26", "nope-not-a-real-id"],
            "eval.results_dir": str(tmp_path),
            "vector_store.persist_dir": str(tmp_path / "idx"),
        }
    )
    with pytest.raises(ValueError, match="nope-not-a-real-id"):
        run_eval(cfg, tag="t")


def test_split_report_respects_a_pinned_split(mock_cfg, tmp_path):
    """`--split holdout` runs only that half, with no `splits` comparison."""
    from eval.run_eval import run_eval_split_report

    cfg = mock_cfg.with_overrides(
        {
            "eval.split": "holdout",
            "eval.results_dir": str(tmp_path),
            "vector_store.persist_dir": str(tmp_path / "idx"),
        }
    )
    out = run_eval_split_report(cfg, tag="t")
    assert "splits" not in out
    assert 0 < out["summary"]["n"] < len(_gold())
