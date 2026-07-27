"""Unit tests for the CI gate's structured verdict and the PR-comment renderer.

All pure functions — no eval run, no network — so they stay fast and CI-safe.
"""

from eval.ci_gate import MetricRow, evaluate_gate, write_badges
from eval.pr_report import (
    MARKER,
    badge_color,
    badge_svg,
    render_comment,
    render_trend,
    sparkline,
)

BUDGETS = {"max_hallucination": 0.15, "min_correct_abstention": 0.5}


def _summary(**over) -> dict:
    base = {
        "n": 62,
        "hallucination_rate": 0.0,
        "over_abstention_rate": 0.104,
        "faithfulness": 1.0,
        "recall_at_k": 1.0,
        "mrr": 0.923,
        "correct_abstention_rate": 1.0,
        "answer_correctness": 0.323,
        "adversarial_robustness_rate": 1.0,
        "citation_precision": 0.875,
        "by_slice": {
            "easy": {"n": 40, "hallucination_rate": 0.0, "faithfulness": 1.0,
                     "recall_at_k": 1.0, "answer_correctness": 0.326},
            "unanswerable": {"n": 8, "hallucination_rate": 0.0, "faithfulness": 1.0},
        },
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- sparkline


def test_sparkline_monotonic_is_ascending():
    line = sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(line) == 5
    assert line[0] < line[-1]           # unicode blocks sort by height
    assert list(line) == sorted(line)


def test_sparkline_flat_series_is_single_level():
    line = sparkline([0.5, 0.5, 0.5])
    assert len(set(line)) == 1 and len(line) == 3


def test_sparkline_empty_is_empty():
    assert sparkline([]) == ""


# -------------------------------------------------------------- evaluate_gate


def test_gate_passes_when_metrics_match_baseline():
    ok, rows = evaluate_gate(_summary(), _summary(), BUDGETS, tolerance=0.05)
    assert ok
    assert all(r.ok for r in rows)
    assert {r.name for r in rows} >= {"hallucination_rate", "faithfulness", "recall_at_k"}


def test_gate_flags_regressed_higher_is_better_metric():
    ok, rows = evaluate_gate(_summary(faithfulness=0.80), _summary(), BUDGETS, tolerance=0.05)
    assert not ok
    row = next(r for r in rows if r.name == "faithfulness")
    assert row.regressed and row.delta < 0
    # Untouched metrics are unaffected.
    assert next(r for r in rows if r.name == "mrr").ok


def test_gate_flags_regressed_lower_is_better_metric():
    ok, rows = evaluate_gate(_summary(hallucination_rate=0.12), _summary(), BUDGETS, tolerance=0.05)
    assert not ok
    row = next(r for r in rows if r.name == "hallucination_rate")
    assert row.regressed and row.delta > 0
    assert not row.over_budget  # 0.12 is still inside the absolute budget


def test_gate_tolerance_absorbs_small_drift():
    ok, _ = evaluate_gate(_summary(faithfulness=0.97), _summary(), BUDGETS, tolerance=0.05)
    assert ok


def test_gate_enforces_absolute_budget_without_baseline():
    ok, rows = evaluate_gate(_summary(hallucination_rate=0.40), None, BUDGETS, tolerance=0.05)
    assert not ok
    row = next(r for r in rows if r.name == "hallucination_rate")
    assert row.over_budget and not row.regressed  # no baseline -> no REGRESSION rows
    assert all(r.baseline is None for r in rows)


def test_gate_passes_without_baseline_when_within_budget():
    ok, rows = evaluate_gate(_summary(), None, BUDGETS, tolerance=0.05)
    assert ok and all(r.baseline is None for r in rows)


# ------------------------------------------------------------- render_comment


def _rows(summary, baseline=None):
    return evaluate_gate(summary, baseline, BUDGETS, tolerance=0.05)


def test_render_comment_passing_has_marker_and_green_header():
    ok, rows = _rows(_summary(), _summary())
    body = render_comment(_summary(), _summary(), rows, ok, history=[])
    assert body.rstrip().endswith(MARKER)
    assert "regression gate PASSED" in body
    assert "🔴" not in body
    assert "hallucination **0.0%**" in body


def test_render_comment_failing_marks_the_offending_row():
    summary = _summary(faithfulness=0.60)
    ok, rows = _rows(summary, _summary())
    body = render_comment(summary, _summary(), rows, ok, history=[])
    assert "regression gate FAILED" in body
    failing = [ln for ln in body.splitlines() if ln.startswith("| faithfulness")]
    assert failing and "❌" in failing[0]
    passing = [ln for ln in body.splitlines() if ln.startswith("| mrr")]
    assert passing and "✅" in passing[0]


def test_render_comment_includes_slice_table():
    ok, rows = _rows(_summary(), _summary())
    body = render_comment(_summary(), _summary(), rows, ok, history=[])
    assert "### Per-slice (full pipeline)" in body
    assert "| easy |" in body and "| unanswerable |" in body


def test_render_comment_notes_missing_baseline():
    ok, rows = _rows(_summary(), None)
    body = render_comment(_summary(), None, rows, ok, history=[])
    assert "no baseline committed" in body.lower()
    assert "make update-baseline" in body


def test_render_comment_footer_carries_provenance_and_links():
    history = [{"git_sha": "a1b2c3d", "config_hash": "239e1859",
                "metrics": {"hallucination_rate": 0.0, "faithfulness": 1.0}}]
    ok, rows = _rows(_summary(), _summary())
    body = render_comment(_summary(), _summary(), rows, ok, history=history,
                          links={"run_url": "https://ci/run/1", "run_number": "123"})
    assert "`a1b2c3d`" in body and "`239e1859`" in body
    assert "[run #123](https://ci/run/1)" in body


# --------------------------------------------------------------- render_trend


def test_render_trend_needs_two_runs():
    one = [{"metrics": {"hallucination_rate": 0.3, "faithfulness": 0.7}}]
    assert "First recorded run" in render_trend(one)
    assert "```" not in render_trend(one)


def test_render_trend_shows_first_to_last():
    hist = [{"metrics": {"hallucination_rate": h, "faithfulness": f}}
            for h, f in [(0.306, 0.742), (0.306, 0.742), (0.048, 1.0), (0.0, 1.0)]]
    out = render_trend(hist)
    assert "0.306 → 0.000" in out
    assert "0.742 → 1.000" in out
    assert "```" in out


def test_render_trend_truncates_to_last_n_runs():
    hist = [{"metrics": {"hallucination_rate": i / 100, "faithfulness": 1.0}} for i in range(20)]
    out = render_trend(hist, n=8)
    line = next(ln for ln in out.splitlines() if ln.startswith("hallucination_rate"))
    assert "0.120 → 0.190" in line          # last 8 of 20
    assert len(line.split()[2]) == 8        # sparkline width == run count


# --------------------------------------------------------------------- badge


def test_badge_color_thresholds():
    assert badge_color(0.0) == badge_color(0.05)          # green up to 5%
    assert badge_color(0.06) != badge_color(0.0)          # yellow band
    assert badge_color(0.40) not in (badge_color(0.0), badge_color(0.06))


def test_badge_svg_is_self_contained_and_labelled():
    svg = badge_svg(0.0)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "0.0%" in svg and "hallucination" in svg
    # No external fetches — the point of committing it instead of using shields.
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in svg


def test_badge_svg_recolors_with_the_metric():
    assert badge_color(0.0) in badge_svg(0.0)
    assert badge_color(0.4) in badge_svg(0.4)
    assert badge_svg(0.0) != badge_svg(0.4)


def test_write_badges_emits_both_forms(tmp_path):
    import json as _json

    written = write_badges(0.0, tmp_path / "badge.json")
    assert [p.name for p in written] == ["badge.json", "badge.svg"]
    payload = _json.loads((tmp_path / "badge.json").read_text())
    assert payload["schemaVersion"] == 1 and payload["message"] == "0.0%"
    assert (tmp_path / "badge.svg").read_text().startswith("<svg")


def test_metric_row_ok_property():
    assert MetricRow("m", 1.0, 1.0, 0.0, True, regressed=False).ok
    assert not MetricRow("m", 1.0, 0.5, -0.5, True, regressed=True).ok
    assert not MetricRow("m", 1.0, 0.5, -0.5, True, regressed=False, over_budget=True).ok
