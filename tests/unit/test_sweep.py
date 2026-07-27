"""The sweep's selection rule is the part that can quietly do damage.

A threshold search that maximizes coverage will happily trade hallucination away
— which is the one thing this project exists to drive down. `pick_best` therefore
only considers points that hold hallucination and correct-abstention at least as
good as the shipped config.
"""

from eval.sweep import _grid, _label, pick_best


def _point(over_abstention, hallucination=0.0, correct_abstention=1.0, ans_f1=0.32, **ov):
    return {
        "overrides": ov,
        "label": _label(ov),
        "summary": {
            "over_abstention_rate": over_abstention,
            "hallucination_rate": hallucination,
            "correct_abstention_rate": correct_abstention,
            "answer_correctness": ans_f1,
        },
    }


BASE = {"hallucination_rate": 0.0, "correct_abstention_rate": 1.0, "over_abstention_rate": 0.104}


def test_grid_is_the_full_cartesian_product():
    grid = _grid({"a.x": [1, 2], "b.y": [3, 4, 5]})
    assert len(grid) == 6
    assert {"a.x": 2, "b.y": 4} in grid


def test_picks_the_lowest_over_abstention_among_safe_points():
    best = pick_best(
        [_point(0.104, **{"agent.support_threshold": 0.5}),
         _point(0.021, **{"agent.support_threshold": 0.3}),
         _point(0.062, **{"agent.support_threshold": 0.7})],
        BASE,
    )
    assert best["overrides"] == {"agent.support_threshold": 0.3}


def test_refuses_to_trade_hallucination_for_coverage():
    """A point with zero over-abstention but any hallucination must lose."""
    best = pick_best(
        [_point(0.000, hallucination=0.05, **{"agent.support_threshold": 0.1}),
         _point(0.083, **{"agent.support_threshold": 0.5})],
        BASE,
    )
    assert best["overrides"] == {"agent.support_threshold": 0.5}


def test_refuses_to_trade_correct_abstention_for_coverage():
    best = pick_best(
        [_point(0.000, correct_abstention=0.75, **{"agent.support_threshold": 0.1}),
         _point(0.083, **{"agent.support_threshold": 0.5})],
        BASE,
    )
    assert best["overrides"] == {"agent.support_threshold": 0.5}


def test_no_safe_point_returns_none():
    assert pick_best([_point(0.0, hallucination=0.2)], BASE) is None


def test_answer_correctness_breaks_ties():
    best = pick_best(
        [_point(0.05, ans_f1=0.30, **{"agent.support_threshold": 0.5}),
         _point(0.05, ans_f1=0.40, **{"agent.support_threshold": 0.3})],
        BASE,
    )
    assert best["overrides"] == {"agent.support_threshold": 0.3}
