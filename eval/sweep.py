"""Grid-sweep pipeline thresholds and report the safe operating points.

The abstention gate trades two errors against each other: answer something
unsupported (hallucination) vs. decline something answerable (over-abstention).
The ablation reports both but never searches the space, so the shipped defaults
were never shown to be the right point on the curve.

This runs the full eval over a grid of config overrides and prints every point
plus the best one under a constraint — by default: minimize over-abstention
subject to hallucination staying at its current level and correct-abstention not
dropping. Nothing is applied automatically; the winning override is printed for
you to put in `config.yaml` deliberately.

    make sweep                      # the abstention grid
    python -m eval.sweep cache      # semantic-cache threshold grid

Grids live in `eval.sweeps` in config.yaml, so adding one needs no code change.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

# The pipeline is evaluated as it ships, so a sweep result is directly actionable.
_FULL_PIPELINE = {
    "retrieval.use_hybrid": True,
    "retrieval.use_rerank": True,
    "agent.enabled": True,
    "agent.critic": "both",
    "agent.crag.enabled": True,
}

# Metrics printed for every grid point, and how to read them.
_REPORT = [
    ("hallucination_rate", "hallu", False),
    ("over_abstention_rate", "overAbst", False),
    ("correct_abstention_rate", "abstOK", True),
    ("answer_correctness", "ansF1", True),
    ("faithfulness", "faith", True),
]


def _grid(spec: dict) -> list[dict]:
    """Cartesian product of {dotted_key: [values]} -> list of override dicts."""
    keys = list(spec)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(spec[k] for k in keys))]


def _label(ov: dict) -> str:
    return " ".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in ov.items())


def run_sweep(cfg, spec: dict, tag: str = "sweep") -> list[dict]:
    points = []
    for i, ov in enumerate(_grid(spec)):
        run_cfg = cfg.with_overrides({**_FULL_PIPELINE, **ov})
        summary = run_eval(run_cfg, tag=f"{tag}_{i}")["summary"]
        points.append({"overrides": ov, "label": _label(ov), "summary": summary})
        print(f"  {_label(ov):52s} " + "  ".join(
            f"{short}={summary.get(key, 0.0):.3f}" for key, short, _ in _REPORT
        ))
    return points


def pick_best(points: list[dict], baseline: dict) -> dict | None:
    """Lowest over-abstention among points that don't regress the safety metrics.

    'Safety' is deliberately strict: hallucination must not rise above the
    current value and correct-abstention must not fall below it. A sweep that
    trades hallucination for coverage would be optimizing the wrong thing.
    """
    safe = [
        p for p in points
        if p["summary"].get("hallucination_rate", 1.0) <= baseline.get("hallucination_rate", 0.0) + 1e-9
        and p["summary"].get("correct_abstention_rate", 0.0)
        >= baseline.get("correct_abstention_rate", 1.0) - 1e-9
    ]
    if not safe:
        return None
    return min(
        safe,
        key=lambda p: (
            p["summary"].get("over_abstention_rate", 1.0),
            -p["summary"].get("answer_correctness", 0.0),
        ),
    )


def main() -> int:
    args = [a for a in sys.argv[1:]]
    cfg_path = next((a for a in args if a.endswith((".yaml", ".yml"))), "config/config.yaml")
    name = next((a for a in args if not a.endswith((".yaml", ".yml"))), "abstention")
    cfg = load_config(cfg_path)

    spec = cfg.get(f"eval.sweeps.{name}")
    if not spec:
        available = list(cfg.get("eval.sweeps") or {})
        print(f"No sweep named '{name}' in eval.sweeps. Available: {available or '(none)'}")
        return 1

    print(f"Sweep '{name}' — {len(_grid(spec))} points, full pipeline:\n")
    points = run_sweep(cfg, spec, tag=f"sweep_{name}")

    # The current shipped config is the point to beat.
    print("\n  current shipped config:")
    baseline = run_eval(cfg.with_overrides(_FULL_PIPELINE), tag="sweep_current")["summary"]
    print("  " + "  ".join(f"{short}={baseline.get(key, 0.0):.3f}" for key, short, _ in _REPORT))

    best = pick_best(points, baseline)
    print()
    if best is None:
        print("No grid point improves on the shipped config without regressing "
              "hallucination or correct-abstention. Defaults stand.")
    elif best["summary"].get("over_abstention_rate", 1.0) >= baseline.get("over_abstention_rate", 1.0):
        print("Shipped config is already the best point on this grid.")
    else:
        print(f"Best safe point: {best['label']}")
        print(f"  over_abstention {baseline.get('over_abstention_rate', 0):.3f} -> "
              f"{best['summary'].get('over_abstention_rate', 0):.3f}  "
              f"(hallucination {best['summary'].get('hallucination_rate', 0):.3f}, "
              f"correct abstention {best['summary'].get('correct_abstention_rate', 0):.3f})")
        print("  Apply by editing config/config.yaml:")
        for k, v in best["overrides"].items():
            print(f"    {k} = {v}")

    out = Path(cfg.get("eval.results_dir", "eval/results")) / f"sweep_{name}.json"
    out.write_text(json.dumps(
        {"grid": spec, "points": points, "baseline": baseline,
         "best": best and best["overrides"]}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
