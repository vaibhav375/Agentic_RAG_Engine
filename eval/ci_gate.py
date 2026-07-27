"""CI regression gate: run the full agentic pipeline over the gold set and fail
the build if quality regresses. Wired into GitHub Actions so prompt/model/config
changes can't silently degrade answers.

Two consumers share one source of truth (`evaluate_gate`):
  * the CLI gate — prints the diff and exits non-zero on regression;
  * the PR bot — `--report PATH` renders the same rows as a markdown comment
    (see `eval/pr_report.py`), so what CI enforces is exactly what the PR shows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running both as `python -m eval.ci_gate` and `python eval/ci_gate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

# Metrics where a DROP vs. baseline is a regression, and where a RISE is.
_HIGHER_IS_BETTER = ["faithfulness", "recall_at_k", "mrr", "correct_abstention_rate",
                     "answer_correctness", "adversarial_robustness_rate", "citation_precision"]
_LOWER_IS_BETTER = ["hallucination_rate", "over_abstention_rate"]


@dataclass
class MetricRow:
    """One metric's baseline → current comparison, plus why it did or didn't pass."""

    name: str
    baseline: float | None
    current: float
    delta: float               # current - baseline (0.0 when there is no baseline)
    higher_is_better: bool
    regressed: bool            # moved the wrong way by more than `tolerance`
    over_budget: bool = False  # violates an absolute budget (independent of baseline)

    @property
    def ok(self) -> bool:
        return not (self.regressed or self.over_budget)


def load_baseline(path: str | Path) -> dict | None:
    """Read a committed baseline summary. Accepts a bare summary or a full
    `run_eval` wrapper. Returns None when the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    base = json.loads(p.read_text())
    if "summary" in base and "hallucination_rate" not in base:
        base = base["summary"]
    return base


def evaluate_gate(
    summary: dict,
    baseline: dict | None,
    budgets: dict,
    tolerance: float,
) -> tuple[bool, list[MetricRow]]:
    """Compare a run against absolute budgets and (optionally) a baseline.

    Returns `(ok, rows)`. Rows are ordered lower-is-better first so the headline
    hallucination_rate leads the table.
    """
    max_hallu = float(budgets.get("max_hallucination", 1.0))
    min_abst = float(budgets.get("min_correct_abstention", 0.0))
    over_budget = {
        "hallucination_rate": summary.get("hallucination_rate", 0.0) > max_hallu,
        "correct_abstention_rate": summary.get("correct_abstention_rate", 0.0) < min_abst,
    }

    rows: list[MetricRow] = []
    for name, higher_is_better in (
        [(m, False) for m in _LOWER_IS_BETTER] + [(m, True) for m in _HIGHER_IS_BETTER]
    ):
        current = float(summary.get(name, 0.0))
        base_val = None if baseline is None else float(baseline.get(name, 0.0))
        delta = 0.0 if base_val is None else round(current - base_val, 4)
        # Regression = moved the wrong way by more than the tolerance.
        drift = delta if not higher_is_better else -delta
        rows.append(
            MetricRow(
                name=name,
                baseline=base_val,
                current=current,
                delta=delta,
                higher_is_better=higher_is_better,
                regressed=base_val is not None and drift > tolerance,
                over_budget=over_budget.get(name, False),
            )
        )
    return all(r.ok for r in rows), rows


def _gate_defaults(cfg) -> dict:
    """Gate budgets/paths come from config (`eval.ci_gate`); CLI flags override."""
    return {
        "max_hallucination": float(cfg.get("eval.ci_gate.max_hallucination", 0.15)),
        "min_correct_abstention": float(cfg.get("eval.ci_gate.min_correct_abstention", 0.5)),
        "tolerance": float(cfg.get("eval.ci_gate.tolerance", 0.05)),
        "baseline_path": cfg.get("eval.ci_gate.baseline_path", "eval/results/ci_baseline.json"),
        "trend_runs": int(cfg.get("eval.ci_gate.trend_runs", 8)),
    }


def _print_rows(rows: list[MetricRow], baseline_path: str, tolerance: float, has_baseline: bool) -> None:
    if not has_baseline:
        print(f"\n(no baseline at {baseline_path}; run with --update-baseline to create one)")
        return
    print(f"\nRegression diff vs {baseline_path} (tolerance {tolerance}):")
    for r in rows:
        flag = "REGRESSION" if r.regressed else ("OVER BUDGET" if r.over_budget else "ok")
        print(f"  {r.name:28s} {r.baseline:.3f} -> {r.current:.3f}  [{flag}]")


def _badge_payload(hallucination: float) -> dict:
    """shields.io endpoint badge: green ≤5%, yellow ≤15%, red otherwise."""
    pct = hallucination * 100
    color = "brightgreen" if pct <= 5 else ("yellow" if pct <= 15 else "red")
    return {
        "schemaVersion": 1,
        "label": "hallucination",
        "message": f"{pct:.1f}%",
        "color": color,
    }


def write_badges(hallucination: float, json_path: str | Path) -> list[Path]:
    """Write both badge forms next to each other: the shields.io endpoint JSON
    (for public repos) and a self-contained SVG the README links relatively
    (works while the repo is private, since GitHub serves it itself)."""
    from eval.pr_report import badge_svg

    json_path = Path(json_path)
    svg_path = json_path.with_suffix(".svg")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_badge_payload(hallucination), indent=2) + "\n")
    svg_path.write_text(badge_svg(hallucination))
    return [json_path, svg_path]


def _github_links() -> dict:
    """Best-effort run/commit links from the Actions environment (empty locally)."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    links: dict = {}
    if repo and run_id:
        links["run_url"] = f"{server}/{repo}/actions/runs/{run_id}"
        links["run_number"] = os.environ.get("GITHUB_RUN_NUMBER", run_id)
    return links


def _write_github_output(ok: bool, hallucination: float) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a") as f:
        f.write(f"gate_passed={'true' if ok else 'false'}\n")
        f.write(f"hallucination={hallucination * 100:.1f}%\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--max-hallucination", type=float, default=None)
    ap.add_argument("--min-correct-abstention", type=float, default=None)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--baseline", default=None,
                    help="Committed baseline summary to diff against (optional)")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="Allowed regression per metric vs. baseline")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Write the current summary as the new baseline and exit 0")
    ap.add_argument("--report", default=None,
                    help="Write the PR-comment markdown body to this path")
    ap.add_argument("--badge", default=None,
                    help="Write a shields.io endpoint badge JSON to this path")
    ap.add_argument("--github-output", action="store_true",
                    help="Append gate_passed/hallucination to $GITHUB_OUTPUT")
    args = ap.parse_args()

    cfg = load_config(args.config).with_overrides(
        {
            # Evaluate the full pipeline the way it ships.
            "retrieval.use_hybrid": True,
            "retrieval.use_rerank": True,
            "agent.enabled": True,
            "agent.critic": "both",
            "agent.crag.enabled": True,
        }
    )
    if args.subset:
        cfg = cfg.with_overrides({"eval.subset": args.subset})

    defaults = _gate_defaults(cfg)
    budgets = {
        "max_hallucination": args.max_hallucination
        if args.max_hallucination is not None else defaults["max_hallucination"],
        "min_correct_abstention": args.min_correct_abstention
        if args.min_correct_abstention is not None else defaults["min_correct_abstention"],
    }
    tolerance = args.tolerance if args.tolerance is not None else defaults["tolerance"]
    baseline_path = args.baseline or defaults["baseline_path"]

    summary = run_eval(cfg, tag="ci_gate")["summary"]
    hallu = summary["hallucination_rate"]
    abst = summary["correct_abstention_rate"]
    print(f"hallucination_rate={hallu:.3f} (budget {budgets['max_hallucination']})")
    print(f"correct_abstention_rate={abst:.3f} (min {budgets['min_correct_abstention']})")

    if args.update_baseline:
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        Path(baseline_path).write_text(json.dumps(summary, indent=2))
        print(f"wrote baseline -> {baseline_path}")
        return 0

    baseline = load_baseline(baseline_path)
    ok, rows = evaluate_gate(summary, baseline, budgets, tolerance)
    _print_rows(rows, baseline_path, tolerance, has_baseline=baseline is not None)

    if args.report:
        from eval.pr_report import render_comment
        from eval.registry import load_history

        body = render_comment(
            summary=summary,
            baseline=baseline,
            rows=rows,
            ok=ok,
            history=load_history(),
            tolerance=tolerance,
            trend_runs=defaults["trend_runs"],
            links=_github_links(),
        )
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(body)
        print(f"wrote PR report -> {args.report}")

    if args.badge:
        written = write_badges(hallu, args.badge)
        print("wrote badges -> " + ", ".join(str(p) for p in written))

    if args.github_output:
        _write_github_output(ok, hallu)

    if not ok:
        print("REGRESSION GATE FAILED", file=sys.stderr)
        return 1
    print("regression gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
