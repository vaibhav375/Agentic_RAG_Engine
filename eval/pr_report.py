"""Render the eval regression gate as a GitHub PR comment.

The gate already computes a full per-metric diff on every PR — but it lands in
CI logs, where nobody reads it. This module turns the same structured rows
(`ci_gate.evaluate_gate`) into one sticky markdown comment: verdict, per-metric
deltas vs. the committed baseline, the per-slice breakdown, and a sparkline
trend over recent recorded runs.

Everything here is a pure function of its arguments — no network, no git, no
filesystem — so it is fully testable in mock mode and deterministic in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval.report import _fmt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.ci_gate import MetricRow

# Hidden marker used to find-and-update the comment instead of posting a new one.
MARKER = "<!-- arag-eval-report -->"

_BLOCKS = "▁▂▃▄▅▆▇█"

# Metrics summarised in the one-line KPI header, as (key, label, decimals-as-pct).
_KPIS = [
    ("hallucination_rate", "hallucination", True),
    ("faithfulness", "faithfulness", False),
    ("correct_abstention_rate", "correct-abstention", False),
]

_TREND_METRICS = ("hallucination_rate", "faithfulness")

_SLICE_COLS = ["n", "hallucination_rate", "faithfulness", "recall_at_k",
               "answer_correctness", "robustness_pass"]
_SLICE_HEADERS = ["n", "hallu", "faith", "recall@k", "ans.F1", "robust"]


def sparkline(values: list[float]) -> str:
    """8-level unicode block sparkline, normalized over min/max. No dependencies.

    A flat series renders at a neutral mid level (there is no scale to imply).
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return _BLOCKS[3] * len(vals)
    return "".join(_BLOCKS[min(int((v - lo) / span * (len(_BLOCKS) - 1) + 0.5), len(_BLOCKS) - 1)]
                   for v in vals)


def badge_color(hallucination: float) -> str:
    pct = hallucination * 100
    return "#4c1" if pct <= 5 else ("#dfb317" if pct <= 15 else "#e05d44")


def badge_svg(hallucination: float, label: str = "hallucination") -> str:
    """A self-contained flat badge, committed to the repo and linked relatively
    from the README.

    Why not only the shields.io endpoint: shields fetches raw.githubusercontent
    anonymously, so an endpoint badge renders as a broken image for as long as
    the repo is private. A committed SVG referenced by relative path is served
    by GitHub itself, so it works in both visibility states with no external
    dependency at all.
    """
    value = f"{hallucination * 100:.1f}%"
    # ~6.2px per char at 11px DejaVu Sans, plus 10px padding each side.
    lw = int(len(label) * 6.2) + 20
    vw = int(len(value) * 6.6) + 20
    total = lw + vw
    color = badge_color(hallucination)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
        f'role="img" aria-label="{label}: {value}">'
        f"<title>{label}: {value}</title>"
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>'
        '<g clip-path="url(#r)">'
        f'<rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{vw}" height="20" fill="{color}"/>'
        f'<rect width="{total}" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        'font-size="11">'
        f'<text x="{lw / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>'
        f'<text x="{lw / 2:.0f}" y="14">{label}</text>'
        f'<text x="{lw + vw / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{value}</text>'
        f'<text x="{lw + vw / 2:.0f}" y="14">{value}</text>'
        "</g></svg>\n"
    )


def render_metric_table(rows: list[MetricRow], tolerance: float, has_baseline: bool) -> str:
    """Per-metric baseline → current table with direction glyphs and pass flags."""
    if not has_baseline:
        head = ("### Metrics (no baseline committed)\n\n"
                "> No baseline found — showing absolute values only. Run "
                "`make update-baseline` and commit the result to enable per-metric "
                "regression diffs.\n\n"
                "| Metric | This PR | |\n|---|---|---|\n")
        body = "".join(
            f"| {r.name} {'↑' if r.higher_is_better else '↓'} | {_fmt(r.current)} | "
            f"{'✅' if r.ok else '❌'} |\n"
            for r in rows
        )
        return head + body

    head = (f"### Metrics vs. baseline (tolerance {tolerance})\n\n"
            "| Metric | Baseline | This PR | Δ | |\n|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        arrow = "↑" if r.higher_is_better else "↓"
        body += (f"| {r.name} {arrow} | {_fmt(r.baseline)} | {_fmt(r.current)} | "
                 f"{r.delta:+.3f} | {'✅' if r.ok else '❌'} |\n")
    return head + body


def render_slice_table(summary: dict) -> str:
    """Per-slice breakdown, so a gain on easy questions can't mask a hard-slice drop."""
    by = summary.get("by_slice") or {}
    if not by:
        return ""
    lines = ["### Per-slice (full pipeline)\n",
             "| Slice | " + " | ".join(_SLICE_HEADERS) + " |",
             "|" + "|".join(["---"] * (len(_SLICE_COLS) + 1)) + "|"]
    for name in ("easy", "multi_hop", "unanswerable", "adversarial"):
        s = by.get(name)
        if not s:
            continue
        lines.append("| " + name + " | " + " | ".join(_fmt(s.get(c, "-")) for c in _SLICE_COLS) + " |")
    return "\n".join(lines) + "\n"


def render_trend(history: list[dict], metrics: tuple[str, ...] = _TREND_METRICS, n: int = 8) -> str:
    """Sparkline + `first → last` for each tracked metric over the last `n` runs."""
    series: dict[str, list[float]] = {}
    for m in metrics:
        vals = [h.get("metrics", {}).get(m) for h in history]
        vals = [v for v in vals if v is not None][-n:]
        if len(vals) >= 2:
            series[m] = vals
    if not series:
        return "### Trend\n\nFirst recorded run — no trend yet.\n"

    width = max(len(m) for m in series)
    lines = [f"### Trend (last {min(n, max(len(v) for v in series.values()))} recorded runs)\n", "```"]
    for m, vals in series.items():
        direction = "↓" if m in ("hallucination_rate", "over_abstention_rate") else "↑"
        lines.append(f"{m:<{width}} {direction}  {sparkline(vals)}   {vals[0]:.3f} → {vals[-1]:.3f}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def _header(ok: bool, summary: dict) -> str:
    verdict = ("## 🟢 Eval Report — regression gate PASSED" if ok
               else "## 🔴 Eval Report — regression gate FAILED")
    kpis = []
    for key, label, as_pct in _KPIS:
        v = summary.get(key)
        if v is None:
            continue
        kpis.append(f"{label} **{v * 100:.1f}%**" if as_pct else f"{label} **{v:.3f}**")
    n = summary.get("n", "?")
    return (f"{verdict}\n\n"
            f"Full agentic pipeline over the {n}-question gold set (mock mode, deterministic).\n"
            + " · ".join(kpis) + "\n")


def _footer(history: list[dict], links: dict | None) -> str:
    links = links or {}
    last = history[-1] if history else {}
    bits = []
    if last.get("git_sha"):
        bits.append(f"commit `{last['git_sha']}`")
    if last.get("config_hash"):
        bits.append(f"config `{last['config_hash']}`")
    if links.get("dashboard_url"):
        bits.append(f"[dashboard artifact]({links['dashboard_url']})")
    if links.get("run_url"):
        bits.append(f"[run #{links.get('run_number', '')}]({links['run_url']})")
    if not bits:
        return f"\n{MARKER}\n"
    return "\n<sub>" + " · ".join(bits) + "</sub>\n\n" + MARKER + "\n"


def render_comment(
    summary: dict,
    baseline: dict | None,
    rows: list[MetricRow],
    ok: bool,
    history: list[dict] | None = None,
    tolerance: float = 0.05,
    trend_runs: int = 8,
    links: dict | None = None,
) -> str:
    """Assemble the full sticky PR comment body. Ends with the hidden marker."""
    history = history or []
    parts = [
        _header(ok, summary),
        render_metric_table(rows, tolerance, has_baseline=baseline is not None),
        render_slice_table(summary),
        render_trend(history, n=trend_runs),
        _footer(history, links).lstrip("\n"),
    ]
    return "\n".join(p.strip("\n") + "\n" for p in parts if p.strip())
