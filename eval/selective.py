"""Selective prediction: risk–coverage analysis of the abstention mechanism.

An abstaining ("selective") QA system trades coverage (how often it answers) for
risk (how often an answer is wrong). Sweeping the answer/abstain confidence
threshold traces a risk–coverage curve; the area under it (lower = better) and the
maximum coverage attainable at zero risk summarize how good the abstention signal
is — a standard way to evaluate selective classifiers that most RAG demos never do.

Confidence = the CRAG IDF-weighted retrieval coverage. "Risk" here = answering a
question that is unanswerable/adversarial (i.e., should have been declined).

Outputs eval/results/risk_coverage.png and a summary dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arag.agent.retrieval_grader import idf_coverage
from arag.common.config import load_config
from arag.engine import build_components, retrieve_contexts
from arag.ingest.index import build_index
from eval.build_gold_set import load_gold


def _confidence_labels(cfg) -> tuple[list[float], list[int]]:
    """Return (confidence, answerable_label) per gold question under full retrieval."""
    cfg = cfg.with_overrides({"retrieval.use_hybrid": True, "retrieval.use_rerank": True})
    store = build_index(cfg)
    comp = build_components(cfg, store=store)
    gold = load_gold(cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl"))
    conf, ans_label = [], []
    for g in gold:
        _, ctx = retrieve_contexts(comp, g.question)
        conf.append(idf_coverage(store, g.question, ctx))
        ans_label.append(1 if g.is_answerable else 0)
    return conf, ans_label


def risk_coverage(conf: list[float], answerable: list[int]) -> dict:
    thresholds = np.linspace(0.0, 1.0, 101)
    points = []  # (coverage, risk, threshold)
    for t in thresholds:
        answered = [i for i, c in enumerate(conf) if c >= t]
        coverage = len(answered) / len(conf)
        if answered:
            # risk = fraction of answered that should have abstained (not answerable)
            risk = sum(1 - answerable[i] for i in answered) / len(answered)
        else:
            risk = 0.0
        points.append((round(coverage, 4), round(risk, 4), round(float(t), 3)))

    # AUC of risk over coverage (sort by coverage ascending, trapezoid).
    pts = sorted(points, key=lambda p: p[0])
    covs = [p[0] for p in pts]
    risks = [p[1] for p in pts]
    # np.trapz was removed in NumPy 2.0; prefer trapezoid and fall back lazily
    # (a plain `getattr(np, "trapezoid", np.trapz)` would eval np.trapz eagerly
    # and raise on NumPy 2.x before the fallback could apply).
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(_trap(risks, covs)) if len(covs) > 1 else 0.0

    max_safe_coverage = max((c for c, r, _ in points if r == 0.0), default=0.0)
    base_risk = sum(1 - a for a in answerable) / len(answerable)  # risk at full coverage
    return {
        "risk_coverage_auc": round(auc, 4),
        "max_safe_coverage": round(max_safe_coverage, 4),
        "risk_at_full_coverage": round(base_risk, 4),
        "answerable_fraction": round(sum(answerable) / len(answerable), 4),
        "points": points,
    }


def _plot(points, out: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pts = sorted(points, key=lambda p: p[0])
    covs = [p[0] for p in pts]
    risks = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(covs, risks, marker="o", ms=3, color="#8e44ad")
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Risk (answered-but-should-abstain rate)")
    ax.set_title("Selective prediction: risk–coverage curve")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, max(risks) * 1.1))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def run_selective(cfg) -> dict:
    conf, answerable = _confidence_labels(cfg)
    summary = risk_coverage(conf, answerable)
    out_dir = Path(cfg.get("eval.results_dir", "eval/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["plot"] = _plot(summary["points"], out_dir / "risk_coverage.png")
    (out_dir / "selective.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    s = run_selective(cfg)
    print(json.dumps({k: v for k, v in s.items() if k != "points"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
