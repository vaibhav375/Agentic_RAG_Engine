"""Ablation study — the single most important artifact for this project.

Runs the pipeline in cumulative configurations and reports how each component
moves the metrics, so the contribution of hybrid retrieval, reranking, contextual
enrichment, self-correction, the router, and the cache is each visible.

    dense-only  ->  +hybrid  ->  +rerank  ->  +contextual  ->  +self-correction
                ->  +router  ->  +cache

Writes eval/results/*.json per config and assembles RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from arag.common.config import load_config
from arag.engine import build_components
from arag.ingest.index import build_index
from eval.report import write_results_md
from eval.run_eval import run_eval

# (tag, human label, cumulative config overrides)
ABLATION_MATRIX: list[tuple[str, str, dict]] = [
    (
        "baseline",
        "Dense-only baseline",
        {
            "retrieval.use_hybrid": False,
            "retrieval.use_rerank": False,
            "chunking.contextual_enrichment": False,
            "agent.enabled": False,
            "agent.router.enabled": False,
            "cache.enabled": False,
        },
    ),
    ("hybrid", "+ Hybrid (BM25 + RRF)", {"retrieval.use_hybrid": True}),
    ("rerank", "+ Cross-encoder rerank", {"retrieval.use_rerank": True}),
    ("contextual", "+ Contextual chunk enrichment", {"chunking.contextual_enrichment": True}),
    (
        "selfcorrect",
        "+ Self-correction (LLM+NLI critic)",
        {"agent.enabled": True, "agent.critic": "both"},
    ),
    ("crag", "+ CRAG answerability gate", {"agent.crag.enabled": True}),
    ("router", "+ Query router", {"agent.router.enabled": True}),
    ("cache", "+ Semantic cache", {"cache.enabled": True}),
]


def _cumulative_overrides() -> list[tuple[str, str, dict]]:
    acc: dict = {}
    out = []
    for tag, label, ov in ABLATION_MATRIX:
        acc = {**acc, **ov}
        out.append((tag, label, dict(acc)))
    return out


def run_ablation(cfg) -> list[dict]:
    results = []
    for tag, label, overrides in _cumulative_overrides():
        run_cfg = cfg.with_overrides(overrides)
        if tag == "cache":
            res = _run_with_warm_cache(run_cfg, tag)
        else:
            res = run_eval(run_cfg, tag=tag)
        res["label"] = label
        results.append(res)
        _print_row(label, res["summary"])
    return results


def _run_with_warm_cache(cfg, tag: str) -> dict:
    """Cache config: run the gold set cold (populate) then warm (measure hits),
    and quantify false hits by comparing per-question correctness across passes."""
    store = build_index(cfg)
    comp = build_components(cfg, store=store)

    cold = run_eval(cfg, tag=f"{tag}_cold", comp=comp)
    warm = run_eval(cfg, tag=tag, comp=comp)

    # False-hit check: among warm-pass cache hits, did any answer's correctness
    # drop vs the cold pass? (conservative-threshold cache should give ~0.)
    cold_by_id = {d["id"]: d for d in cold["detail"]}
    false_hits = 0
    hits = 0
    for d in warm["detail"]:
        if d["from_cache"]:
            hits += 1
            c = cold_by_id.get(d["id"], {})
            if d["metrics"].get("answer_correctness", 1.0) < c.get("metrics", {}).get(
                "answer_correctness", 1.0
            ):
                false_hits += 1
    warm["summary"]["cache_warm_hits"] = hits
    warm["summary"]["cache_false_hits"] = false_hits
    if comp.cache is not None:
        warm["summary"]["cache_stats"] = comp.cache.stats()
    return warm


def _print_row(label: str, s: dict) -> None:
    print(
        f"{label:38s} hallu={s['hallucination_rate']:.3f} "
        f"faith={s['faithfulness']:.3f} ctxP={s['context_precision']:.3f} "
        f"ctxR={s['context_recall']:.3f} correctAbst={s['correct_abstention_rate']:.3f} "
        f"p95={s['latency_p95_ms']:.1f}ms"
    )


def main():
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    results = run_ablation(cfg)
    results_dir = Path(cfg.get("eval.results_dir", "eval/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ablation.json").write_text(json.dumps(results, indent=2))

    # Fold judge calibration + selective-prediction analysis into the report.
    try:
        from eval.calibrate_judge import calibrate

        calibration = calibrate(cfg)
    except Exception:
        calibration = None
    try:
        from eval.selective import run_selective

        selective = run_selective(cfg)
    except Exception:
        selective = None

    md_path = write_results_md(
        results, out_path="RESULTS.md", calibration=calibration, selective=selective
    )
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
